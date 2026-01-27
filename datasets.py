import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from scipy.spatial.transform import Rotation as R


def extract_yaw(quat: np.ndarray) -> float:
    """
    Извлекает yaw (вращение вокруг Z) из кватерниона.
    Возвращает значение в радианах.
    """
    quat = quat / np.linalg.norm(quat)
    rot = R.from_quat(quat)
    yaw = rot.as_euler('zyx', degrees=False)[0]
    return yaw


class DroneTrajectoryDataset(Dataset):
    """
    Датасет траекторий дрона с локальной эго-СК.
    Позиции и yaw пересчитываются относительно последней точки контекста.
    Добавлена аугментация: случайный линейный дрейф для таргет-позиций,
    масштабируемый квадратом целевой частоты.
    В target добавлены каналы накопленного дрейфа (dx, dy, dz).
    """
    def __init__(
        self,
        csv_path: str,
        context_points: int = 5,
        target_points: int = 21,
        target_offset_points: int = 5,
        goal_offset_points: int = 10,
        stride_points: int = 4,
        original_freq: int = 200,
        target_freq: int = 10,
        pos_columns=(' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]'),
        quat_columns=(' q_RS_x []', ' q_RS_y []', ' q_RS_z []', ' q_RS_w []'),
        time_column='#timestamp',
        norm_margin: float = 1.1,
        norm_min_scale: float = 1.0,
        # Параметры аугментации дрейфом
        drift_range: tuple[float, float] = (-2,2),     # "сырой" дрейф за шаг при 1 Гц
    ):
        self.norm_margin = norm_margin
        self.norm_min_scale = norm_min_scale
        self.context_points = context_points
        self.target_points = target_points
        self.target_offset_points = target_offset_points
        self.goal_offset_points = goal_offset_points
        self.stride_points = stride_points
        self.original_freq = original_freq
        self.target_freq = target_freq
        
        # Параметры дрейф-аугментации
        self.drift_range = drift_range
        self.drift_scale_factor =0.5*(1/float(target_freq)) ** 2      # основной множитель

        # -------------------- Load CSV --------------------
        df = pd.read_csv(csv_path)
        rename = {time_column: 'time'}
        for i, c in enumerate(pos_columns):
            rename[c] = ['x', 'y', 'z'][i]
        for i, c in enumerate(quat_columns):
            rename[c] = ['qx', 'qy', 'qz', 'qw'][i]
        df = df.rename(columns=rename)
        df['time'] = df['time'] * 1e-9  # ns -> s

        # Нормализация кватернионов
        q = df[['qx', 'qy', 'qz', 'qw']].values
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        df[['qx', 'qy', 'qz', 'qw']] = q
        self.df = df.reset_index(drop=True)

        # -------------------- Downsampling --------------------
        self.downsamp = self.original_freq // self.target_freq
        self.max_needed = (
            (context_points - 1) * self.downsamp
            + target_offset_points
            + (target_points - 1) * self.downsamp
            + goal_offset_points
        )

        # -------------------- Индексы сэмплов --------------------
        self.indices = []
        idx = 0
        N = len(self.df)
        while idx + self.max_needed < N:
            self.indices.append(idx)
            idx += self.stride_points

    def __len__(self):
        return len(self.indices)

    def _safe_rows(self, idxs):
        idxs = [i for i in idxs if i < len(self.df)]
        return self.df.iloc[idxs]

    def _to_local_frame(self, pos_global, ego_pos, R_local):
        return R_local.apply(pos_global - ego_pos)

    def _generate_sample(
        self,
        start_idx: int,
        custom_raw_drift: np.ndarray | None = None,  # [3] — кастомные dx, dy, dz (сырые значения)
    ):
        # ================== EGO ==================
        ego_idx = start_idx + (self.context_points - 1) * self.downsamp
        ego_row = self.df.iloc[ego_idx]
        ego_pos = ego_row[['x', 'y', 'z']].values
        ego_quat = ego_row[['qx', 'qy', 'qz', 'qw']].values
        R_local = R.from_euler('z', extract_yaw(ego_quat)).inv()
        # ================== CONTEXT ==================
        ctx_idxs = [start_idx + i * self.downsamp for i in range(self.context_points)]
        ctx = self._safe_rows(ctx_idxs)[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].copy()
        ctx_pos_local = self._to_local_frame(ctx[['x', 'y', 'z']].values, ego_pos, R_local)
        ctx_yaw = np.array([extract_yaw(q) for q in ctx[['qx','qy','qz','qw']].values])
        ctx_yaw_local = ctx_yaw - ctx_yaw[-1]
        ctx_sincos = np.stack([np.sin(ctx_yaw_local), np.cos(ctx_yaw_local)], axis=1)
        # ================== TARGET ==================
        tgt_start = ego_idx + self.target_offset_points
        tgt_idxs = [tgt_start + i * self.downsamp for i in range(self.target_points)]
        tgt = self._safe_rows(tgt_idxs)[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].copy()
        tgt_pos_local = self._to_local_frame(tgt[['x', 'y', 'z']].values, ego_pos, R_local)
        tgt_yaw = np.array([extract_yaw(q) for q in tgt[['qx', 'qy', 'qz', 'qw']].values])
        tgt_yaw_local = tgt_yaw - ctx_yaw[-1]
        tgt_sincos = np.stack([np.sin(tgt_yaw_local), np.cos(tgt_yaw_local)], axis=1)
        # ================== АУГМЕНТАЦИЯ: ЛИНЕЙНЫЙ ДРЕЙФ ==================
        if custom_raw_drift is not None:
            # Используем то, что ты явно задал
            raw_drift = np.array(custom_raw_drift, dtype=np.float32)
            assert raw_drift.shape == (3,), "custom_raw_drift должен быть вектором [3]"
        else:
            # Случайный в диапазоне (как раньше)
            raw_drift = np.random.uniform(
                self.drift_range[0],
                self.drift_range[1],
                size=3
            )

        # Масштабируем фиксированным коэффициентом (target_freq ** 2)
        scaled_drift = raw_drift * self.drift_scale_factor
        # print(scaled_drift)

        i = np.arange(self.target_points)[:, None]  # [T, 1]
        displacement = scaled_drift * i             # [T, 3] — накопленное смещение
        # print(displacement)
        # print('-'*10)
        tgt_pos_augmented = tgt_pos_local - displacement
        # print(tgt_pos_local)
        # print('-'*10)
        # print(tgt_pos_augmented)
        # ================== GOAL ==================
        goal_start = tgt_idxs[-1]
        goal_end = min(goal_start + self.goal_offset_points, len(self.df))
        goal_idx = np.random.randint(goal_start, goal_end)
        goal_row = self.df.iloc[goal_idx]
        goal_pos_local = self._to_local_frame(goal_row[['x', 'y', 'z']].values, ego_pos, R_local)
        return {
            'ctx_pos': ctx_pos_local,
            'ctx_sincos': ctx_sincos,
            'tgt_pos': tgt_pos_local,          # идеальная
            'tgt_sincos': tgt_sincos,
            'tgt_drift': tgt_pos_augmented,    # сдвинутая
            'goal_pos': goal_pos_local,
            'drift': scaled_drift
    }
    def _normalize_sample(self, raw_sample: dict) -> dict:
        ctx_pos = raw_sample['ctx_pos']
        tgt_pos = raw_sample['tgt_pos']
        tgt_drift = raw_sample['tgt_drift']
        goal_pos = raw_sample['goal_pos']
        drift = raw_sample['drift']

        # Нормализуем всё одним масштабом (позиции + дрейф)
        all_pos = np.concatenate([ctx_pos, tgt_pos, tgt_drift, goal_pos[None, :]], axis=0)
        max_abs_per_axis = np.max(np.abs(all_pos), axis=0)
        scale_per_axis = np.maximum(self.norm_min_scale, max_abs_per_axis * self.norm_margin)

        ctx_pos_norm = ctx_pos / scale_per_axis
        tgt_pos_norm = tgt_pos / scale_per_axis
        tgt_drift_norm = tgt_drift / scale_per_axis     # дрейф нормализуется тем же коэффициентом!
        goal_pos_norm = goal_pos / scale_per_axis

        context = np.concatenate([ctx_pos_norm, raw_sample['ctx_sincos']], axis=1)
        
        # Target: 3 (xyz) + 2 (sincos) + 3 (drift) = 8 каналов
        target = np.concatenate([
            tgt_pos_norm,
            raw_sample['tgt_sincos'],
            tgt_drift_norm
        ], axis=1)

        return {
            'context': torch.tensor(context, dtype=torch.float32),          # [C, 5]
            'target': torch.tensor(target,  dtype=torch.float32),          # [T, 8]
            'goal': torch.tensor(goal_pos_norm, dtype=torch.float32),
            'drift': torch.tensor(drift, dtype=torch.float32),
            'scale': torch.tensor(scale_per_axis, dtype=torch.float32)
        }

    def __getitem__(
            self,
            idx,
            custom_raw_drift: np.ndarray | None = None,  # [3] — твои значения dx, dy, dz
            normalize: bool = True
        ):
            start_idx = self.indices[idx]
            raw_sample = self._generate_sample(start_idx, custom_raw_drift=custom_raw_drift)
            
            if normalize:
                return self._normalize_sample(raw_sample)
            else:
                # Без нормализации
                context = np.concatenate([raw_sample['ctx_pos'], raw_sample['ctx_sincos']], axis=1)
                target = np.concatenate([
                    raw_sample['tgt_pos'],
                    raw_sample['tgt_sincos'],
                    raw_sample['tgt_drift']
                ], axis=1)
                return {
                    'context': torch.tensor(context, dtype=torch.float32),
                    'target': torch.tensor(target, dtype=torch.float32),
                    'goal': torch.tensor(raw_sample['goal_pos'], dtype=torch.float32),
                    'drift': torch.tensor(raw_sample['drift'], dtype=torch.float32),
                    'scale': torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
                }
    def get_sample(self, idx, custom_raw_drift=None, normalize=True):
        """Удобный метод для передачи кастомных параметров"""
        return self.__getitem__(idx, custom_raw_drift=custom_raw_drift, normalize=normalize)


def build_dataset(cfgs):
    return ConcatDataset([DroneTrajectoryDataset(**cfg) for cfg in cfgs])
        