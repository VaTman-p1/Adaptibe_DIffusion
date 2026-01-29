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
    Поддерживает downsampling и stride.
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
        norm_min_scale: float = 1.0      
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

        # -------------------- Load CSV --------------------
        df = pd.read_csv(csv_path)
        rename = {time_column: 'time'}
        for i, c in enumerate(pos_columns):
            rename[c] = ['x','y','z'][i]
        for i, c in enumerate(quat_columns):
            rename[c] = ['qx','qy','qz','qw'][i]
        df = df.rename(columns=rename)
        df['time'] = df['time'] * 1e-9  # ns -> s

        # Нормализация кватернионов
        q = df[['qx','qy','qz','qw']].values
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        df[['qx','qy','qz','qw']] = q

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
        # stride учитывается в исходных точках (original_freq)
        self.indices = []
        idx = 0
        N = len(self.df)
        # ooo = 0
        while idx + self.max_needed < N:
            # ooo+=1
            self.indices.append(idx)
            idx += self.stride_points  # можно менять stride_points для прореживания
        # print(ooo)
    def __len__(self):
        return len(self.indices)

    def _safe_rows(self, idxs):
        idxs = [i for i in idxs if i < len(self.df)]
        return self.df.iloc[idxs]

    def _to_local_frame(self, pos_global, ego_pos, R_local):
        return R_local.apply(pos_global - ego_pos)

    def _generate_sample(self, start_idx: int):
        # ================== EGO ==================
        ego_idx = start_idx + (self.context_points - 1) * self.downsamp
        ego_row = self.df.iloc[ego_idx]
        ego_pos = ego_row[['x','y','z']].values
        ego_quat = ego_row[['qx','qy','qz','qw']].values
        R_local = R.from_euler('z', extract_yaw(ego_quat)).inv()

        # ================== CONTEXT ==================
        ctx_idxs = [start_idx + i * self.downsamp for i in range(self.context_points)]
        ctx = self._safe_rows(ctx_idxs)[['x','y','z','qx','qy','qz','qw']].copy()
        ctx_pos_local = self._to_local_frame(ctx[['x','y','z']].values, ego_pos, R_local)
        ctx_yaw = np.array([extract_yaw(q) for q in ctx[['qx','qy','qz','qw']].values])
        ctx_yaw_local = ctx_yaw - ctx_yaw[-1]
        ctx_sincos = np.stack([np.sin(ctx_yaw_local), np.cos(ctx_yaw_local)], axis=1)

        # ================== TARGET ==================
        tgt_start = ego_idx + self.target_offset_points
        tgt_idxs = [tgt_start + i * self.downsamp for i in range(self.target_points)]
        tgt = self._safe_rows(tgt_idxs)[['x','y','z','qx','qy','qz','qw']].copy()
        tgt_pos_local = self._to_local_frame(tgt[['x','y','z']].values, ego_pos, R_local)
        tgt_yaw = np.array([extract_yaw(q) for q in tgt[['qx','qy','qz','qw']].values])
        tgt_yaw_local = tgt_yaw - ctx_yaw[-1]
        tgt_sincos = np.stack([np.sin(tgt_yaw_local), np.cos(tgt_yaw_local)], axis=1)

        # ================== GOAL ==================
        goal_start = tgt_idxs[-1]
        goal_end = min(goal_start + self.goal_offset_points, len(self.df))
        goal_idx = np.random.randint(goal_start, goal_end)
        goal_row = self.df.iloc[goal_idx]
        goal_pos_local = self._to_local_frame(goal_row[['x','y','z']].values, ego_pos, R_local)

        return {
                    'ctx_pos': ctx_pos_local,          # np.array [C, 3]
                    'ctx_sincos': ctx_sincos,          # np.array [C, 2]
                    'tgt_pos': tgt_pos_local,          # np.array [T, 3]
                    'tgt_sincos': tgt_sincos,          # np.array [T, 2]
                    'goal_pos': goal_pos_local         # np.array [3]
                }
    def _normalize_sample(self, raw_sample: dict) -> dict:
        ctx_pos = raw_sample['ctx_pos']      # [C, 3]
        tgt_pos = raw_sample['tgt_pos']      # [T, 3]
        goal_pos = raw_sample['goal_pos']    # [3]

        # Все позиции вместе
        all_pos = np.concatenate([ctx_pos, tgt_pos, goal_pos[None, :]], axis=0)  # [total, 3]

        # Max abs по каждой оси отдельно
        max_abs_per_axis = np.max(np.abs(all_pos), axis=0)  # [3]

        # Scale по осям с margin и min
        scale_per_axis = np.maximum(self.norm_min_scale, max_abs_per_axis * self.norm_margin)  # [3]

        # Нормализуем (broadcasting)
        ctx_pos_norm = ctx_pos / scale_per_axis
        tgt_pos_norm = tgt_pos / scale_per_axis
        goal_pos_norm = goal_pos / scale_per_axis

        # Собираем тензоры
        context = np.concatenate([ctx_pos_norm, raw_sample['ctx_sincos']], axis=1)
        target = np.concatenate([tgt_pos_norm, raw_sample['tgt_sincos']], axis=1)

        return {
            'context': torch.tensor(context, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'goal': torch.tensor(goal_pos_norm, dtype=torch.float32),
            'scale': torch.tensor(scale_per_axis, dtype=torch.float32)  # теперь [3] для денормализации
        }

    # ================== __getitem__ ==================
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        raw_sample = self._generate_sample(start_idx)
        normalized_sample = self._normalize_sample(raw_sample)
        return normalized_sample


def build_dataset(cfgs):
    return ConcatDataset([DroneTrajectoryDataset(**cfg) for cfg in cfgs])
