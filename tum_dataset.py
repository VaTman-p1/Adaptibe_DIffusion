import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset
from scipy.spatial.transform import Rotation as R

def extract_yaw(quat: np.ndarray) -> float:
    """Извлекает yaw из кватерниона для Gravity Aligned Frame."""
    norm = np.linalg.norm(quat)
    quat = quat / norm if norm > 1e-6 else np.array([0, 0, 0, 1])
    return R.from_quat(quat).as_euler('zyx', degrees=False)[0]

class DroneTUMDataset(Dataset):
    def __init__(
        self,
        txt_path: str,
        start_idx: int = 0,
        end_idx: int = None,
        context_points: int = 5,
        target_points: int = 21,
        target_offset_points: int = 3,
        goal_offset_points: int = 8,
        stride_points: int = 4,
        target_freq: int = 10,
        time_horizon_scale: float = 2.1,
        fixed_scale: float = None,
        augment: bool = True,
    ):
        self.context_points = context_points
        self.target_points = target_points
        self.target_offset_points = target_offset_points
        self.goal_offset_points = goal_offset_points
        self.stride_points = stride_points
        self.target_freq = target_freq
        self.augment = augment
        self.time_horizon_scale = time_horizon_scale
        self.fixed_scale = fixed_scale

        # 1. Загрузка данных
        df = pd.read_csv(txt_path, sep='\s+', comment='#', header=None)
        df.columns = ['time', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        
        # 2. Применение среза (Slicing)
        total_len = len(df)
        s_idx = max(0, start_idx)
        e_idx = total_len if (end_idx is None or end_idx > total_len) else end_idx
        self.df = df.iloc[s_idx:e_idx].reset_index(drop=True)

        # 3. Частота и даунсэмплинг
        avg_dt = np.mean(np.diff(self.df['time'].values))
        self.original_freq = int(round(1.0 / avg_dt)) if avg_dt > 0 else 30
        self.downsamp = max(1, self.original_freq // self.target_freq)

        # 4. Формирование индексов с фильтрацией (5 точек вперед)
        self.max_needed = (
            (context_points - 1) * self.downsamp +
            target_offset_points +
            (target_points - 1) * self.downsamp +
            goal_offset_points
        )
        
        potential_indices = []
        curr = 0
        while curr + self.max_needed < len(self.df):
            potential_indices.append(curr)
            curr += self.stride_points
            
        self.indices = [idx for idx in potential_indices if self._is_forward_looking(idx)]
        
        print(f"Loaded {txt_path}: {len(self.indices)} forward samples (filtered from {len(potential_indices)})")

    def _is_forward_looking(self, start_idx: int) -> bool:
        """Проверяет, что первые 5 точек таргета в среднем по X > 0."""
        ego_idx = start_idx + (self.context_points - 1) * self.downsamp
        tgt_start = ego_idx + self.target_offset_points
        
        # Берем до 5 точек (если таргет короче, берем сколько есть)
        num_check = min(5, self.target_points)
        tgt_idxs = [tgt_start + i * self.downsamp for i in range(num_check)]
        
        if tgt_idxs[-1] >= len(self.df):
            return False
            
        ego_row = self.df.iloc[ego_idx]
        ego_pos = ego_row[['x','y','z']].values.astype(float)
        ego_yaw = extract_yaw(ego_row[['qx','qy','qz','qw']].values)
        R_local = R.from_euler('z', ego_yaw).inv()
        
        tgt_pts_global = self.df.iloc[tgt_idxs][['x','y','z']].values.astype(float)
        tgt_pts_local = R_local.apply(tgt_pts_global - ego_pos)
        
        return np.mean(tgt_pts_local[:, 0]) > 0

    def get_max_speed(self):
        pos = self.df[['x', 'y', 'z']].values
        dt = np.diff(self.df['time'].values)
        dt[dt == 0] = 1e-6
        speed = np.linalg.norm(np.diff(pos, axis=0), axis=1) / dt
        # Игнорируем скачки выше 50 м/с (ошибки датчиков)
        valid_speeds = speed[speed < 50.0] 
        return np.max(valid_speeds) if len(valid_speeds) > 0 else 1.0

    def _generate_sample(self, start_idx: int):
        # EGO
        ego_idx = start_idx + (self.context_points - 1) * self.downsamp
        ego_row = self.df.iloc[ego_idx]
        ego_pos = ego_row[['x','y','z']].values.astype(float)
        ego_yaw = extract_yaw(ego_row[['qx','qy','qz','qw']].values)
        R_local = R.from_euler('z', ego_yaw).inv()

        # CONTEXT
        ctx_idxs = [start_idx + i * self.downsamp for i in range(self.context_points)]
        ctx_df = self.df.iloc[ctx_idxs]
        ctx_pos = R_local.apply(ctx_df[['x','y','z']].values - ego_pos)
        ctx_yaws = np.array([extract_yaw(q) for q in ctx_df[['qx','qy','qz','qw']].values])
        ctx_sc = np.stack([np.sin(ctx_yaws - ego_yaw), np.cos(ctx_yaws - ego_yaw)], axis=1)

        # TARGET
        tgt_start = ego_idx + self.target_offset_points
        tgt_idxs = [tgt_start + i * self.downsamp for i in range(self.target_points)]
        tgt_df = self.df.iloc[tgt_idxs]
        tgt_pos = R_local.apply(tgt_df[['x','y','z']].values - ego_pos)
        tgt_yaws = np.array([extract_yaw(q) for q in tgt_df[['qx','qy','qz','qw']].values])
        tgt_sc = np.stack([np.sin(tgt_yaws - ego_yaw), np.cos(tgt_yaws - ego_yaw)], axis=1)

        # GOAL
        goal_idx = np.random.randint(tgt_idxs[-1], min(tgt_idxs[-1] + self.goal_offset_points + 1, len(self.df)))
        goal_pos = R_local.apply(self.df.iloc[goal_idx][['x','y','z']].values - ego_pos)

        # Augmentation (Mirroring)
        if self.augment and np.random.rand() > 0.5:
            ctx_pos[:, 1] *= -1; tgt_pos[:, 1] *= -1; goal_pos[1] *= -1
            ctx_sc[:, 0] *= -1; tgt_sc[:, 0] *= -1

        return {'ctx_p': ctx_pos, 'ctx_sc': ctx_sc, 'tgt_p': tgt_pos, 'tgt_sc': tgt_sc, 'goal': goal_pos}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        raw = self._generate_sample(self.indices[idx])
        s = self.fixed_scale
        return {
            'context': torch.tensor(np.concatenate([raw['ctx_p']/s, raw['ctx_sc']], axis=1), dtype=torch.float32),
            'target': torch.tensor(np.concatenate([raw['tgt_p']/s, raw['tgt_sc']], axis=1), dtype=torch.float32),
            'goal': torch.tensor(raw['goal']/s, dtype=torch.float32),
            'scale': torch.tensor(s, dtype=torch.float32)
        }

def build_dataset(files_config, global_params, augment=False):
    """
    Сборка с расчетом единого скейла по всем срезам.
    """
    # max_v = 0
    # # Первый проход для поиска глобального максимума скорости
    # for cfg in files_config:
    #     ds = DroneTUMDataset(txt_path=cfg['path'], start_idx=cfg.get('start', 0), 
    #                          end_idx=cfg.get('end', None), **global_params)
    #     max_v = max(max_v, ds.get_max_speed())
    
    # global_scale = max_v * global_params.get('time_horizon_scale', 3.0)
    # print(f"\nFinal Global Scale: {global_scale:.4f} (based on {max_v:.2f} m/s)")
    global_scale = [1,1,1]

    # Создание финальных объектов
    datasets = [DroneTUMDataset(txt_path=cfg['path'], start_idx=cfg.get('start', 0), 
                                end_idx=cfg.get('end', None), fixed_scale=[1,1,1], 
                                **global_params) for cfg in files_config]
    
    return ConcatDataset(datasets), global_scale



# import yaml
# import json
# import os

# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         return yaml.safe_load(f)

# def save_training_stats(scale, config, save_path="norm_stats.json"):
#     """Сохраняет скейл и параметры для инференса"""
#     stats = {
#         "global_scale": float(scale),
#         "params": config['dataset_params']
#     }
#     with open(save_path, 'w') as f:
#         json.dump(stats, f, indent=4)
#     print(f"Normalization stats saved to {save_path}")

# # --- ПРИМЕР ИСПОЛЬЗОВАНИЯ ---

# # 1. Загружаем конфиг из YAML
# cfg = load_config("config.yaml")

# # 2. Собираем датасет
# # Передаем список файлов и словарь параметров
# dataset, global_scale = build_dataset(
#     files_config=cfg['data_sources'], 
#     global_params=cfg['dataset_params']
# )

# # 3. Сохраняем скейл (он нам позарез нужен будет при запуске планировщика на дроне)
# save_training_stats(global_scale, cfg)

# # 4. Создаем загрузчик для PyTorch
# from torch.utils.data import DataLoader
# loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

# print(f"Total samples in dataset: {len(dataset)}")
# import plotly.graph_objects as go
# import numpy as np

# import plotly.graph_objects as go
# import numpy as np

# def visualize_sample_plotly(dataset_item, scale=1.0):
#     """
#     Визуализация сэмпла в 3D с направлениями взгляда (Yaw) для каждой точки.
#     """
#     # Распаковка тензоров
#     ctx = dataset_item['context'].numpy() 
#     tgt = dataset_item['target'].numpy()
#     goal = dataset_item['goal'].numpy() * scale
    
#     # Разделяем на позиции и sin/cos
#     ctx_pos = ctx[:, :3] * scale
#     ctx_sc = ctx[:, 3:]
    
#     tgt_pos = tgt[:, :3] * scale
#     tgt_sc = tgt[:, 3:]
    
#     fig = go.Figure()

#     def add_direction_vectors(pos, sc, name, color):
#         # Направление взгляда (вектор Forward)
#         # Так как Yaw локальный, то в локальной СК вектор Forward это [cos(yaw), sin(yaw), 0]
#         # В нашем датасете sc[:, 0] = sin, sc[:, 1] = cos
#         u = sc[:, 1]  # dx
#         v = sc[:, 0]  # dy
#         w = np.zeros_like(u) # dz (так как Gravity Aligned)

#         fig.add_trace(go.Cone(
#             x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
#             u=u, v=v, w=w,
#             sizemode="absolute",
#             sizeref=0.2 * scale, # Размер стрелочки
#             colorscale=[[0, color], [1, color]],
#             showscale=False,
#             name=f'Heading {name}'
#         ))

#     # 1. Контекст (Past)
#     fig.add_trace(go.Scatter3d(
#         x=ctx_pos[:, 0], y=ctx_pos[:, 1], z=ctx_pos[:, 2],
#         mode='lines+markers',
#         marker=dict(size=3, color='green'),
#         line=dict(color='green', width=2),
#         name='Context Path'
#     ))
#     add_direction_vectors(ctx_pos, ctx_sc, "Past", "green")

#     # 2. Таргет (Future)
#     fig.add_trace(go.Scatter3d(
#         x=tgt_pos[:, 0], y=tgt_pos[:, 1], z=tgt_pos[:, 2],
#         mode='lines+markers',
#         marker=dict(size=3, color='blue'),
#         line=dict(color='blue', width=4),
#         name='Target Path'
#     ))
#     add_direction_vectors(tgt_pos, tgt_sc, "Future", "blue")

#     # 3. Цель (Goal)
#     fig.add_trace(go.Scatter3d(
#         x=[goal[0]], y=[goal[1]], z=[goal[2]],
#         mode='markers',
#         marker=dict(size=8, color='red', symbol='diamond'),
#         name='Goal'
#     ))

#     # Оси координат (Мировые/Локальные в 0,0,0)
#     axis_len = scale * 0.3
#     fig.add_trace(go.Scatter3d(x=[0, axis_len], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=5), name='Ego X (Forward)'))

#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X (Forward)',
#             yaxis_title='Y (Left)',
#             zaxis_title='Z (Up)',
#             aspectmode='data'
#         ),
#         title=f"Sample with Yaw Directions (Scale: {scale:.2f}m)",
#         margin=dict(l=0, r=0, b=0, t=40)
#     )
    
#     fig.show()

# import numpy as np

# def visualize_random_samples(dataset, scale, num_samples=10):
#     """
#     Выбирает num_samples случайных индексов из датасета и отображает их.
#     """
#     total_len = len(dataset)
#     if total_len == 0:
#         print("Датасет пуст!")
#         return

#     # Выбираем случайные индексы
#     random_indices = np.random.choice(total_len, size=min(num_samples, total_len), replace=False)
    
#     print(f"Визуализация {len(random_indices)} случайных сэмплов...")
    
#     for i, idx in enumerate(random_indices):
#         print(f"Отображение сэмпла {i+1}/{num_samples} (Индекс в сете: {idx})")
#         sample = dataset[idx]
        
#         # Используем функцию визуализации с конусами, которую мы сделали ранее
#         visualize_sample_plotly(sample, scale=scale)

# # --- ЗАПУСК ---
# # Если у тебя уже создан full_dataset и есть global_scale:
# visualize_random_samples(dataset, global_scale, num_samples=20)