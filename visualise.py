# import plotly.graph_objects as go
# from plotly.offline import iplot
# import numpy as np
# from datasets_nodrift import DroneTrajectoryDataset

# import torch
# import numpy as np
# import plotly.graph_objects as go
# from diffusers import DDPMScheduler

# def visualize_single_channel_interactive(sample, channel_idx=0):
#     """
#     Интерактивная визуализация диффузии для одного выбранного канала.
#     channel_idx: 0-X, 1-Y, 2-Z, 3-sin(yaw), 4-cos(yaw)
#     """
#     target = sample['target']  # [32, 5]
#     horizon = target.shape[0]
#     channel_names = ['Coord X', 'Coord Y', 'Coord Z', 'Sin(Yaw)', 'Cos(Yaw)']
    
#     scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear')
    
#     # Создаем набор шагов для слайдера (например, каждые 50 шагов)
#     diffusion_steps = np.arange(0, 1001, 1)
    
#     fig = go.Figure()

#     # Генерируем фиксированный шум, чтобы при перемещении слайдера 
#     # мы видели трансформацию конкретного шума, а не мерцание рандома
#     fixed_noise = torch.randn_like(target)

#     # Добавляем "кадры" (траектории на разных этапах зашумления)
#     for t in diffusion_steps:
#         if t == 0:
#             noisy_data = target
#         else:
#             t_tensor = torch.tensor([t - 1])
#             noisy_data = scheduler.add_noise(target, fixed_noise, t_tensor)
        
#         y_values = noisy_data[:, channel_idx].numpy()
        
#         fig.add_trace(
#             go.Scatter(
#                 x=np.arange(horizon),
#                 y=y_values,
#                 mode='lines+markers',
#                 name=f'T={t}',
#                 visible=False, # Скроем все сначала
#                 line=dict(width=3, color='#00FF00' if t == 0 else None),
#                 marker=dict(size=6)
#             )
#         )

#     # Делаем первый кадр видимым
#     fig.data[0].visible = True

#     # Настраиваем слайдер
#     steps = []
#     for i, t in enumerate(diffusion_steps):
#         step = dict(
#             method="update",
#             args=[{"visible": [False] * len(fig.data)},
#                   {"title": f"Diffusion Step T={t} for {channel_names[channel_idx]}"}],
#             label=str(t)
#         )
#         step["args"][0]["visible"][i] = True
#         steps.append(step)

#     sliders = [dict(
#         active=0,
#         currentvalue={"prefix": "Step: "},
#         pad={"t": 50},
#         steps=steps
#     )]

#     fig.update_layout(
#         sliders=sliders,
#         height=600,
#         title=f"Interactive Diffusion: {channel_names[channel_idx]} (T=0)",
#         xaxis_title="Trajectory Step (Horizon)",
#         yaxis_title="Normalized Value",
#         template="plotly_dark"
#     )

#     fig.show()

# # Выбираем канал (0 - это X)


# dsd1 = DroneTrajectoryDataset('V2_01_easy.csv')

# sample = dsd1[2000]


# visualize_single_channel_interactive(sample, channel_idx = 0)



import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

def visualize_tum_raw(file_path, start_index=0, end_index=None, step=20):
    # 1. Загрузка данных
    df = pd.read_csv(file_path, sep='\s+', comment='#', header=None)
    total_len = len(df)

    # --- БЛОК ПРОВЕРКИ ИНДЕКСОВ ---
    # Если начальный индекс отрицательный — ставим 0
    if start_index < 0:
        print(f"Предупреждение: start_index ({start_index}) < 0. Установлено в 0.")
        start_index = 0
    
    # Если конечный индекс не задан или больше максимума — ставим последнюю точку
    if end_index is None or end_index > total_len:
        if end_index is not None:
            print(f"Предупреждение: end_index ({end_index}) > max ({total_len}). Установлено в {total_len}.")
        end_index = total_len
        
    if start_index >= end_index:
        print(f"Ошибка: Начальный индекс ({start_index}) больше или равен конечному ({end_index})!")
        return
    # ------------------------------

    # 2. Обрезка данных
    df_sliced = df.iloc[start_index:end_index].reset_index(drop=True)
    
    t = df_sliced[0].values
    pos = df_sliced[[1, 2, 3]].values 
    quats = df_sliced[[4, 5, 6, 7]].values

    # 3. Расчет частоты и скоростей
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-6, dt) 
    
    avg_dt = np.mean(dt)
    frequency = 1.0 / avg_dt if avg_dt > 0 else 0

    dist_segments = np.sqrt(np.sum(np.diff(pos, axis=0)**2, axis=1))
    velocities = dist_segments / dt
    
    avg_speed = np.mean(velocities)
    max_speed = np.max(velocities)
    min_speed = np.min(velocities)
    total_path = np.sum(dist_segments)

    print(f"--- Statistics [Range: {start_index} to {end_index}] ---")
    print(f"Points: {len(df_sliced)} | Path: {total_path:.2f} m | Freq: {frequency:.2f} Hz")
    print(f"Speed: Avg {avg_speed:.2f}, Max {max_speed:.2f}, Min {min_speed:.2f} m/s")

    # 4. Визуализация
    fig = go.Figure()

    # Линия траектории
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='lines',
        line=dict(color='white', width=4),
        opacity=0.5,
        name='Trajectory'
    ))

    # Ориентация (векторы)
    indices = np.arange(0, len(pos), step)
    axis_length = 0.2 
    for i in indices:
        curr_pos = pos[i]
        r = R.from_quat(quats[i])
        rot_mat = r.as_matrix()
        axes = [(rot_mat[:, 0], 'red'), (rot_mat[:, 1], 'green'), (rot_mat[:, 2], 'blue')]
        for direction, color in axes:
            end_p = curr_pos + direction * axis_length
            fig.add_trace(go.Scatter3d(
                x=[curr_pos[0], end_p[0]], y=[curr_pos[1], end_p[1]], z=[curr_pos[2], end_p[2]],
                mode='lines', line=dict(color=color, width=3),
                showlegend=False, hoverinfo='none'
            ))

    fig.update_layout(
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='data'),
        title=(f"TUM Plot | {start_index}:{end_index} | {frequency:.1f}Hz<br>"
               f"Speed: Avg {avg_speed:.2f}, Max {max_speed:.2f} m/s"),
        template="plotly_dark"
    )

    fig.show()

# Тест: специально вводим кривые индексы (-500 и 999999)
visualize_tum_raw('/home/ivan/Desktop/sets/factory2.txt', start_index=2000, end_index=48000, step=50)