import plotly.graph_objects as go
from plotly.offline import iplot
import numpy as np
from datasets_nodrift import DroneTrajectoryDataset

import torch
import numpy as np
import plotly.graph_objects as go
from diffusers import DDPMScheduler

def visualize_single_channel_interactive(sample, channel_idx=0):
    """
    Интерактивная визуализация диффузии для одного выбранного канала.
    channel_idx: 0-X, 1-Y, 2-Z, 3-sin(yaw), 4-cos(yaw)
    """
    target = sample['target']  # [32, 5]
    horizon = target.shape[0]
    channel_names = ['Coord X', 'Coord Y', 'Coord Z', 'Sin(Yaw)', 'Cos(Yaw)']
    
    scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='scaled_linear')
    
    # Создаем набор шагов для слайдера (например, каждые 50 шагов)
    diffusion_steps = np.arange(0, 1001, 1)
    
    fig = go.Figure()

    # Генерируем фиксированный шум, чтобы при перемещении слайдера 
    # мы видели трансформацию конкретного шума, а не мерцание рандома
    fixed_noise = torch.randn_like(target)

    # Добавляем "кадры" (траектории на разных этапах зашумления)
    for t in diffusion_steps:
        if t == 0:
            noisy_data = target
        else:
            t_tensor = torch.tensor([t - 1])
            noisy_data = scheduler.add_noise(target, fixed_noise, t_tensor)
        
        y_values = noisy_data[:, channel_idx].numpy()
        
        fig.add_trace(
            go.Scatter(
                x=np.arange(horizon),
                y=y_values,
                mode='lines+markers',
                name=f'T={t}',
                visible=False, # Скроем все сначала
                line=dict(width=3, color='#00FF00' if t == 0 else None),
                marker=dict(size=6)
            )
        )

    # Делаем первый кадр видимым
    fig.data[0].visible = True

    # Настраиваем слайдер
    steps = []
    for i, t in enumerate(diffusion_steps):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                  {"title": f"Diffusion Step T={t} for {channel_names[channel_idx]}"}],
            label=str(t)
        )
        step["args"][0]["visible"][i] = True
        steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Step: "},
        pad={"t": 50},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders,
        height=600,
        title=f"Interactive Diffusion: {channel_names[channel_idx]} (T=0)",
        xaxis_title="Trajectory Step (Horizon)",
        yaxis_title="Normalized Value",
        template="plotly_dark"
    )

    fig.show()

# Выбираем канал (0 - это X)


dsd1 = DroneTrajectoryDataset('V2_01_easy.csv')

sample = dsd1[2000]


visualize_single_channel_interactive(sample, channel_idx = 0)