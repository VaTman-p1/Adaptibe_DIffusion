import plotly.graph_objects as go
from plotly.offline import iplot
import numpy as np
from datasets import DroneTrajectoryDataset
import torch
from tqdm import tqdm
from typing import Dict, Any
from adaUnet import UNet1D_AdaLN
import time
import yaml
import os

# ──────────────────────────────────────────────────────────────────────────────
# Визуализация (твой оригинальный код, слегка отформатирован)
# ──────────────────────────────────────────────────────────────────────────────
def visualize_gt_vs_generated(gt_sample, generated_sample=None, title="Сравнение GT и сгенерированной траектории"):
    """
    Визуализирует сравнение ground truth и сгенерированной моделью траектории
    """
    if isinstance(gt_sample['context'], torch.Tensor):
        context = gt_sample['context'].cpu().numpy()
        target  = gt_sample['target'].cpu().numpy()
        goal    = gt_sample['goal'].cpu().numpy()
        scale   = gt_sample['scale'].cpu().numpy() if 'scale' in gt_sample else np.array([1.0, 1.0, 1.0])
    else:
        context = gt_sample['context']
        target  = gt_sample['target']
        goal    = gt_sample['goal']
        scale   = gt_sample['scale'] if 'scale' in gt_sample else np.array([1.0, 1.0, 1.0])

    # Позиции GT
    ctx_pos       = context[:, :3] * scale
    tgt_pos_clean = target[:, :3] * scale
    tgt_pos_drift = target[:, 5:8] * scale

    # Yaw GT
    ctx_yaw       = np.arctan2(context[:, 3], context[:, 4])
    tgt_yaw_clean = np.arctan2(target[:, 3], target[:, 4])

    # Эгоцентр
    ego_pos = ctx_pos[-1]
    ego_yaw = ctx_yaw[-1]

    # Все позиции для расчёта диапазона
    all_positions = [ctx_pos, tgt_pos_clean, tgt_pos_drift, goal[None, :]]

    gen_data = None
    if generated_sample is not None:
        gen_positions   = generated_sample['positions'].cpu().numpy()
        gen_orientations = generated_sample['orientations'].cpu().numpy()
        gen_drifts      = generated_sample['drifts'].cpu().numpy()

        if len(gen_positions.shape) == 3:
            gen_positions   = gen_positions[0]
            gen_orientations = gen_orientations[0]
            gen_drifts      = gen_drifts[0]

        gen_pos_drift = gen_positions + gen_drifts
        gen_yaw       = np.arctan2(gen_orientations[:, 0], gen_orientations[:, 1])

        all_positions.extend([gen_positions, gen_pos_drift])

        gen_data = {
            'positions': gen_positions,
            'drift_positions': gen_pos_drift,
            'yaw': gen_yaw
        }

    # Диапазон графика
    all_pos = np.vstack([pos for pos_list in all_positions for pos in pos_list])
    padding = np.max(np.abs(all_pos)) * 0.1
    min_v = np.min(all_pos, axis=0) - padding
    max_v = np.max(all_pos, axis=0) + padding

    fig = go.Figure()

    # 1. Контекст (GT)
    fig.add_trace(go.Scatter3d(
        x=ctx_pos[:,0], y=ctx_pos[:,1], z=ctx_pos[:,2],
        mode='lines+markers',
        name='Контекст (GT)',
        line=dict(color='royalblue', width=5),
        marker=dict(size=7, color='royalblue', symbol='circle'),
        hoverinfo='text',
        text=[f'Контекст t={i}, yaw={np.degrees(y):.1f}°' for i, y in enumerate(ctx_yaw)]
    ))

    # 2. Идеальная цель (GT)
    fig.add_trace(go.Scatter3d(
        x=tgt_pos_clean[:,0], y=tgt_pos_clean[:,1], z=tgt_pos_clean[:,2],
        mode='lines+markers',
        name='Идеальная цель (GT)',
        line=dict(color='crimson', width=5, dash='dash'),
        marker=dict(size=7, color='crimson'),
        hoverinfo='text',
        text=[f'GT pos=({x:.2f},{y:.2f},{z:.2f})' for x,y,z in tgt_pos_clean]
    ))

    # 3. Сдвинутая цель (GT)
    fig.add_trace(go.Scatter3d(
        x=tgt_pos_drift[:,0], y=tgt_pos_drift[:,1], z=tgt_pos_drift[:,2],
        mode='lines+markers',
        name='Сдвинутая цель (GT)',
        line=dict(color='purple', width=5),
        marker=dict(size=7, color='purple'),
        hoverinfo='text',
        text=[f'GT drift pos=({x:.2f},{y:.2f},{z:.2f})' for x,y,z in tgt_pos_drift]
    ))

    # 4. Цель (goal)
    fig.add_trace(go.Scatter3d(
        x=[goal[0]], y=[goal[1]], z=[goal[2]],
        mode='markers',
        name='Цель (goal)',
        marker=dict(size=14, color='gold', symbol='diamond'),
        hoverinfo='text',
        text=['Цель (goal)']
    ))

    # 5. Эгоцентр
    fig.add_trace(go.Scatter3d(
        x=[ego_pos[0]], y=[ego_pos[1]], z=[ego_pos[2]],
        mode='markers',
        name='Эгоцентр',
        marker=dict(size=14, color='black', symbol='diamond'),
        hoverinfo='text',
        text=[f'Эгоцентр, yaw = {np.degrees(ego_yaw):.1f}°']
    ))

    # 6. Сгенерированные траектории (если есть)
    if gen_data is not None:
        fig.add_trace(go.Scatter3d(
            x=gen_data['positions'][:,0], y=gen_data['positions'][:,1], z=gen_data['positions'][:,2],
            mode='lines+markers',
            name='Идеальная цель (Сгенерировано)',
            line=dict(color='limegreen', width=5),
            marker=dict(size=7, color='limegreen'),
            hoverinfo='text',
            text=[f'Gen pos=({x:.2f},{y:.2f},{z:.2f})' for x,y,z in gen_data['positions']]
        ))

        fig.add_trace(go.Scatter3d(
            x=gen_data['drift_positions'][:,0], y=gen_data['drift_positions'][:,1], z=gen_data['drift_positions'][:,2],
            mode='lines+markers',
            name='Сдвинутая цель (Сгенерировано)',
            line=dict(color='orange', width=5),
            marker=dict(size=7, color='orange'),
            hoverinfo='text',
            text=[f'Gen drift pos=({x:.2f},{y:.2f},{z:.2f})' for x,y,z in gen_data['drift_positions']]
        ))

    # Стрелки yaw (твоя функция)
    def add_yaw_arrows(fig, pos, yaws, color, name_prefix, showlegend=True):
        arrow_length = (max_v - min_v).max() * 0.05
        dx = np.cos(yaws) * arrow_length
        dy = np.sin(yaws) * arrow_length
        dz = np.zeros_like(yaws)
        step = max(1, len(pos) // 10)
        for i in range(0, len(pos), step):
            p = pos[i]
            ux, uy, uz = dx[i], dy[i], dz[i]
            fig.add_trace(go.Scatter3d(
                x=[p[0], p[0]+ux],
                y=[p[1], p[1]+uy],
                z=[p[2], p[2]+uz],
                mode='lines',
                line=dict(color=color, width=4),
                name=f"{name_prefix} yaw" if i == 0 and showlegend else None,
                showlegend=(i == 0 and showlegend),
                hoverinfo='skip'
            ))

    add_yaw_arrows(fig, ctx_pos, ctx_yaw, 'royalblue', 'Контекст GT')
    add_yaw_arrows(fig, tgt_pos_clean, tgt_yaw_clean, 'crimson', 'GT Идеал')
    add_yaw_arrows(fig, tgt_pos_drift, tgt_yaw_clean, 'purple', 'GT Дрейф')

    if gen_data is not None:
        add_yaw_arrows(fig, gen_data['positions'], gen_data['yaw'], 'limegreen', 'Сгенерировано Идеал')
        add_yaw_arrows(fig, gen_data['drift_positions'], gen_data['yaw'], 'orange', 'Сгенерировано Дрейф')

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(range=[min_v[0], max_v[0]], title='X'),
            yaxis=dict(range=[min_v[1], max_v[1]], title='Y'),
            zaxis=dict(range=[min_v[2], max_v[2]], title='Z'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.8, y=1.8, z=0.8))
        ),
        width=1200,
        height=900,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, b=0, t=50)
    )
    iplot(fig)

# ──────────────────────────────────────────────────────────────────────────────
# Семплер: чистый детерминированный OT Flow Matching
# ──────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def sample_ot_fm(
    model: torch.nn.Module,
    goal: torch.Tensor,
    context: torch.Tensor,
    force: torch.Tensor,
    in_channels: int = 8,
    seq_len: int = 21,
    scale: torch.Tensor = None,
    num_inference_steps: int = 20,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    generator_seed: int = None
) -> Dict[str, torch.Tensor]:
    """
    Чистый OT Flow Matching без стохастичности (линейные пути).
    """
    batch_size = goal.shape[0]
    gen_seq = 1 << (seq_len - 1).bit_length()

    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=device).manual_seed(generator_seed)

    # Начальный шум
    x = torch.randn((batch_size, in_channels, gen_seq), device=device, generator=generator)

    ts = torch.linspace(0, 1, num_inference_steps + 1, device=device)
    dt = 1.0 / num_inference_steps
    start_time = time.time()
    for i in tqdm(range(num_inference_steps), desc=f"OT-FM чистый · {num_inference_steps} шагов"):
        t = ts[i].expand(batch_size)

        v_pred = model(x, t, goal.to(device), context.to(device), force.to(device))

        x = x + dt * v_pred

    generated_raw = x[:, :, :seq_len]
    end_time = time.time()
    return denormalize_sample(generated_raw, scale), end_time-start_time

# ──────────────────────────────────────────────────────────────────────────────
# Денормализация
# ──────────────────────────────────────────────────────────────────────────────
def denormalize_sample(input_tensor: torch.Tensor, scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Денормализует выход модели [B, 8, seq_len] → словарь с позициями, ориентациями, дрейфами
    """
    device = input_tensor.device

    # Приводим scale к виду [B, 1, 3] для broadcasting по seq_len
    if scale.dim() == 1:               # [3]
        scale = scale.view(1, 1, 3)
    elif scale.dim() == 2:             # [B, 3]
        scale = scale.unsqueeze(1)     # → [B, 1, 3]
    scale = scale.to(device)

    # Переводим input в [B, seq_len, 8]
    input_tensor = input_tensor.permute(0, 2, 1)

    tgt_pos_norm     = input_tensor[..., :3]      # [B, seq_len, 3]
    tgt_sincos       = input_tensor[..., 3:5]     # [B, seq_len, 2]
    tgt_drift_norm   = input_tensor[..., 5:8]     # [B, seq_len, 3]

    # Денормализация только позиций и дрейфов (ориентации не масштабируются)
    tgt_pos_denorm   = tgt_pos_norm   * scale
    tgt_drift_denorm = tgt_drift_norm * scale

    return {
        'positions':     tgt_pos_denorm,     # [B, seq_len, 3]
        'orientations':  tgt_sincos,         # [B, seq_len, 2] — без масштабирования
        'drifts':        tgt_drift_denorm    # [B, seq_len, 3]
    }

# ──────────────────────────────────────────────────────────────────────────────
# Демонстрация
# ──────────────────────────────────────────────────────────────────────────────
def demo_sampling_and_visualization(dataset, model, cfg, idx=200, seed=42):
    print(f"Загружаем сэмпл №{idx} из датасета...")
    normalized_sample = dataset.get_sample(idx, np.array([2.0, 2.0, 2.0]))

    context = normalized_sample['context'].unsqueeze(0)
    goal    = normalized_sample['goal'].unsqueeze(0)
    force   = normalized_sample['drift'].unsqueeze(0)
    scale   = normalized_sample['scale']
    print(force)
    print("Генерируем траекторию моделью (чистый OT-FM)...")
    

    generated_output, time = sample_ot_fm(
        model=model,
        goal=goal,
        context=context,
        force=force,
        in_channels=8,
        seq_len=21,
        scale=scale,
        num_inference_steps=8,
        generator_seed=seed
    )

    
    print(f"Генерация заняла {time:.2f} сек")

    raw_sample = dataset.get_sample(idx=idx, normalize=False, custom_raw_drift=np.array([2,2,2]))

    print("Визуализация результатов...")
    visualize_gt_vs_generated(
        gt_sample=raw_sample,
        generated_sample=generated_output,
        title=f"Чистый OT Flow Matching (без стохастичности) | Сэмпл №{idx}, seed={seed}"
    )

    return {
        'gt_sample': raw_sample,
        'generated_sample': generated_output,
        'normalized_input': normalized_sample
    }

# ──────────────────────────────────────────────────────────────────────────────
# Запуск
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dataset = DroneTrajectoryDataset('trajectory_diffusion/V2_02_medium.csv')

    with open('trajectory_diffusion/experiments.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    model = UNet1D_AdaLN(in_channels=8, cond_dim=128, channels=[32,64,128,256])
    checkpoint_path = 'trajectory_diffusion/checkpoints/20260118_181424_run/weights.pt'  # подставь свой путь
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    device = next(model.parameters()).device
    print(f"Модель загружена на {device}, параметров: {sum(p.numel() for p in model.parameters()):,}")

    results = demo_sampling_and_visualization(
        dataset=dataset,
        model=model,
        cfg=cfg,
        idx=3568,
        seed=42
    )