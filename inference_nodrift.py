import plotly.graph_objects as go
from plotly.offline import iplot
import numpy as np
from datasets_nodrift import build_dataset
import torch
from tqdm import tqdm
from typing import Dict, Any
from diffusers import DDIMScheduler, DDPMScheduler
from model import UNet1D
import time
import yaml
import os
import torch.nn.functional as F

def visualize_gt_vs_generated(gt_sample, generated_sample=None, title="Сравнение GT и сгенерированной траектории"):
    """
    Визуализирует сравнение ground truth и сгенерированной моделью траектории без дрифта.
    Args:
        gt_sample: исходный сэмпл из датасета (может быть нормализованным или денормализованным).
        generated_sample: результат работы модели после денормализации (словарь с 'positions', 'orientations').
        title: заголовок графика.
    """
    # Обработка GT сэмпла
    if isinstance(gt_sample['context'], torch.Tensor):
        context = gt_sample['context'].cpu().numpy()
        target = gt_sample['target'].cpu().numpy()
        goal = gt_sample['goal'].cpu().numpy()
        scale = gt_sample['scale'].cpu().numpy() if 'scale' in gt_sample else np.array([1.0, 1.0, 1.0])
    else:
        context = gt_sample['context']
        target = gt_sample['target']
        goal = gt_sample['goal']
        scale = gt_sample['scale'] if 'scale' in gt_sample else np.array([1.0, 1.0, 1.0])
    
    # Позиции GT (без дрифта)
    ctx_pos = context[:, :3] * scale  # денормализуем контекст
    tgt_pos_clean = target[:, :3] * scale  # идеальная траектория
    
    # Yaw GT
    ctx_yaw = np.arctan2(context[:, 3], context[:, 4])
    tgt_yaw_clean = np.arctan2(target[:, 3], target[:, 4])
    
    # Эгоцентр
    ego_pos = ctx_pos[-1]
    ego_yaw = ctx_yaw[-1]
    
    # Подготовка данных для визуализации
    all_positions = [ctx_pos, tgt_pos_clean, goal[None, :]]
    
    # Если есть сгенерированные данные
    if generated_sample is not None:
        # Обработка сгенерированного сэмпла
        gen_positions = generated_sample['positions'].cpu().numpy()  # [seq_len, 3] или [batch_size, seq_len, 3]
        gen_orientations = generated_sample['orientations'].cpu().numpy()  # [seq_len, 2] или [batch_size, seq_len, 2]
        
        # Если батч размер больше 1, берем первый элемент
        if len(gen_positions.shape) == 3:
            gen_positions = gen_positions[0]
            gen_orientations = gen_orientations[0]
        
        # Yaw из сгенерированных ориентаций
        gen_yaw = np.arctan2(gen_orientations[:, 0], gen_orientations[:, 1])
        
        # Добавляем сгенерированные позиции для расчета диапазона
        all_positions.extend([gen_positions])
        
        # Сохраняем для визуализации
        gen_data = {
            'positions': gen_positions,
            'yaw': gen_yaw
        }
    else:
        gen_data = None
    
    # Расчет диапазона для графика
    all_pos = np.vstack(all_positions)
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
        # text=[f'Контекст t={i}, yaw={np.degrees(y):.1f}°'
        #       for i, y in enumerate(ctx_yaw)]
    ))
    print(tgt_pos_clean.shape)
    # 2. Идеальная цель (GT)
    fig.add_trace(go.Scatter3d(
        x=tgt_pos_clean[:,0], y=tgt_pos_clean[:,1], z=tgt_pos_clean[:,2],
        mode='lines+markers',
        name='Идеальная цель (GT)',
        line=dict(color='crimson', width=5, dash='dash'),
        marker=dict(size=7, color='crimson'),
        hoverinfo='text',
        # text=[f'GT pos=({x:.2f},{y:.2f},{z:.2f})'
        #       for x, y, z in tgt_pos_clean.T]
    ))

    # 3. Цель (goal)
    fig.add_trace(go.Scatter3d(
        x=[goal[0]], y=[goal[1]], z=[goal[2]],
        mode='markers',
        name='Цель (goal)',
        marker=dict(size=14, color='gold', symbol='diamond'),
        hoverinfo='text',
        text=['Цель (goal)']
    ))
    
    # 4. Эгоцентр
    fig.add_trace(go.Scatter3d(
        x=[ego_pos[0]], y=[ego_pos[1]], z=[ego_pos[2]],
        mode='markers',
        name='Эгоцентр',
        marker=dict(size=14, color='black', symbol='diamond'),
        hoverinfo='text',
        text=[f'Эгоцентр, yaw = {np.degrees(ego_yaw):.1f}°']
    ))
    
    # 5. Сгенерированная моделью траектория (если есть)
    if gen_data is not None:
        # Идеальная сгенерированная траектория
        fig.add_trace(go.Scatter3d(
            x=gen_data['positions'][:,0], y=gen_data['positions'][:,1], z=gen_data['positions'][:,2],
            mode='lines+markers',
            name='Идеальная цель (Сгенерировано)',
            line=dict(color='limegreen', width=5, dash='solid'),
            marker=dict(size=7, color='limegreen'),
            hoverinfo='text',
            # text=[f'Generated pos=({x:.2f},{y:.2f},{z:.2f})'
            #       for x, y, z in gen_data['positions'].T]
        ))
    
    # 6. Стрелки направления yaw
    def add_yaw_arrows(fig, pos, yaws, color, name_prefix, showlegend=True):
        arrow_length = (max_v - min_v).max() * 0.05  # адаптивная длина стрелки
        dx = np.cos(yaws) * arrow_length
        dy = np.sin(yaws) * arrow_length
        dz = np.zeros_like(yaws)
        # Рисуем стрелки только для ключевых точек (каждая 3-я точка)
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
    
    # Стрелки для GT
    add_yaw_arrows(fig, ctx_pos, ctx_yaw, 'royalblue', 'Контекст GT')
    add_yaw_arrows(fig, tgt_pos_clean, tgt_yaw_clean, 'crimson', 'GT Идеал')
    
    # Стрелки для сгенерированных данных
    if gen_data is not None:
        add_yaw_arrows(fig, gen_data['positions'], gen_data['yaw'], 'limegreen', 'Сгенерировано Идеал')
    
    # Настройки графика
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

@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    goal: torch.Tensor,
    context: torch.Tensor,
    in_channels: int,
    seq_len: int,
    scale: torch.Tensor,
    cfg: Dict[str, Any],
    num_inference_steps: int = 20,
    generator_seed: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Генерирует траекторию с помощью DDPM сэмплера (без дрифта).
    """
    device = next(model.parameters()).device
    scheduler = DDPMScheduler(
        num_train_timesteps=200,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        beta_start=1e-7,
        beta_end=0.02
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    print(timesteps)
    
    batch_size = goal.shape[0]
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=device).manual_seed(generator_seed)
    
    # Округляем seq_len до ближайшей степени двойки для модели
    gen_seq = 1 << (seq_len - 1).bit_length()
    x = torch.randn(
        (batch_size, in_channels, gen_seq),
        device=device,
        generator=generator
    ) 
    
    # Перемещаем все входные данные на устройство
    goal = goal.to(device)
    context = context.to(device)
    
    for t in tqdm(timesteps, desc=f"DDIM · {num_inference_steps} шагов"):
        t_tensor = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
        pred_noise = model(
            x,
            t_tensor.float() / 200,
            goal,
            context,
        )

        x = scheduler.step(
            model_output=pred_noise,
            timestep=t,
            sample=x,
            generator=generator,
            # use_clipped_model_output=True
        ).prev_sample
    
    # Берем только нужную длину последовательности
    generated_raw = x[:, :, :seq_len]  # [batch_size, 5, seq_len]
    
    # Денормализуем результат
    return denormalize_sample(generated_raw, scale, goal)

@torch.no_grad()
def sample_ddpm_reconstruction(
    model: torch.nn.Module,
    goal: torch.Tensor,
    context: torch.Tensor,
    in_channels: int,
    target: torch.Tensor,  # Существующая траектория для восстановления
    seq_len: int,
    scale: torch.Tensor,
    cfg: Dict[str, Any],
    num_inference_steps: int = 20,
    strength: float = 0.1,  # 0.0 - оригинал, 1.0 - полный шум
    generator_seed: int = None,
) -> Dict[str, torch.Tensor]:
    """
    Восстанавливает траекторию, добавляя шум к target и прогоняя через DDPM.
    """
    device = next(model.parameters()).device
    
    # 1. Настройка шедулера (те же параметры, что при обучении)
    scheduler = DDPMScheduler(
        num_train_timesteps=200,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
        beta_start=1e-7,
        beta_end=0.02
    )
    scheduler.set_timesteps(num_inference_steps, device=device)
    
    batch_size = goal.shape[0]
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=device).manual_seed(generator_seed)

    # 2. Подготовка target (нормализация и padding)
    # Предполагаем, что target уже в том же пространстве, что и выход модели (нормализован)
    # Если нет, его нужно нормализовать функцией, обратной denormalize_sample
    
    # Округляем до степени 2, как в оригинале
    gen_seq = 1 << (seq_len - 1).bit_length()
    print(gen_seq)
    target = target.permute(0,2,1)
    x_padded = F.pad(target, (0, gen_seq-seq_len), mode="replicate")
    # target_padded = torch.zeros((batch_size, in_channels, gen_seq), device=device)
    # target_padded[:, :, :seq_len] = target.to(device)

    # 3. Определяем стартовый шаг
    # strength=0.8 означает, что мы пропустим первые 20% шагов денойзинга и начнем с 80% зашумления
    init_timestep = int(num_inference_steps * strength)
    t_start = scheduler.timesteps[num_inference_steps - init_timestep]
    
    # 4. Зашумляем оригинал до уровня t_start
    noise = torch.randn(x_padded.shape, device=device, generator=generator)
    x = scheduler.add_noise(x_padded, noise, t_start)

    # 5. Цикл денойзинга (начинаем не с начала, а с t_start)
    timesteps = scheduler.timesteps[num_inference_steps - init_timestep:]
    
    goal = goal.to(device)
    context = context.to(device)
    
    for t in tqdm(timesteps, desc=f"Reconstruction · {len(timesteps)} шагов"):
        t_tensor = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
        
        pred_noise = model(
            x,
            t_tensor.float() / 200,
            goal,
            context,
        )

        x = scheduler.step(
            model_output=pred_noise,
            timestep=t,
            sample=x,
            generator=generator,
        ).prev_sample
    
    # 6. Обрезка и денормализация
    generated_raw = x[:, :, :seq_len]
    return denormalize_sample(generated_raw, scale, goal)


@torch.no_grad()
def sample_flow_matching(
    model: torch.nn.Module,
    goal: torch.Tensor,
    context: torch.Tensor,
    in_channels: int,
    seq_len: int,
    scale: torch.Tensor,
    num_steps: int = 50,
    generator_seed: int = None,
) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    batch_size = goal.shape[0]
    generator = None
    if generator_seed is not None:
        generator = torch.Generator(device=device).manual_seed(generator_seed)
    
    # Округляем seq_len до ближайшей степени двойки для модели
    gen_seq = 1 << (seq_len - 1).bit_length()
    x = torch.randn(
        (batch_size, in_channels, gen_seq),
        device=device,
        generator=generator
    )
    
    # Перемещаем все входные данные на устройство
    goal = goal.to(device)
    context = context.to(device)
    
    dt = 1.0 / num_steps
    for i in tqdm(range(num_steps), desc=f"Flow Matching · {num_steps} шагов"):
        t_val = 1.0 - (i * dt)  # from ~1 to ~0
        t_tensor = torch.full((batch_size,), t_val, device=device, dtype=torch.float)
        v = model(
            x,
            t_tensor,  
            goal,
            context,
        )
        x = x - v * dt  # reverse direction with -v
    
    # Берем только нужную длину последовательности
    generated_raw = x[:, :, :seq_len]  # [batch_size, 8, seq_len]
    
    # Денормализуем результат
    return denormalize_sample(generated_raw, scale, goal)

def denormalize_sample(input_tensor: torch.Tensor, scale: torch.Tensor, goal: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Денормализует выход модели (без дрифта).
    Args:
        input_tensor: [batch_size, 5, seq_len]
        scale: [3] или [batch_size, 3]
    Returns:
        Словарь с денормализованными тензорами в формате [batch_size, seq_len, channels]
    """
    device = input_tensor.device
    # Приводим scale к правильной форме
    if scale.dim() == 1:  # [3]
        scale = scale.view(1, 3, 1)  # [1, 3, 1]
    elif scale.dim() == 2:  # [batch_size, 3]
        scale = scale.unsqueeze(2)  # [batch_size, 3, 1]
    scale = scale.to(device)
    goal = goal.to(device)
    
    # Переставляем размерности: [batch_size, seq_len, 5]
    input_tensor = input_tensor.permute(0, 2, 1)
    
    # Извлекаем компоненты
    tgt_pos_norm = input_tensor[..., :3]  # [batch_size, seq_len, 3]
    tgt_sincos = input_tensor[..., 3:5]  # [batch_size, seq_len, 2]
    
    scale = scale.permute(0, 2, 1)
    
    # Денормализуем
    tgt_pos_denorm = tgt_pos_norm * scale  # [batch_size, seq_len, 3]
    goal = goal*scale
    
    return {
        'positions': tgt_pos_denorm,  # [batch_size, seq_len, 3]
        'orientations': tgt_sincos,  # [batch_size, seq_len, 2]
        'goal': goal
    }

def demo_sampling_and_visualization(dataset, model, cfg, idx=200, seed=42, method='ddim'):
    """
    Демонстрация: берет сэмпл из датасета, генерирует траекторию моделью (без дрифта), визуализирует сравнение.
    Добавлен параметр method для выбора 'ddim' или 'flow'.
    """
    print(f"Загружаем сэмпл №{idx} из датасета...")
    # 1. Получаем нормализованный сэмпл из датасета
    normalized_sample = dataset[idx]  # это уже нормализованный сэмпл
    
    # 2. Подготавливаем данные для модели
    target = normalized_sample['target'].unsqueeze(0)
    context = normalized_sample['context'].unsqueeze(0)  # [1, C, 5]
    goal = normalized_sample['goal'].unsqueeze(0)  # [1, 3]
    scale = normalized_sample['scale']  # [3]
    
    # 3. Генерируем траекторию
    print(f"Генерируем траекторию моделью ({method})...")
    start_time = time.time()
    if method == 'ddpm':
        generated_output = sample_ddpm(
            model=model,
            goal=goal,
            context=context,
            in_channels=5,
            seq_len=21,
            scale=scale,
            cfg=cfg,
            num_inference_steps=200,
            generator_seed=seed
        )

    elif method == 'reconstruct':
        generated_output = sample_ddpm_reconstruction(
            model=model,
            goal=goal,
            target=target,
            context=context,
            in_channels=5,
            seq_len=21,
            scale=scale,
            cfg=cfg,
            strength=0.5,
            num_inference_steps=200,
            generator_seed=seed)
        
    elif method == 'flow':
        generated_output = sample_flow_matching(
            model=model,
            goal=goal,
            context=context,
            in_channels=5,
            seq_len=21,
            scale=scale,
            num_steps=10,
            generator_seed=seed
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    end_time = time.time()
    print(f"Время генерации: {end_time - start_time:.2f} сек")
    
    # 4. Получаем исходный (денормализованный) сэмпл для сравнения
    raw_sample = dataset[idx]
    
    # 5. Визуализируем сравнение
    print("Визуализация результатов...")
    visualize_gt_vs_generated(
        gt_sample=raw_sample,
        generated_sample=generated_output,
        title=f"Сравнение GT и сгенерированной траектории (Сэмпл №{idx}, seed={seed}, method={method})"
    )
    
    return {
        'gt_sample': raw_sample,
        'generated_sample': generated_output,
        'normalized_input': normalized_sample
    }

# Основной код
# 1. Загружаем датасет (предполагаем, что build_dataset возвращает DroneTrajectoryDataset без дрифта)
 # Исправлено на build_dataset для соответствия импорту

# 2. Загружаем конфигурацию
config_path = 'experiments_simple.yaml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
print("Конфиг загружен:")
# print(cfg)

# print(cfg['dataset'][0][0])
dataset = build_dataset(cfg['dataset'][0], augment=False) 


# 3. Создаем модель (in_channels=5 без дрифта)
# Предполагаем, что cfg содержит 'model' с ключами; иначе хардкод
model_channels = cfg.get('channels', [[64, 128, 256]])
model_cond_dim = cfg.get('cond_dim', 128)
# print(model_channels)
model = UNet1D(
    in_channels=5,  # pos3 + sincos2 = 5 (без дрифта)
    cond_dim=model_cond_dim[0],
    channels=model_channels[0]
)

# 4. Загружаем веса (предполагаем, что модель обучена без дрифта; если нет, нужно переобучить)
checkpoint_path = 'checkpoints/20260203_111328_ddpm_run/last_weights.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint)

# 5. Перемещаем на устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()  # режим оценки
print(f"Модель загружена на {device}")
print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

# 6. Запускаем демонстрацию (можно изменить method на 'flow' для тестирования)
results = demo_sampling_and_visualization(
    dataset=dataset,
    model=model,
    cfg=cfg,
    idx=2874,  # номер сэмпла из датасета
    seed=42,  # seed для воспроизводимости
    method='reconstruct'  # или 'flow'
)