import plotly.graph_objects as go
from plotly.offline import iplot
import numpy as np
from datasets import DroneTrajectoryDataset
import torch
from tqdm import tqdm
from typing import Dict, Any
from diffusers import DDIMScheduler, DDPMScheduler
from adaUnet import UNet1D_AdaLN
import time

def visualize_gt_vs_generated(gt_sample, generated_sample=None, title="Сравнение GT и сгенерированной траектории"):
    """
    Визуализирует сравнение ground truth и сгенерированной моделью траектории
    
    Args:
        gt_sample: исходный сэмпл из датасета (может быть нормализованным или денормализованным)
        generated_sample: результат работы модели после денормализации (словарь с 'positions', 'orientations', 'drifts')
        title: заголовок графика
    """
    # Обработка GT сэмпла
    if isinstance(gt_sample['context'], torch.Tensor):
        context = gt_sample['context'].cpu().numpy()
        target = gt_sample['target'].cpu().numpy() 
        goal = gt_sample['goal'].cpu().numpy()
        scale = gt_sample['scale'].cpu().numpy() if 'scale' in gt_sample else np.array([1.0, 1.0, 1.0])
        # drift = gt_sample['drift'].cpu().numpy()
    else:
        context = gt_sample['context']
        target = gt_sample['target']
        goal = gt_sample['goal']
        scale = gt_sample['scale'] if 'scale' in gt_sample else np.array([1.0, 1.0, 1.0])
        # drift = gt_sample['drift']

    # Позиции GT
    ctx_pos = context[:, :3] * scale  # денормализуем контекст
    tgt_pos_clean = target[:, :3] * scale  # идеальная траектория
    tgt_pos_drift = target[:, 5:8] * scale  # сдвинутая траектория (дрейф)
    
    # Yaw GT
    ctx_yaw = np.arctan2(context[:, 3], context[:, 4])
    tgt_yaw_clean = np.arctan2(target[:, 3], target[:, 4])
    
    # Эгоцентр
    ego_pos = ctx_pos[-1]
    ego_yaw = ctx_yaw[-1]

    # Подготовка данных для визуализации
    all_positions = [ctx_pos, tgt_pos_clean, tgt_pos_drift, goal[None, :]]
    
    # Если есть сгенерированные данные
    if generated_sample is not None:
        # Обработка сгенерированного сэмпла
        gen_positions = generated_sample['positions'].cpu().numpy()  # [seq_len, 3] или [batch_size, seq_len, 3]
        gen_orientations = generated_sample['orientations'].cpu().numpy()  # [seq_len, 2] или [batch_size, seq_len, 2]
        gen_drifts = generated_sample['drifts'].cpu().numpy()  # [seq_len, 3] или [batch_size, seq_len, 3]
        
        # Если батч размер больше 1, берем первый элемент
        if len(gen_positions.shape) == 3:
            gen_positions = gen_positions[0]
            gen_orientations = gen_orientations[0]
            gen_drifts = gen_drifts[0]
        
        # Вычисляем сдвинутую траекторию для сгенерированных данных
        gen_pos_drift = gen_positions + gen_drifts
        
        # Yaw из сгенерированных ориентаций
        gen_yaw = np.arctan2(gen_orientations[:, 0], gen_orientations[:, 1])
        
        # Добавляем сгенерированные позиции для расчета диапазона
        all_positions.extend([gen_positions, gen_pos_drift])
        
        # Сохраняем для визуализации
        gen_data = {
            'positions': gen_positions,
            'drift_positions': gen_pos_drift,
            'yaw': gen_yaw
        }
    else:
        gen_data = None

    # Расчет диапазона для графика
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
        text=[f'Контекст t={i}, yaw={np.degrees(y):.1f}°' 
              for i, y in enumerate(ctx_yaw)]
    ))

    # 2. Идеальная цель (GT)
    fig.add_trace(go.Scatter3d(
        x=tgt_pos_clean[:,0], y=tgt_pos_clean[:,1], z=tgt_pos_clean[:,2],
        mode='lines+markers',
        name='Идеальная цель (GT)',
        line=dict(color='crimson', width=5, dash='dash'),
        marker=dict(size=7, color='crimson'),
        hoverinfo='text',
        text=[f'GT pos=({x:.2f},{y:.2f},{z:.2f})' 
              for x, y, z in zip(tgt_pos_clean[:,0], tgt_pos_clean[:,1], tgt_pos_clean[:,2])]
    ))

    # 3. Сдвинутая цель (GT)
    fig.add_trace(go.Scatter3d(
        x=tgt_pos_drift[:,0], y=tgt_pos_drift[:,1], z=tgt_pos_drift[:,2],
        mode='lines+markers',
        name='Сдвинутая цель (GT)',
        line=dict(color='purple', width=5),
        marker=dict(size=7, color='purple'),
        hoverinfo='text',
        text=[f'GT drift pos=({x:.2f},{y:.2f},{z:.2f})' 
              for x, y, z in zip(tgt_pos_drift[:,0], tgt_pos_drift[:,1], tgt_pos_drift[:,2])]
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

    # 6. Сгенерированная моделью траектория (если есть)
    if gen_data is not None:
        # Идеальная сгенерированная траектория
        fig.add_trace(go.Scatter3d(
            x=gen_data['positions'][:,0], y=gen_data['positions'][:,1], z=gen_data['positions'][:,2],
            mode='lines+markers',
            name='Идеальная цель (Сгенерировано)',
            line=dict(color='limegreen', width=5, dash='solid'),
            marker=dict(size=7, color='limegreen'),
            hoverinfo='text',
            text=[f'Generated pos=({x:.2f},{y:.2f},{z:.2f})' 
                  for x, y, z in zip(gen_data['positions'][:,0], gen_data['positions'][:,1], gen_data['positions'][:,2])]
        ))
        
        # Сдвинутая сгенерированная траектория
        fig.add_trace(go.Scatter3d(
            x=gen_data['drift_positions'][:,0], y=gen_data['drift_positions'][:,1], z=gen_data['drift_positions'][:,2],
            mode='lines+markers',
            name='Сдвинутая цель (Сгенерировано)',
            line=dict(color='orange', width=5, dash='solid'),
            marker=dict(size=7, color='orange'),
            hoverinfo='text',
            text=[f'Generated drift pos=({x:.2f},{y:.2f},{z:.2f})' 
                  for x, y, z in zip(gen_data['drift_positions'][:,0], gen_data['drift_positions'][:,1], gen_data['drift_positions'][:,2])]
        ))

    # 7. Стрелки направления yaw
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
    add_yaw_arrows(fig, tgt_pos_drift, tgt_yaw_clean, 'purple', 'GT Дрейф')

    # Стрелки для сгенерированных данных
    if gen_data is not None:
        add_yaw_arrows(fig, gen_data['positions'], gen_data['yaw'], 'limegreen', 'Сгенерировано Идеал')
        add_yaw_arrows(fig, gen_data['drift_positions'], gen_data['yaw'], 'orange', 'Сгенерировано Дрейф')

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

# Пример использования:

def demo_sampling_and_visualization(dataset, model, cfg, idx=200, seed=42):
    """
    Демонстрация: берет сэмпл из датасета, генерирует траекторию моделью, визуализирует сравнение
    """
    print(f"Загружаем сэмпл №{idx} из датасета...")
    
    # 1. Получаем нормализованный сэмпл из датасета
    normalized_sample = dataset.get_sample(idx, np.array([2.0, 2.0, 2.0]) ) # это уже нормализованный сэмпл
    
    # 2. Подготавливаем данные для модели
    context = normalized_sample['context'].unsqueeze(0)  # [1, C, 5]
    goal = normalized_sample['goal'].unsqueeze(0)        # [1, 3]
    force = normalized_sample['drift'].unsqueeze(0)      # [1, 3]
    scale = normalized_sample['scale']                    # [3]

    
    # 3. Генерируем траекторию
    print("Генерируем траекторию моделью...")
    start_time = time.time()
    generated_output = sample_ddim(
        model=model,
        goal=goal,
        context=context,
        force=force,
        in_channels=8,  # tgt_pos(3) + tgt_sincos(2) + tgt_drift(3) = 8
        seq_len=21,
        scale=scale,
        cfg=cfg,
        num_inference_steps=20,
        eta=0.1,
        generator_seed=seed
    )
    end_time = time.time()
    print(end_time-start_time)
    # 4. Получаем исходный (денормализованный) сэмпл для сравнения
    raw_sample = dataset.get_sample(
        idx=idx,
        normalize=False,
        custom_raw_drift=np.array([2,2,2])  # используем тот же дрейф
    )
    
    # 5. Визуализируем сравнение
    print("Визуализация результатов...")
    visualize_gt_vs_generated(
        gt_sample=raw_sample,
        generated_sample=generated_output,
        title=f"Сравнение GT и сгенерированной траектории (Сэмпл №{idx}, seed={seed})"
    )
    
    return {
        'gt_sample': raw_sample,
        'generated_sample': generated_output,
        'normalized_input': normalized_sample
    }

# Функция sample_ddim (исправленная версия для работы с новой денормализацией)
@torch.no_grad()
@torch.no_grad()
def sample_ddim(
    model: torch.nn.Module,
    goal: torch.Tensor,
    context: torch.Tensor,
    force: torch.Tensor,
    in_channels,
    seq_len,
    scale,
    cfg: Dict[str, Any],
    num_inference_steps: int = 200,
    eta: float = 0.3,
    generator_seed: int = None,
) -> Dict[str, torch.Tensor]:
    device = next(model.parameters()).device

    scheduler = DDPMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
    )
    

    scheduler.set_timesteps(num_inference_steps, device=device)  # Указываем устройство
    timesteps = scheduler.timesteps  # Уже на правильном устройстве

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
    ) * scheduler.init_noise_sigma

    # Перемещаем все входные данные на устройство один раз
    goal = goal.to(device)
    context = context.to(device)
    force = force.to(device)
    
    for t in tqdm(timesteps, desc=f"DDIM · {num_inference_steps} шагов · η={eta}"):
        t_tensor = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)

        pred_noise = model(
            x,
            t_tensor.float() / 1000,
            goal,  # Уже на устройстве
            context,  # Уже на устройстве
            force  # Уже на устройстве
        )

        x = scheduler.step(
            model_output=pred_noise,
            timestep=t,
            sample=x,
            generator=generator,
            # use_clipped_model_output=True
        ).prev_sample

    # Берем только нужную длину последовательности
    generated_raw = x[:, :, :seq_len]  # [batch_size, 8, seq_len]
    
    # Денормализуем результат
    return denormalize_sample(generated_raw, scale)

# Исправленная функция денормализации для формата модели
def denormalize_sample(input_tensor: torch.Tensor, scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Денормализует выход модели
    
    Args:
        input_tensor: [batch_size, 8, seq_len]
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
    
    # Переставляем размерности: [batch_size, seq_len, 8]
    # input_tensor = input_tensor.permute(0, 2, 1)
    input_tensor = input_tensor.permute(0, 2, 1)
    # Извлекаем компоненты
    tgt_pos_norm = input_tensor[..., :3]      # [batch_size, seq_len, 3]
    tgt_sincos = input_tensor[..., 3:5]        # [batch_size, seq_len, 2]
    tgt_drift_norm = input_tensor[..., 5:8]    # [batch_size, seq_len, 3]
    
    scale = scale.permute(0, 2, 1)
    # Денормализуем
    print(tgt_pos_norm.shape)
    print(scale.shape)
    tgt_pos_denorm = tgt_pos_norm * scale      # [batch_size, seq_len, 3]
    tgt_drift_denorm = tgt_drift_norm * scale  # [batch_size, seq_len, 3]
    
    return {
        'positions': tgt_pos_denorm,           # [batch_size, seq_len, 3]
        'orientations': tgt_sincos,            # [batch_size, seq_len, 2]
        'drifts': tgt_drift_denorm             # [batch_size, seq_len, 3]
    }



# 1. Загружаем датасет и модель
dataset = DroneTrajectoryDataset('trajectory_diffusion/V2_02_medium.csv')
import torch
import yaml
import os

# 1. Загружаем конфигурацию
config_path = 'trajectory_diffusion/checkpoints/' \
'20260116_173356_lr_2e-4_channels_64_128_256_goal_w_0.05_yaw_w_0.2_path_w_0.05_consist_w_0.2_vconsist_w_0.2_batch_size_256_timesteps_1000/'\
'config.yaml'
with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

print("Конфиг загружен:")
print(cfg)

# 2. Создаем модель


model = UNet1D_AdaLN(
    in_channels=8,      # обычно 8
    cond_dim=128, 
    channels=[64,128,256]      # обычно 5  

)

# 3. Загружаем веса
checkpoint_path = 'trajectory_diffusion/checkpoints/' \
'20260116_173356_lr_2e-4_channels_64_128_256_goal_w_0.05_yaw_w_0.2_path_w_0.05_consist_w_0.2_vconsist_w_0.2_batch_size_256_timesteps_1000/'\
'weights.pt'  # или ваш путь к весам
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint) 

# 4. Перемещаем на устройство
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()  # режим оценки

print(f"Модель загружена на {device}")
print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

# Теперь модель готова к использованию
# Пример использования:
# output = model(x, t, goal, context, force)

# 2. Запускаем демонстрацию
results = demo_sampling_and_visualization(
    dataset=dataset,
    model=model,
    cfg=cfg,
    idx=3568,        # номер сэмпла из датасета
    seed=42         # seed для воспроизводимости
)


