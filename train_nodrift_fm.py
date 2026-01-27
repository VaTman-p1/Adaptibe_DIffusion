import numpy as np

from datetime import datetime  # ← добавь этот импорт наверху

from comet_ml import Experiment
from datetime import datetime
import yaml
import itertools
import os
import torch
from torch.utils.data import DataLoader, random_split
from diffusers import DDPMScheduler
from tqdm import tqdm

from datasets_nodrift import build_dataset
from model import UNet1D
from diffusion import reconstruct_x0
from utils import pad_to_pow2
from lossese_nodrift import diffusion_loss, yaw_loss, goal_loss, path_loss, flow_matching_loss, yaw_loss_cosine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc


def visualize_all_channels(x0_gt, x0_pred, epoch, batch_idx, experiment, sample_idx=0):
    
    # Выбираем один пример из батча
    gt = x0_gt[sample_idx].cpu().numpy()  # [C, T]
    pred = x0_pred[sample_idx].cpu().numpy()  # [C, T]
    
    assert gt.shape[0] == 5, f"Ожидается 5 каналов, получено {gt.shape[0]}"
    
    # СОЗДАЕМ ФИГУРУ с помощью plt.subplots (не plt.figure!)
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    channel_names = ["X", "Y", "Z", "sin", "cos"]
    time_steps = np.arange(gt.shape[1])
    
    for ch in range(5):
        ax = axes[ch]
        
        # Строим линии
        ax.plot(time_steps, gt[ch], 'b-', linewidth=2, label='Реальные', alpha=0.7)
        ax.plot(time_steps, pred[ch], 'r--', linewidth=2, label='Предсказанные', alpha=0.7)
        
        # Заполнение между кривыми
        ax.fill_between(time_steps, gt[ch], pred[ch], 
                       where=(pred[ch] >= gt[ch]), 
                       facecolor='red', alpha=0.2, interpolate=True)
        ax.fill_between(time_steps, gt[ch], pred[ch], 
                       where=(pred[ch] < gt[ch]), 
                       facecolor='blue', alpha=0.2, interpolate=True)
        
        # Настройки
        ax.set_xlabel('Временной шаг', fontsize=10)
        ax.set_ylabel('Значение', fontsize=10)
        ax.set_title(f'{channel_names[ch]}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle(
        f'Сравнение всех 5 каналов траектории\nЭпоха: {epoch}, Батч: {batch_idx}, Пример: {sample_idx}',
        fontsize=14, 
        fontweight='bold', 
        y=1.02
    )
    
    plt.tight_layout()

    figure_name = f"all_channels_epoch{epoch:03d}_batch{batch_idx:03d}_sample{sample_idx}"
    experiment.log_figure(figure_name=figure_name, figure=fig, step=epoch)
    
    plt.close(fig)
    plt.cla()
    plt.clf()
    import gc
    gc.collect()
    
    
    return

def get_x_data_prediction(x_t, v_pred, t):
    """
    Восстанавливает x_data (x1) из текущего состояния x_t и предсказанной скорости v_pred.
    Формула OT path: x_t = (1 - t) * x_noise + t * x_data
    v_t = x_data - x_noise
    Следовательно: x_data = x_t + (1 - t) * v_pred
    """
    # t shape: [B] -> [B, 1, 1]
    t = t.view(-1, 1, 1)
    return x_t + (1 - t) * v_pred

@torch.no_grad()
def sample_flow_euler(model, noise, goal, context, steps=50):
    """
    Простой Euler solver для генерации.
    Интегрируем от t=0 (шум) до t=1 (данные).
    """
    b = noise.shape[0]
    device = noise.device
    
    # x start = noise
    x = noise.clone()
    
    # Временная сетка
    times = torch.linspace(0, 1, steps + 1, device=device)
    dt = 1.0 / steps
    
    for i in range(steps):
        t_curr = times[i]
        
        # Готовим t для батча [B]
        t_batch = torch.ones(b, device=device) * t_curr
        
        # Предсказание скорости v_t
        v_pred = model(x, t_batch, goal, context)
        
        # Шаг Эйлера: x_{t+1} = x_t + v_t * dt
        x = x + v_pred * dt
        
    return x # Это и есть x_data

def run(cfg, base_save_dir="/checkpoints"):
    print("Starting Flow Matching Training...")
    
    experiment = Experiment(
        api_key='',
        project_name="diffusion-trajectory",
        auto_param_logging=False,
        auto_metric_logging=False
    )
    experiment.log_parameters(cfg)

    # Уникальное имя
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_save_dir, f"{timestamp}_FlowMatching")
    os.makedirs(exp_dir, exist_ok=True)
    best_model_path = os.path.join(exp_dir, "best_model.pt")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === МОДЕЛЬ ===
    model = UNet1D(
        in_channels=5,
        channels=cfg["channels"],
        cond_dim=cfg["cond_dim"],
        # dropout=cfg.get("dropout", 0.1) # нету
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=1e-4)

    # === ДАТАСЕТ ===
    dataset = build_dataset(cfg["dataset"])
    val_size = max(1, int(0.1 * len(dataset)))
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, drop_last=True)

    best_val_loss = float("inf")

    # === TRAINING ===
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        metrics = {"fm": 0.0, "yaw": 0.0, "goal": 0.0, "path": 0.0, "total": 0.0}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            # 1. Данные
            x_data = batch["target"].to(device).permute(0, 2, 1) # [B, 5, T]
            goal = batch["goal"].to(device)
            context = batch["context"].to(device)
            
            # Паддинг до степени 2 (для UNet)
            x_data, mask, last_point = pad_to_pow2(x_data)
            
            # 2. Flow Matching Setup
            B = x_data.shape[0]
            
            # x0 (шум) -> x1 (данные)
            x_noise = torch.randn_like(x_data)
            
            # Сэмплируем время t из равномерного распределения [0, 1]
            t = torch.rand(B, device=device)
            
            # Интерполяция Optimal Transport (прямая линия)
            # x_t = (1 - t) * x_noise + t * x_data
            # Но для broadcasting делаем reshape t
            t_reshaped = t.view(-1, 1, 1)
            x_t = (1 - t_reshaped) * x_noise + t_reshaped * x_data
            
            # Целевая скорость (Target Velocity)
            # v_target = d/dt (x_t) = x_data - x_noise
            v_target = x_data - x_noise
            
            # 3. Предсказание модели
            # Модель принимает x_t и t (в Flow Matching t передается как есть, от 0 до 1)
            v_pred = model(x_t, t, goal, context)
            
            # 4. Лоссы
            # Основной лосс векторного поля
            loss_fm = flow_matching_loss(v_pred, v_target, mask)
            
            # Вспомогательные лоссы (нужно восстановить x_data)
            # x_data_pred = x_t + (1 - t) * v_pred
            x_data_pred = get_x_data_prediction(x_t, v_pred, t)
            
            loss_yaw = yaw_loss_cosine(x_data_pred, x_data, mask)
            loss_goal = goal_loss(x_data_pred, goal, last_point)
            
            # Path loss можно считать только для t близких к 1 (данным),
            # или с весом, зависящим от t. Но в FM можно и всегда.
            loss_path = path_loss(x_data_pred, mask)

            # Адаптивный вес для goal/yaw (ближе к t=1 точнее предсказание)
            # Но в FM модель видит направление сразу, можно ставить const
            total_loss = (
                loss_fm 
                + cfg["yaw_w"] * loss_yaw.mean() 
                + cfg["goal_w"] * loss_goal.mean()
                + cfg["path_w"] * loss_path.mean()
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            metrics["fm"] += loss_fm.item()
            metrics["yaw"] += loss_yaw.item()
            metrics["goal"] += loss_goal.item()
            metrics["total"] += total_loss.item()

        # Логирование train
        for k in metrics:
            metrics[k] /= len(train_loader)
            experiment.log_metric(f"train/{k}", metrics[k], step=epoch)

        # === VALIDATION ===
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Val]")):
                x_data = batch["target"].to(device).permute(0, 2, 1)
                goal = batch["goal"].to(device)
                context = batch["context"].to(device)
                x_data, mask, last_point = pad_to_pow2(x_data)

                # Для валидации лосса делаем то же самое (random t)
                x_noise = torch.randn_like(x_data)
                t = torch.rand(x_data.shape[0], device=device)
                t_reshaped = t.view(-1, 1, 1)
                x_t = (1 - t_reshaped) * x_noise + t_reshaped * x_data
                v_target = x_data - x_noise
                
                v_pred = model(x_t, t, goal, context)
                loss_fm = flow_matching_loss(v_pred, v_target, mask)
                val_loss += loss_fm.item()

                # ВИЗУАЛИЗАЦИЯ (Только раз в эпоху)
                if idx == 0 and epoch % 5 == 0:
                    # Для визуализации нам нужно РЕШИТЬ ODE (сгенерировать путь)
                    # Берем первый пример из батча
                    sample_noise = torch.randn(1, 5, x_data.shape[2], device=device)
                    sample_goal = goal[0:1]
                    sample_context = context[0:1]
                    
                    # Генерация через Euler Solver
                    generated_traj = sample_flow_euler(model, sample_noise, sample_goal, sample_context, steps=20)
                    
                    visualize_all_channels(
                        x0_gt=x_data[0:1], 
                        x0_pred=generated_traj, 
                        epoch=epoch, 
                        batch_idx=idx, 
                        experiment=experiment
                    )

        avg_val = val_loss / len(val_loader)
        experiment.log_metric("val/loss", avg_val, step=epoch)
        
        print(f"Epoch {epoch} | Train: {metrics['total']:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved Best Model: {avg_val:.4f}")

    experiment.end()



if __name__ == "__main__":
    with open("trajectory_diffusion/experiments_simple.yaml") as f:
        grid = yaml.safe_load(f)

    keys, values = zip(*grid.items())
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        run(cfg)