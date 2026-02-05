import numpy as np

from datetime import datetime 

from comet_ml import Experiment
from datetime import datetime
import yaml
import itertools
import os
import torch
from torch.utils.data import DataLoader, random_split
from diffusers import DDPMScheduler
from tqdm import tqdm

from tum_dataset import build_dataset
from model_inpaint import UNet1D
from diffusion import reconstruct_x0
from utils import pad_to_pow2_cond
from losses_nodrift import diffusion_loss, yaw_loss, goal_loss, path_loss, smoothness_loss, reconstruction_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc
import torch
import torch.nn.functional as F

def visualize_all_channels(x0_gt, x0_pred, epoch, batch_idx, experiment, context_len=5, target_len=20, sample_idx=0):
    """
    Рисуем только Target зону (реконструируемую часть).
    context_len: K (сколько шагов было в истории)
    target_len: L (длина предсказываемого будущего)
    """
    # Вырезаем только окно таргета: от K до K+L
    start_idx = context_len
    end_idx = context_len + target_len
    
    gt = x0_gt[sample_idx].cpu().numpy()[:, start_idx:end_idx]
    pred = x0_pred[sample_idx].cpu().numpy()[:, start_idx:end_idx]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    channel_names = ["X", "Y", "Z", "cos", "sin"]
    
    # Создаем временную шкалу для оси X, чтобы она соответствовала шагам будущего
    time_steps = np.arange(start_idx, end_idx)
    
    for ch in range(5):
        ax = axes[ch]
        ax.plot(time_steps, gt[ch], 'b-', label='GT (Target)', alpha=0.8, linewidth=2)
        ax.plot(time_steps, pred[ch], 'r--', label='Pred (Target)', alpha=0.9, linewidth=1.5)
        
        ax.set_title(f"Channel {channel_names[ch]} [Target Only]")
        ax.set_xlabel("Time Step")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Удаляем лишний 6-й пустой график
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    # Логируем в Comet
    experiment.log_figure(figure_name=f"target_only_epoch_{epoch}", figure=fig, step=epoch)
    plt.close(fig)

def run(cfg, base_save_dir="checkpoints"):
    experiment = Experiment(
        api_key='NXUBEZL5VrY1FNYOQpa5xiyP9',
        project_name="adaptive_diffusion",
        auto_param_logging=False,
        auto_metric_logging=False
    )
    experiment.log_parameters(cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_save_dir, f"{timestamp}_run")
    os.makedirs(exp_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # === MODEL ===
    model = UNet1D(
        in_channels=5,
        goal_dim=3,
        channels=cfg["channels"],
        cond_dim=cfg["cond_dim"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    
    # Шедулер настроен на предсказание шума (epsilon)
    scheduler = DDPMScheduler(
        num_train_timesteps=int(cfg["timesteps"]),
        beta_schedule=cfg["beta_schedule"],
        prediction_type="epsilon" 
    )

    # === DATASET ===
    dataset, global_scale = build_dataset(files_config=cfg['data_sources'], 
    global_params=cfg['dataset_params'])
    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=True)

    best_val_loss = float("inf")

    # === TRAINING LOOP ===
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_metrics = {"diff": 0.0, "goal": 0.0, "recon": 0.0, "total": 0.0}

        for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Train]"):
            # x0: [B, 5, L]
            target = batch["target"].to(device).permute(0, 2, 1)
            goal = batch["goal"].to(device)
            context = batch['context'].to(device).permute(0, 2, 1)


            B, C, K = context.shape
            L = target.shape[-1]
            
            # 1. Расчет длины и паддинг
            total_len = K + L
            x0_full = torch.cat([context, target], dim=-1)
            # print(x0_full.size())
            # 1. Паддинг до кратности 8 (UNet downsamples)
            x0, mask, last_idx = pad_to_pow2_cond(x0_full, K)
            # print(x0.size())

            # 2. Диффузия: добавляем шум
            t = torch.randint(0, cfg["timesteps"], (x0.size(0),), device=device).long()
            noise = torch.randn_like(x0)
            # print(noise.size())
            xt = scheduler.add_noise(x0, noise, t)
            # print(xt.size())
            xt[:, :, :K] = x0[:, :, :K]
    # Паддинг тоже лучше держать чистым (нулевым)
            xt[:, :, last_idx+1:] = 0.0
            # print(xt.size())
            # 3. Forward: предсказываем шум (внутри конкатенация с goal)
            noise_pred = model(xt, t.float(), goal)
            # print(noise_pred.size())
            # print(noise_pred.size())

            # 4. Основной Loss (MSE по шуму с учетом маски паддинга)
            diff_loss = diffusion_loss(noise_pred, noise, mask)

            # 5. Геометрические лоссы (только на малых t)
            # Восстанавливаем x0 из предсказанного шума для оценки координат
            x0_recon = reconstruct_x0(xt, noise_pred, t, scheduler)
            ts_mask = (t.float() < (cfg["timesteps"] * cfg["path_t_thresh"])).float() 
            # 1. Получаем лоссы для каждого примера [B]
            # goal_loss_raw = goal_loss(x0_recon, goal, last_idx)     # Твоя функция
            # recon_loss_raw = reconstruction_loss(x0_recon, x0, mask) # Моя исправленная версия
            goal_loss_val= goal_loss(x0_recon, goal, last_idx).mean()     # Твоя функция
            recon_loss_val = reconstruction_loss(x0_recon, x0, mask).mean() # Моя исправленная версия

            # 2. Создаем маску по времени [B]
            # ts_mask = (t.float() < (cfg["timesteps"] * cfg["path_t_thresh"])).float()

            # # 3. Фильтруем (теперь это математически верно!)
            # if ts_mask.sum() > 0:
            #     # Ошибка i-го примера обнулится, если ts_mask[i] == 0
            #     goal_loss_val = (goal_loss_raw * ts_mask).sum() / (ts_mask.sum() + 1e-8)
            #     recon_loss_val = (recon_loss_raw * ts_mask).sum() / (ts_mask.sum() + 1e-8)
            # else:
            #     goal_loss_val = torch.tensor(0.0, device=device)
            #     recon_loss_val = torch.tensor(0.0, device=device)
                

            loss = diff_loss + cfg["goal_w"] * goal_loss_val + cfg['recon_w']*recon_loss_val
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_metrics["diff"] += diff_loss.item()
            train_metrics["goal"] += goal_loss_val.item()
            train_metrics["recon"] += recon_loss_val.item()

            train_metrics["total"] += loss.item()

        # Логирование
        for k in train_metrics:
            experiment.log_metric(f"train/{k}", train_metrics[k]/len(val_loader), step=epoch)

        # === VALIDATION ===
        model.eval()
        # Словарик для накопления средних значений за эпоху
        val_metrics = {"diff": 0.0, "recon": 0.0, "total": 0.0}
        
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                # 1. Подготовка данных
                target = batch["target"].to(device).permute(0, 2, 1)
                goal = batch["goal"].to(device)
                context = batch['context'].to(device).permute(0, 2, 1)
                
                B, C, K = context.shape
                L = target.shape[-1]
                
                # 1. Расчет длины и паддинг
                total_len = K + L
                x0_full = torch.cat([context, target], dim=-1)
                # print(x0_full.size())
                # 1. Паддинг до кратности 8 (UNet downsamples)
                x0, mask, last_idx = pad_to_pow2_cond(x0_full, K)
                # print(x0.size())

                # 2. Диффузия: добавляем шум
                t = torch.randint(0, cfg["timesteps"], (x0.size(0),), device=device).long()
                noise = torch.randn_like(x0)
                # print(noise.size())
                xt = scheduler.add_noise(x0, noise, t)
                # print(xt.size())
                xt[:, :, :K] = x0[:, :, :K]
        # Паддинг тоже лучше держать чистым (нулевым)
                xt[:, :, last_idx+1:] = 0.0
                # print(xt.size())
                # 3. Forward: предсказываем шум (внутри конкатенация с goal)
                noise_pred = model(xt, t.float(), goal)
                diff_loss = diffusion_loss(noise_pred, noise, mask)
                x0_recon = reconstruct_x0(xt, noise_pred, t, scheduler)
                # goal_loss_val= goal_loss(x0_recon, goal, last_idx).mean()
                recon_loss_val = reconstruction_loss(x0_recon, x0, mask).mean()
                loss = diff_loss + cfg["goal_w"] * goal_loss_val + cfg['recon_w']*recon_loss_val
                val_metrics["diff"] += diff_loss.item()
                val_metrics["recon"] += recon_loss_val.item()
                val_metrics["total"] += loss.item()
                if epoch % 5 == 0 and idx == 0:
                    visualize_all_channels(x0, x0_recon, epoch, idx, experiment, 
                                           context_len=K, target_len = L)
            for k in val_metrics:
                experiment.log_metric(f"val/{k}", train_metrics[k]/len(val_loader), step=epoch)

            if val_metrics["total"]/len(val_loader) < best_val_loss:
                best_val_loss = val_metrics["total"]/len(val_loader)
                save_path = os.path.join(exp_dir, "best_weights.pt")
                torch.save(model.state_dict(), save_path)
                print(f" -> Best model saved at epoch {epoch} (Loss: {best_val_loss:.6f})")

        print(f"Epoch {epoch} | Val Total: {val_metrics['total']/len(val_loader):.6f}")

    experiment.end()


if __name__ == "__main__":
    with open("experiments_simple.yaml") as f:
        full_grid = yaml.safe_load(f)

    # Отделяем параметры датасета, которые не участвуют в Grid Search
    data_sources = full_grid.pop('data_sources')
    dataset_params = full_grid.pop('dataset_params')

    keys, values = zip(*full_grid.items())
    
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        
        # Возвращаем датасет в конфиг каждой итерации
        cfg['data_sources'] = data_sources
        cfg['dataset_params'] = dataset_params
        
        run(cfg)