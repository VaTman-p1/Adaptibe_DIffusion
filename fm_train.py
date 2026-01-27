from datetime import datetime
from comet_ml import Experiment
import yaml
import itertools
import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from datasets import build_dataset
from adaUnet import UNet1D_AdaLN
from utils import pad_to_pow2
from losses import yaw_loss, goal_loss, path_loss, internal_consistency_loss, drift_consistency_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import gc

# ──────────────────────────────────────────────────────────────────────────────
# Визуализация (без изменений)
# ──────────────────────────────────────────────────────────────────────────────
def visualize_all_channels(x0_gt, x0_pred, epoch, batch_idx, experiment, sample_idx=0):
    gt = x0_gt[sample_idx].cpu().numpy()
    pred = x0_pred[sample_idx].cpu().numpy()
    assert gt.shape[0] == 8
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    axes = axes.flatten()
    channel_names = ["X", "Y", "Z", "cos", "sin", "X_drift", "Y_drift", "Z_drift"]
    time_steps = np.arange(gt.shape[1])
    for ch in range(8):
        ax = axes[ch]
        ax.plot(time_steps, gt[ch], 'b-', linewidth=2, label='Реальные', alpha=0.7)
        ax.plot(time_steps, pred[ch], 'r--', linewidth=2, label='Предсказанные', alpha=0.7)
        ax.fill_between(time_steps, gt[ch], pred[ch],
                        where=(pred[ch] >= gt[ch]), facecolor='red', alpha=0.2)
        ax.fill_between(time_steps, gt[ch], pred[ch],
                        where=(pred[ch] < gt[ch]), facecolor='blue', alpha=0.2)
        ax.set_xlabel('Временной шаг', fontsize=10)
        ax.set_ylabel('Значение', fontsize=10)
        ax.set_title(channel_names[ch], fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
    plt.suptitle(f'Сравнение всех 8 каналов\nЭпоха: {epoch}, Батч: {batch_idx}, Пример: {sample_idx}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    figure_name = f"all_channels_epoch{epoch:03d}_batch{batch_idx:03d}_sample{sample_idx}"
    experiment.log_figure(figure_name=figure_name, figure=fig, step=epoch)
    plt.close(fig)
    gc.collect()

# ──────────────────────────────────────────────────────────────────────────────
# Семплинг (Euler для OT-FM)
# ──────────────────────────────────────────────────────────────────────────────
def sample_otfm(model, shape, goal, context, force, num_steps=20, device='cuda'):
    dt = 1.0 / num_steps
    x = torch.randn(shape, device=device)          # старт с шума
    ts = torch.linspace(0, 1, num_steps + 1, device=device)
    for i in range(num_steps):
        t = ts[i].expand(shape[0])
        v_pred = model(x, t, goal, context, force)
        x = x + dt * v_pred
    return x

# ──────────────────────────────────────────────────────────────────────────────
# Основная функция обучения
# ──────────────────────────────────────────────────────────────────────────────
def run(cfg, base_save_dir="/checkpoints"):
    experiment = Experiment(
        api_key='',
        project_name="diffusion-trajectory",
        auto_param_logging=False,
        auto_metric_logging=False
    )
    experiment.log_parameters(cfg)

    # Уникальная папка
    exp_keys = ["lr", "channels", "goal_w", "yaw_w", "path_w", "consist_w", "vconsist_w", "batch_size"]
    exp_name_parts = [f"{k}_{cfg.get(k, 'NA')}" for k in exp_keys]
    exp_name = "_".join(exp_name_parts)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_save_dir, f"{timestamp}_run")
    os.makedirs(exp_dir, exist_ok=True)
    best_model_path = os.path.join(exp_dir, "weights.pt")
    cfg_path = os.path.join(exp_dir, "config.yaml")
    with open(cfg_path, "w") as f_cfg:
        yaml.safe_dump(cfg, f_cfg)
    experiment.log_parameter("exp_dir", exp_dir)
    experiment.log_parameter("best_model_path", best_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Experiment directory: {exp_dir}")

    model = UNet1D_AdaLN(
        in_channels=8,
        channels=cfg["channels"],
        cond_dim=cfg["cond_dim"]
    ).to(device)
    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"])
    )

    dataset = build_dataset(cfg["dataset"])
    val_size = max(1, int(0.1 * len(dataset)))
    num_batches = val_size // cfg['batch_size']
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False)

    best_val_loss = float("inf")

    # ──────────────────────────────────────────────────────────────────────────────
    # Параметры OT-FM (можно вынести в cfg)
    # ──────────────────────────────────────────────────────────────────────────────
    SIGMA_MIN = 1e-5   # маленький остаточный шум (часто 0 или 1e-5..1e-3)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_metrics = {"fm_loss": 0.0, "yaw": 0.0, "goal": 0.0, "path": 0.0,
                         "intern_cons": 0.0, "drift_cons": 0.0, "total": 0.0}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            data = batch["target"].to(device).permute(0, 2, 1)   # x₁
            goal = batch["goal"].to(device)
            context = batch["context"].to(device)
            force = batch["drift"].to(device)
            data, mask, last_point = pad_to_pow2(data)

            B = data.size(0)
            t = torch.rand((B,), device=device)                     # t ~ U[0,1]
            noise = torch.randn_like(data)                          # x₀ ~ 𝒩(0,I)
            t_exp = t.view(-1, 1, 1)

            # OT линейная интерполяция
            xt = (1 - t_exp) * noise + t_exp * data

            # Целевая скорость для чистого OT (самый распространённый вариант)
            v_target = data - noise

            # Предсказание модели
            pred = model(xt, t, goal, context, force)               # предсказываем v

            # Основной FM-лосс
            fm_loss_value = F.mse_loss(pred, v_target, reduction='none')
            fm_loss_value = (fm_loss_value * mask.expand_as(pred)).sum() / (mask.sum() + 1e-8)

            # Реконструкция для вспомогательных лоссов
            data_pred = noise + pred                                # ≈ x₁̂

            yaw_loss_value = yaw_loss(data_pred, data, mask)
            goal_loss_value = goal_loss(data_pred, goal, last_point)
            drift_loss_value = drift_consistency_loss(data_pred, mask)
            compensation_loss_value = internal_consistency_loss(data_pred, mask, force)

            path_mask = (t < cfg.get("path_t_thresh", 0.2)).float().view(-1, 1, 1)
            path_loss_raw = path_loss(data_pred, mask)
            path_loss_value = (path_loss_raw * path_mask).sum() / (path_mask.sum() + 1e-8)
            compensation_loss_value = (compensation_loss_value * path_mask).sum() / (path_mask.sum() + 1e-8)

            loss = (
                fm_loss_value +
                cfg["yaw_w"] * yaw_loss_value +
                cfg["goal_w"] * goal_loss_value +
                cfg["path_w"] * path_loss_value +
                cfg["consist_w"] * compensation_loss_value +
                cfg["vconsist_w"] * drift_loss_value
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_metrics["fm_loss"] += fm_loss_value.item()
            train_metrics["yaw"] += yaw_loss_value.item()
            train_metrics["goal"] += goal_loss_value.item()
            train_metrics["path"] += path_loss_value.item()
            train_metrics["drift_cons"] += drift_loss_value.item()
            train_metrics["intern_cons"] += compensation_loss_value.item()
            train_metrics["total"] += loss.item()

        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
            experiment.log_metric(f"train/{k}", train_metrics[k], step=epoch)

        # ──────────────────────────────────────────────────────────────────────────────
        # Валидация
        # ──────────────────────────────────────────────────────────────────────────────
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            r_i = torch.randint(0, num_batches, (1,)).item()
            for idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Val]")):
                data = batch["target"].to(device).permute(0, 2, 1)
                goal = batch["goal"].to(device)
                context = batch["context"].to(device)
                force = batch["drift"].to(device)
                data, mask, last_point = pad_to_pow2(data)

                B = data.size(0)
                t = torch.rand((B,), device=device)
                noise = torch.randn_like(data)
                t_exp = t.view(-1, 1, 1)
                xt = (1 - t_exp) * noise + t_exp * data
                v_target = data - noise

                pred = model(xt, t, goal, context, force)
                fm_loss_val = F.mse_loss(pred, v_target, reduction='none')
                fm_loss_val = (fm_loss_val * mask.expand_as(pred)).sum() / (mask.sum() + 1e-8)
                val_total += fm_loss_val.item()

                if (epoch % 5 == 0 or epoch == 1) and idx == r_i:
                    data_pred = noise + pred
                    sample_idx = torch.randint(0, B, (1,)).item()
                    visualize_all_channels(
                        x0_gt=data,
                        x0_pred=data_pred,
                        epoch=epoch,
                        batch_idx=idx,
                        experiment=experiment,
                        sample_idx=sample_idx
                    )

        avg_val_loss = val_total / len(val_loader)
        experiment.log_metric("val/loss", avg_val_loss, step=epoch)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            experiment.log_asset(best_model_path)
            print(f"New best model saved: {best_model_path} | Val loss: {best_val_loss:.6f}")

        print(f"Epoch {epoch:03d} | Train total: {train_metrics['total']:.6f} | "
              f"Val loss: {avg_val_loss:.6f} | Best val: {best_val_loss:.6f}")

    print(f"Experiment finished. Best model: {best_model_path}")
    experiment.end()

if __name__ == "__main__":
    with open("trajectory_diffusion/experiments.yaml") as f:
        grid = yaml.safe_load(f)
    keys, values = zip(*grid.items())
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        run(cfg)