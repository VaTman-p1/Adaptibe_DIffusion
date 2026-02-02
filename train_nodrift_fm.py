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

from datasets_nodrift import build_dataset
from model import UNet1D
from diffusion import reconstruct_x0
from utils import pad_to_pow2
from losses_nodrift import diffusion_loss, yaw_loss, goal_loss, path_loss, smoothness_loss, flow_matching_loss, ot_flow_matching_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gc


def ot_pairing(x0, x1):
    """
    Approximate OT pairing by sorting batch samples.
    x0, x1: [B, C, T]
    """
    B = x0.size(0)
    score0 = x0.view(B, -1).mean(dim=1)
    score1 = x1.view(B, -1).mean(dim=1)

    idx0 = torch.argsort(score0)
    idx1 = torch.argsort(score1)

    return x0[idx0], x1[idx1]

def visualize_all_channels(x0_gt, x0_pred, epoch, batch_idx, experiment, traj_size = 21, sample_idx=0):
    """
    Визуализация всех 8 каналов траектории отдельно для одного примера.
    """
    # Импорты matplotlib УЖЕ должны быть в начале скрипта с matplotlib.use('Agg')
    
    # Выбираем один пример из батча
    gt = x0_gt[sample_idx].cpu().numpy()[:, :traj_size]  # [C, T]
    pred = x0_pred[sample_idx].cpu().numpy()[:, :traj_size]  # [C, T]
    
    assert gt.shape[0] == 5, f"Ожидается 5 каналов, получено {gt.shape[0]}"
    
    # СОЗДАЕМ ФИГУРУ с помощью plt.subplots (не plt.figure!)
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.flatten()
    
    channel_names = ["X", "Y", "Z", "cos", "sin"]
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
    
    # Записываем в Comet ML
    figure_name = f"all_channels_epoch{epoch:03d}_batch{batch_idx:03d}_sample{sample_idx}"
    experiment.log_figure(figure_name=figure_name, figure=fig, step=epoch)
    
    # ВАЖНО: Закрываем фигуру и очищаем память
    plt.close(fig)
    plt.cla()
    plt.clf()
    
    # Принудительный сбор мусора (опционально, для надежности)
    import gc
    gc.collect()
    
    # total_mae = np.mean(np.abs(gt - pred))
    # print(f"[Визуализация] Эпоха {epoch}, пример {sample_idx}: средняя MAE = {total_mae:.6f}")
    
    return


def run(cfg, base_save_dir="checkpoints"):
    experiment = Experiment(
        api_key='NXUBEZL5VrY1FNYOQpa5xiyP9',
        project_name="adaptive_diffusion",
        auto_param_logging=False,
        auto_metric_logging=False
    )
    experiment.log_parameters(cfg)

# -----------------------------
# RUN FUNCTION
# -----------------------------


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_save_dir, f"{timestamp}_fm_run")
    os.makedirs(exp_dir, exist_ok=True)
    best_model_path = os.path.join(exp_dir, "best_weights.pt")
    last_model_path = os.path.join(exp_dir, "last_weights.pt")

    experiment.log_parameter("exp_dir", exp_dir)
    experiment.log_parameter("best_model_path", best_model_path)
    experiment.log_parameter("last_model_path", last_model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Experiment directory: {exp_dir}")

    # ------------------------
    # Model & optimizer
    # ------------------------
    model = UNet1D(
        in_channels=5,
        channels=cfg["channels"],
        cond_dim=cfg["cond_dim"],
    ).to(device)

    print(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"])
    )

    # ------------------------
    # Dataset
    # ------------------------
    dataset = build_dataset(cfg["dataset"])
    for i, ds in enumerate(dataset.datasets):
        print(f"Subdataset {i} length after stride: {len(ds)}")
    print(f"Total dataset length: {len(dataset)}")

    val_size = max(1, int(0.1 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    best_val_loss = float("inf")

    # ------------------------
    # Training loop
    # ------------------------
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        train_metrics = {"fm":0.0, "yaw":0.0, "goal":0.0, "smoothness":0.0, "total":0.0}

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x0 = batch["target"].to(device).permute(0,2,1)  # [B, C, T]
            goal = batch["goal"].to(device)
            context = batch["context"].to(device)

            x0, mask, last_point = pad_to_pow2(x0)

            # Sample t in [0,1]
            t = torch.rand(x0.size(0), device=device)

            # Sample noise endpoint x1
            x1 = torch.randn_like(x0)

            # Interpolation
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1

            # Ground-truth velocity
            v_target = x1 - x0

            # Predict velocity
            v_pred = model(xt, t, goal, context)

            # Flow matching loss
            fm_loss_val = flow_matching_loss(v_pred, v_target, mask)

            # Recover x0 estimate for auxiliary losses
            x0_pred = xt - t[:, None, None] * v_pred

            # Auxiliary losses
            yaw_loss_val = yaw_loss(x0_pred, x0, mask)
            goal_loss_raw = goal_loss(x0_pred, goal, last_point)
            smooth_loss_raw = smoothness_loss(x0_pred, mask)

            ts_mask = (t < cfg.get("path_t_thresh", 0.2)).float()
            if ts_mask.sum() > 0:
                goal_loss_val = (goal_loss_raw * ts_mask).sum() / (ts_mask.sum() + 1e-8)
                smooth_loss_val = (smooth_loss_raw * ts_mask).sum() / (ts_mask.sum() + 1e-8)
            else:
                goal_loss_val = torch.tensor(0.0, device=device)
                smooth_loss_val = torch.tensor(0.0, device=device)

            # Total loss
            loss = fm_loss_val + cfg["yaw_w"]*yaw_loss_val + cfg["goal_w"]*goal_loss_val + cfg["smooth_w"]*smooth_loss_val

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_metrics["fm"] += fm_loss_val.item()
            train_metrics["yaw"] += yaw_loss_val.item()
            train_metrics["goal"] += goal_loss_val.item()
            train_metrics["smoothness"] += smooth_loss_val.item()
            train_metrics["total"] += loss.item()

        # Save last model
        torch.save(model.state_dict(), last_model_path)
        experiment.log_asset(last_model_path)

        # Log train metrics
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)
            experiment.log_metric(f"train/{k}", train_metrics[k], step=epoch)

        # ------------------------
        # Validation
        # ------------------------
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch} [Val]")):
                x0 = batch["target"].to(device).permute(0,2,1)
                goal = batch["goal"].to(device)
                context = batch["context"].to(device)

                x0, mask, last_point = pad_to_pow2(x0)
                t = torch.rand(x0.size(0), device=device)
                x1 = torch.randn_like(x0)
                xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1

                v_target = x1 - x0
                v_pred = model(xt, t, goal, context)

                val_loss = flow_matching_loss(v_pred, v_target, mask)
                val_total += val_loss.item()

                # Optional visualization
                if (epoch % 5 == 0 or epoch == 1) and idx == 0:
                    sample_idx = torch.randint(0, x0.size(0), (1,)).item()
                    x0_pred = xt - t[:, None, None] * v_pred
                    visualize_all_channels(
                        x0_gt=x0,
                        x0_pred=x0_pred,
                        epoch=epoch,
                        batch_idx=idx,
                        experiment=experiment,
                        sample_idx=sample_idx
                    )

            avg_val_loss = val_total / len(val_loader)
            experiment.log_metric("val/loss", avg_val_loss, step=epoch)

        # ------------------------
        # Save best model
        # ------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            experiment.log_asset(best_model_path)
            print(f"🎉 New best model saved: {best_model_path} | Val loss: {best_val_loss:.6f}")

        print(f"Epoch {epoch:03d} | Train total: {train_metrics['total']:.6f} | Val loss: {avg_val_loss:.6f} | Best val: {best_val_loss:.6f}")

    print(f"Experiment finished. Best model: {best_model_path}")
    experiment.end()


# ------------------------
# Entry point
# ------------------------
if __name__ == "__main__":
    import yaml

    with open("experiments_simple.yaml") as f:
        grid = yaml.safe_load(f)

    keys, values = zip(*grid.items())
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        run(cfg)


if __name__ == "__main__":
    with open("experiments_simple.yaml") as f:
        grid = yaml.safe_load(f)

    keys, values = zip(*grid.items())
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        run(cfg)