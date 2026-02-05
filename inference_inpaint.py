import torch
import yaml
import itertools
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
try:
    matplotlib.use('TkAgg') # Самый стандартный оконный движок
except:
    matplotlib.use('Qt5Agg') # Если стоит PyQT
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Твои импорты
from train_inpaint import UNet1D 
from tum_dataset import build_dataset

def run_inference():
    # 1. ЗАГРУЗКА КОНФИГА (Копия логики из твоего __main__)
    with open("experiments_simple.yaml") as f:
        full_grid = yaml.safe_load(f)

    # Вытаскиваем статические параметры
    data_sources = full_grid.pop('data_sources')
    dataset_params = full_grid.pop('dataset_params')

    # Генерируем комбинацию (берем самую первую из сетки для теста)
    keys, values = zip(*full_grid.items())
    first_combo = next(itertools.product(*values))
    cfg = dict(zip(keys, first_combo))
    
    # Добавляем обратно параметры данных
    cfg['data_sources'] = data_sources
    cfg['dataset_params'] = dataset_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. СБОРКА ДАТАСЕТА (Как в твоем определении build_dataset)
    # Возвращает (dataset, scale)
    dataset, global_scale = build_dataset(
        files_config=cfg['data_sources'], 
        global_params=cfg['dataset_params']
    )
    
    val_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 3. МОДЕЛЬ
    model = UNet1D(
        in_channels=5, 
        goal_dim=3, 
        channels=cfg['channels'], 
        cond_dim=cfg['cond_dim']
    ).to(device)
    
    # Укажи путь к сохраненным весам
    model.load_state_dict(torch.load("/home/ivan/Adaptibe_DIffusion/checkpoints/20260204_173141_run/best_weights.pt", map_location=device))
    model.eval()

    # 4. SCHEDULER
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg['timesteps'],
        beta_schedule=cfg['beta_schedule'],
        prediction_type="epsilon"
    )
    
    # Ставим 50 шагов для скорости
    num_steps = 500
    scheduler.set_timesteps(num_steps)

    # 5. ИНФЕРЕНС (Sampling)
    batch = next(iter(val_loader))
    context = batch['context'].to(device).permute(0, 2, 1) # [1, 5, K]
    target_gt = batch['target'].to(device).permute(0, 2, 1)
    goal = batch['goal'].to(device)
    
    K = context.shape[-1]
    L = target_gt.shape[-1]
    horizon = K + L
    
    # Шум
    xt = torch.randn((1, 5, horizon+3), device=device)
    
    print(xt.size())

    print(f"Sampling with {num_steps} steps...")
    with torch.no_grad():
        for t in scheduler.timesteps:
            t_batch = torch.full((1,), t, device=device).float()
            
            # Вклеиваем контекст как в трейне
            xt[:, :, :K] = context
            xt[:, :, -3:] = 0.0
            
            
            # Предсказание
            noise_pred = model(xt, t_batch, goal)
            # print(noise_pred.size())
            # print(xt.size())
            # print(t.size())
            
            # Шаг DDPM
            xt = scheduler.step(noise_pred, t, xt).prev_sample

    xt[:, :, :K] = context # Финал
    xt= xt[:,:,:K+L]

    # 6. ВИЗУАЛИЗАЦИЯ
    # Переводим в numpy для удобства
    pred_np = xt[0].cpu().numpy()
    context_np = context[0].cpu().numpy()
    target_np = target_gt[0].cpu().numpy()
    gt_full = np.concatenate([context_np, target_np], axis=-1)
    goal_np = goal[0].cpu().numpy()
    
    K = context_np.shape[-1]
    L = target_np.shape[-1]
    total_len = K + L

    # 1. СОЗДАЕМ ТРИ ГРАФИКА (X, Y, Z)
    fig_channels = make_subplots(rows=1, cols=3, subplot_titles=("X Coord", "Y Coord", "Z Coord"))
    
    for i in range(3):
        # Ground Truth
        fig_channels.add_trace(go.Scatter(y=gt_full[i], mode='lines', name=f'GT {i}', 
                                         line=dict(color='green', width=1), opacity=0.3), row=1, col=i+1)
        # Context
        fig_channels.add_trace(go.Scatter(y=gt_full[i, :K], mode='lines', name=f'Context {i}', 
                                         line=dict(color='green', width=3)), row=1, col=i+1)
        # Prediction
        fig_channels.add_trace(go.Scatter(x=np.arange(K, total_len), y=pred_np[i, K:], 
                                         mode='lines', name=f'Pred {i}', 
                                         line=dict(color='red', dash='dash')), row=1, col=i+1)
        # Goal point
        fig_channels.add_trace(go.Scatter(x=[total_len-1], y=[goal_np[i]], 
                                         mode='markers', name=f'Goal {i}', 
                                         marker=dict(color='black', symbol='x', size=10)), row=1, col=i+1)

    fig_channels.update_layout(title_text="Trajectory Channels (Inference)", showlegend=False)
    fig_channels.show() # Откроется в браузере

    # 2. СОЗДАЕМ 3D ТРАЕКТОРИЮ
    fig_3d = go.Figure()

    # GT Trajectory
    fig_3d.add_trace(go.Scatter3d(x=gt_full[0], y=gt_full[1], z=gt_full[2],
                                 mode='lines', name='Ground Truth', line=dict(color='green', width=2)))
    # Prediction
    fig_3d.add_trace(go.Scatter3d(x=pred_np[0, K-1:], y=pred_np[1, K-1:], z=pred_np[2, K-1:],
                                 mode='lines', name='Prediction', line=dict(color='red', width=4, dash='dash')))
    # Goal Point
    fig_3d.add_trace(go.Scatter3d(x=[goal_np[0]], y=[goal_np[1]], z=[goal_np[2]],
                                 mode='markers', name='Goal', marker=dict(color='black', size=5, symbol='diamond')))

    fig_3d.update_layout(title="3D Flight Path", scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
    ))
    fig_3d.show() # Откроется во второй вкладке

if __name__ == "__main__":
    run_inference()

# def plot_results(pred, context, target, goal):
#     pred = pred.cpu().numpy()
#     gt = torch.cat([context, target], dim=-1).cpu().numpy()
#     K = context.shape[-1]
    
#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#     names = ['X', 'Y', 'Z']
#     for i in range(3):
#         axes[i].plot(gt[i], 'g', label='GT Full', alpha=0.3)
#         axes[i].plot(np.arange(K), gt[i, :K], 'g', linewidth=2, label='Context')
#         axes[i].plot(np.arange(K, gt.shape[-1]), pred[i, K:], 'r--', label='Pred')
#         axes[i].scatter([gt.shape[-1]-1], [goal[i].cpu().item()], color='black', marker='X', s=100)
#         axes[i].set_title(names[i])
#         axes[i].legend()
#     plt.show()

if __name__ == "__main__":
    run_inference()