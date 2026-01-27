import plotly.graph_objects as go
from plotly.offline import iplot
import numpy as np
from datasets import DroneTrajectoryDataset

def visualize_trajectory_with_drift(sample, title="Траектория дрона: контекст + идеальная + сдвинутая"):
    """
    Визуализирует:
    1. Контекст (синий)
    2. Идеальную цель (tgt_pos) — красный
    3. Сдвинутую цель (tgt_drift) — фиолетовый/пурпурный
    4. Направления yaw (стрелки)
    """
    # Извлекаем данные из sample
    context = sample['context'].numpy()      # [C, 5] = [x,y,z, sin_yaw, cos_yaw]
    target = sample['target'].numpy()        # [T, 8] = [x,y,z, sin, cos, dx, dy, dz]
    goal = sample['goal'].numpy()            # [3]
    scale = sample['scale'].numpy()          # [3]

    # Позиции
    ctx_pos = context[:, :3]
    tgt_pos_clean = target[:, :3]            # идеальная траектория (ground truth)
    tgt_pos_drift = target[:, 5:8]  # восстанавливаем сдвинутую позицию
    # (если в target хранится чистая + дрейф как смещение)

    # Альтернативный способ — если хочешь использовать напрямую tgt_drift из raw_sample
    # tgt_pos_drift = sample.get('tgt_drift', tgt_pos_clean + target[:, 5:8]).numpy()

    # Yaw из sin/cos (в радианах)
    ctx_yaw = np.arctan2(context[:, 3], context[:, 4])
    tgt_yaw_clean = np.arctan2(target[:, 3], target[:, 4])

    # Эгоцентр — последняя точка контекста
    ego_pos = ctx_pos[-1]
    ego_yaw = ctx_yaw[-1]   # в реальности обычно ≈0 после трансформации в локальную

    # Диапазон для графика
    all_pos = np.vstack([ctx_pos, tgt_pos_clean, tgt_pos_drift, goal[None, :]])
    padding = np.max(np.abs(all_pos)) * 0.08
    min_v = np.min(all_pos, axis=0) - padding
    max_v = np.max(all_pos, axis=0) + padding

    fig = go.Figure()

    # 1. Контекст
    fig.add_trace(go.Scatter3d(
        x=ctx_pos[:,0], y=ctx_pos[:,1], z=ctx_pos[:,2],
        mode='lines+markers',
        name='Контекст',
        line=dict(color='royalblue', width=5),
        marker=dict(size=7, color='royalblue', symbol='circle'),
        hoverinfo='text',
        text=[f'Контекст t={i}, yaw={np.degrees(y):.1f}°' 
              for i, y in enumerate(ctx_yaw)]
    ))

    # 2. Идеальная цель (чистая)
    fig.add_trace(go.Scatter3d(
        x=tgt_pos_clean[:,0], y=tgt_pos_clean[:,1], z=tgt_pos_clean[:,2],
        mode='lines+markers',
        name='Идеальная цель',
        line=dict(color='crimson', width=5, dash='dash'),
        marker=dict(size=7, color='crimson'),
        hoverinfo='text',
        text=[f'pos=({tgt_pos_clean[:,0][i]},{tgt_pos_clean[:,1][i]},{tgt_pos_clean[:,2][i]}' for i in range(len(tgt_pos_drift[:,0]))]
    ))

    # 3. Сдвинутая цель (с дрейфом)
    fig.add_trace(go.Scatter3d(
        x=tgt_pos_drift[:,0], y=tgt_pos_drift[:,1], z=tgt_pos_drift[:,2],
        mode='lines+markers',
        name='Сдвинутая цель (с дрейфом)',
        line=dict(color='purple', width=5),
        marker=dict(size=7, color='purple'),
        hoverinfo='text',
        text=[f'pos=({tgt_pos_drift[:,0][i]},{tgt_pos_drift[:,1][i]},{tgt_pos_drift[:,2][i]}' for i in range(len(tgt_pos_drift[:,0]))]
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

    # 6. Стрелки направления yaw
    def add_yaw_arrows(pos, yaws, color, name_prefix):
        dx = np.cos(yaws) * 0.05   # длина стрелки
        dy = np.sin(yaws) * 0.05
        dz = np.zeros_like(yaws)
        for i, (p, ux, uy, uz) in enumerate(zip(pos, dx, dy, dz)):
            fig.add_trace(go.Scatter3d(
                x=[p[0], p[0]+ux],
                y=[p[1], p[1]+uy],
                z=[p[2], p[2]+uz],
                mode='lines',
                line=dict(color=color, width=5),
                name=f"{name_prefix} {i}" if i == 0 else None,
                showlegend=(i==0),
                hoverinfo='skip'
            ))

    add_yaw_arrows(ctx_pos, ctx_yaw, 'limegreen', 'Контекст yaw')
    add_yaw_arrows(tgt_pos_clean, tgt_yaw_clean, 'orange', 'Идеал yaw')
    add_yaw_arrows(tgt_pos_drift, tgt_yaw_clean, 'magenta', 'Дрейф yaw')

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
        width=1100,
        height=900,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, b=0, t=50)
    )

    iplot(fig)

dsd1 = DroneTrajectoryDataset('trajectory_diffusion/V2_01_easy.csv')

sample = dsd1.get_sample(
    idx=200,
    custom_raw_drift=np.array([1, 1, 1]),
    normalize=False
)

visualize_trajectory_with_drift(sample, title="Пример траектории с дрейфом №500")