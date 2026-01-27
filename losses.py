import torch
import torch.nn.functional as F

def diffusion_loss(pred, noise, mask):
    """
    Masked MSE на шум.
    """
    
    weight = mask.expand_as(pred)  # [B, 1, T] -> [B, C, T]
    loss = F.mse_loss(pred, noise, reduction='none')  # [B, C, T]
    loss = loss * weight
    return loss.sum() / (weight.sum() + 1e-8)

def yaw_loss(x0_pred, x0_true, mask):
    pred = x0_pred[:, 3:5]
    true = x0_true[:, 3:5]
    pred = F.normalize(pred, dim=1)
    true = F.normalize(true, dim=1)
    
    weight = mask[:, :1].expand_as(pred)
    loss = F.mse_loss(pred, true, reduction='none')
    loss = loss * weight
    return loss.sum() / (weight.sum() + 1e-8)

def path_loss(x0_pred, mask):
    diffs = x0_pred[:, :3, 1:] - x0_pred[:, :3, :-1]
    target = torch.zeros_like(diffs)
    
    weight = mask[:, :, 1:].expand_as(diffs)
    loss = F.smooth_l1_loss(diffs, target, reduction='none') #хз попробуем
    loss = loss * weight
    return loss.sum() / (weight.sum() + 1e-8)

def goal_loss(x0_pred, goal, last_point_indices, tolerance=0.01):
    """
    Мягкая регуляризация цели.
    tolerance: радиус (в метрах/единицах), внутри которого лосс зануляется.
    """
    batch_size = x0_pred.size(0)
    # Извлекаем предсказанный конец: [B, 3]
    pred_ends = x0_pred[torch.arange(batch_size), :3, last_point_indices]
    
    # Считаем евклидово расстояние до цели
    dist = torch.norm(pred_ends - goal, dim=1) # [B]
    
    # Применяем deadzone: если dist < tolerance, то 0
    reg_dist = torch.clamp(dist - tolerance, min=0.0)
    
    # Используем среднее по батчу от квадратов (или просто среднее)
    # Huber-style:
    loss = F.smooth_l1_loss(reg_dist, torch.zeros_like(reg_dist), beta=0.5)
    
    return loss




def drift_consistency_loss(x0_pred, mask):
    """
    Дрейф должен быть почти постоянным по траектории
    (штраф за изменение скорости дрейфа по шагам)
    """
    pred_drift_per_step = x0_pred[:, 5:8] - x0_pred[:, :3]  # [B, T, 3]
    pred_drift_per_step = torch.clamp(pred_drift_per_step, min=-1.0, max=1.0)
    # print("x0_pred shape:", x0_pred.shape)   # должно быть [B, T, 8] или [B, 8, T]
    # print("pred_drift_per_step shape:", pred_drift_per_step.shape)
    
    # Средний дрейф по всей траектории
    avg_drift = pred_drift_per_step.mean(dim=1, keepdim=True)  # [B, 1, 3]
    consistent_drift = avg_drift.repeat(1, pred_drift_per_step.shape[1], 1)
    # print("consistent_drift shape:", consistent_drift.shape)
    
    weight = mask.expand_as(pred_drift_per_step)
    loss = F.mse_loss(pred_drift_per_step, consistent_drift, reduction='none')
    loss = loss * weight
    
    return torch.clamp((loss.sum() / (weight.sum() + 1e-8)), -1.0, 1.0)


def internal_consistency_loss(
    x0_pred, 
    mask,
    scaled_drift,            
    clamp_min=-1.0,
    clamp_max=1.0
):

    t = torch.arange(x0_pred.shape[-1], device=x0_pred.device, dtype=torch.float32)
    t = t.view(1, 1, -1) 
    if scaled_drift.dim() == 2:
        scaled_drift = scaled_drift.unsqueeze(-1)  
    cumulative_drift = scaled_drift * t  

    recovered_clean = x0_pred[:, 5:8] + cumulative_drift
    weight = mask.expand_as(recovered_clean)  # [B, T, 3]

    loss = F.mse_loss(recovered_clean, x0_pred[:, :3], reduction='none')
    loss = loss * weight

    return torch.clamp(loss.sum() / (weight.sum() + 1e-8), min=clamp_min, max=clamp_max)