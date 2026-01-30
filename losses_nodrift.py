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

def yaw_loss_cosine(x0_pred, x0_true, mask):
    # Извлекаем sin (index 3) и cos (index 4)
    pred = x0_pred[:, 3:5]
    true = x0_true[:, 3:5]
    
    # Обязательно нормируем предсказания, чтобы они лежали на единичной окружности
    pred = F.normalize(pred, p=2, dim=1)
    true = F.normalize(true, p=2, dim=1) # true уже должен быть нормирован, но для страховки можно
    
    # Скалярное произведение: sin1*sin2 + cos1*cos2 = cos(alpha - beta)
    cos_diff = (pred * true).sum(dim=1)
    
    # Мы хотим, чтобы cos_diff стремился к 1, значит loss стремится к 0
    loss = 1.0 - cos_diff
    
    # Применяем маску
    weight = mask[:, 0]
    loss = (loss * weight).sum() / (weight.sum() + 1e-8)
    return loss

def path_loss(x0_pred, mask):
    diffs = x0_pred[:, :3, 1:] - x0_pred[:, :3, :-1]
    target = torch.zeros_like(diffs)
    
    weight = mask[:, :, 1:].expand_as(diffs)
    loss = F.smooth_l1_loss(diffs, target, reduction='none') #хз попробуем
    loss = loss * weight
    return loss.sum() / (weight.sum() + 1e-8)

def goal_loss(x0_pred, goal, last_point_indices, tolerance=0.01):
    batch_size = x0_pred.size(0)
    # Извлекаем предсказанный конец
    pred_ends = x0_pred[torch.arange(batch_size), :3, last_point_indices]
    
    dist = torch.norm(pred_ends - goal, dim=1) 
    reg_dist = torch.clamp(dist - tolerance, min=0.0)
    
    # Используем reduction='none', чтобы получить лосс для каждого примера в батче
    # Это позволит нам применить маску t снаружи
    loss_per_sample = F.smooth_l1_loss(reg_dist, torch.zeros_like(reg_dist), beta=0.5, reduction='none')
    
    return loss_per_sample


def flow_matching_loss(v_pred, v_target, mask):
    """
    MSE Loss для векторного поля.
    v_target = x_data - x_noise
    """
    weight = mask.expand_as(v_pred)
    loss = F.mse_loss(v_pred, v_target, reduction='none')
    return (loss * weight).sum() / (weight.sum() + 1e-8)


def smoothness_loss(x0_pred, mask):
    coords = x0_pred[:, :3, :]
    # Ускорение (вторая разность)
    accel = coords[:, :, 2:] - 2 * coords[:, :, 1:-1] + coords[:, :, :-2]
    
    # Маска паддинга (учитываем, что длина стала T-2)
    m = mask[:, :, 2:].expand_as(accel)
    
    # Считаем MSE, но НЕ усредняем по батчу (dim=0 оставляем)
    # loss_per_sample будет иметь размер [B]
    loss_per_sample = (F.mse_loss(accel * m, torch.zeros_like(accel), reduction='none')).sum(dim=(1, 2)) 
    # Нормируем на количество непустых точек в каждом примере
    loss_per_sample = loss_per_sample / (m.sum(dim=(1, 2)) + 1e-8)
    
    return loss_per_sample