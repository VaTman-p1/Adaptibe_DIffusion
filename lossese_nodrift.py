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

def goal_loss(x0_pred, goal, last_point):
    return F.mse_loss(torch.clamp(x0_pred[:, :3, last_point],-1.0,1.0), goal)  # по умолчанию reduction='mean'


def flow_matching_loss(v_pred, v_target, mask):
    """
    MSE Loss для векторного поля.
    v_target = x_data - x_noise
    """
    weight = mask.expand_as(v_pred)
    loss = F.mse_loss(v_pred, v_target, reduction='none')
    return (loss * weight).sum() / (weight.sum() + 1e-8)
