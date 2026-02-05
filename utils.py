import torch
import torch.nn.functional as F


def pad_to_pow2(x):
    B, C, T = x.shape
    if (T & (T - 1)) == 0:
        return x, torch.ones(B, 1, T, device=x.device), T-1

    T2 = 1 << (T - 1).bit_length()
    last_point = T-1
    pad = T2 - T

    mask = torch.zeros(B, 1, T2, device=x.device)
    mask[:, :, :T] = 1.0

    x = F.pad(x, (0, pad), mode="replicate")
    return x, mask, last_point

def pad_to_pow2_cond(x, context_len):
    """
    x: [B, 5, T_total] (уже склеенная лента: context + target)
    context_len: K (длина истории)
    """
    B, C, T = x.shape
    
    # Находим ближайшую степень двойки вверх
    T2 = 1 << (T - 1).bit_length()
    pad_len = T2 - T
    
    # Маска: 1 только там, где таргет (от конца контекста до конца реальных данных)
    mask = torch.zeros(B, 1, T2, device=x.device)
    mask[:, :, context_len:T] = 1.0
    
    # Индекс последней реальной точки (для goal_loss)
    last_idx = T - 1

    # Паддинг нулями (константа). Маска исключит их из лосса.
    # Если забивать репликацией, модель будет путаться, где реальная цель.
    if pad_len > 0:
        x = F.pad(x, (0, pad_len), mode="constant", value=0.0)
        
    return x, mask, last_idx


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
