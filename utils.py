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
