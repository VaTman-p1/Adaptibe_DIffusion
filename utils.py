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
