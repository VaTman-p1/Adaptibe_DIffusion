import torch


def reconstruct_x0(x_t, eps, t, scheduler):
    alpha_bar = scheduler.alphas_cumprod[t].view(-1, 1, 1).to(x_t.device)
    return (x_t - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
