import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from utils import pad_to_pow2

# -----------------------------------------------------------------------
# 1. Sinusoidal Time Embedding (Positional Encoding)
# -----------------------------------------------------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(half, device=device) / (half - 1))
        # t: [B] -> [B, 1]
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb

# -----------------------------------------------------------------------
# 2. AdaLN-1D (Adaptive Layer Norm для сверток)
# -----------------------------------------------------------------------
class AdaLN1D(nn.Module):
    """
    Adaptive Layer Normalization для тензоров формы [B, C, T].
    Эквивалентно LayerNorm, но без permute, реализовано через GroupNorm(1).
    Параметры (gamma, beta) предсказываются из вектора условий.
    """
    def __init__(self, num_channels, cond_dim=128):
        super().__init__()
        # GroupNorm с 1 группой = LayerNorm по каналам
        self.norm = nn.GroupNorm(1, num_channels, eps=1e-5, affine=False)
        
        # Проекция условия в параметры модуляции (Scale & Shift)
        self.emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_channels * 2)
        )
        
        # ВАЖНО: Инициализируем нулями. 
        # В начале обучения блок работает как Identity (gamma=0 -> scale=1, beta=0).
        nn.init.zeros_(self.emb[1].weight)
        nn.init.zeros_(self.emb[1].bias)

    def forward(self, x, cond):
        # x: [B, C, T]
        # cond: [B, cond_dim]
        
        # 1. Нормализуем вход (стандартное отклонение = 1, среднее = 0)
        h = self.norm(x)
        
        # 2. Предсказываем параметры сдвига и масштаба
        stats = self.emb(cond) # [B, 2*C]
        gamma, beta = stats.chunk(2, dim=1) # [B, C], [B, C]
        
        # 3. Решейпим для умножения: [B, C, 1] для бродкастинга по времени T
        gamma = gamma.unsqueeze(-1) + 1.0 # 1 + gamma (чтобы стартовать с единицы)
        beta = beta.unsqueeze(-1)
        
        # 4. Модуляция
        return h * gamma + beta

# -----------------------------------------------------------------------
# 3. Residual Block с AdaLN
# -----------------------------------------------------------------------
class AdaResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = AdaLN1D(out_channels, cond_dim)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = AdaLN1D(out_channels, cond_dim)
        
        self.act = nn.SiLU()
        self.skip = (nn.Conv1d(in_channels, out_channels, 1) 
                     if in_channels != out_channels else nn.Identity())

    def forward(self, x, cond):
        # Структура: Conv -> AdaLN -> Act -> Conv -> AdaLN -> Act + Skip
        h = self.conv1(x)
        h = self.norm1(h, cond)
        h = self.act(h)

        h = self.conv2(h)
        h = self.norm2(h, cond)
        
        return self.act(h + self.skip(x))

# -----------------------------------------------------------------------
# 4. Основная модель UNet1D (AdaLN Edition)
# -----------------------------------------------------------------------
class UNet1D_AdaLN(nn.Module):
    def __init__(self, 
                 in_channels=8, 
                 channels: List[int] = [64, 128, 256], 
                 cond_dim=128):
        super().__init__()
        
        self.target_len = 32  # Целевая длина последовательности (степень двойки)

        # --- Embedding Blocks (как в твоем примере) ---
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim)
        )

        self.goal_emb = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, cond_dim)
        )

        # Context (Flatten -> Linear) подходит для фиксированной короткой истории
        self.context_emb = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(25, 128), # Предполагаем вход 5x5 или 25 flat
            nn.SiLU(),
            nn.Linear(128, cond_dim)
        )
        
        self.force_emb = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, cond_dim)
        )

        # --- Fusion ---
        # Вход: 4 вектора по cond_dim. Выход: 1 вектор cond_dim
        self.cond_fusion = nn.Sequential(
            nn.Linear(cond_dim * 4, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim)
        )

        # --- Encoder ---
        self.down_blocks = nn.ModuleList()
        prev = in_channels
        for c in channels:
            self.down_blocks.append(nn.ModuleDict({
                'res': AdaResBlock1D(prev, c, cond_dim=cond_dim),
                'down': nn.Conv1d(c, c, kernel_size=4, stride=2, padding=1)
            }))
            prev = c

        # --- Mid ---
        self.mid_res1 = AdaResBlock1D(prev, prev, cond_dim=cond_dim)
        self.mid_res2 = AdaResBlock1D(prev, prev, cond_dim=cond_dim)

        # --- Decoder ---
        self.up_blocks = nn.ModuleList()
        for c in reversed(channels):
            self.up_blocks.append(nn.ModuleDict({
                'up': nn.ConvTranspose1d(prev, c, kernel_size=4, stride=2, padding=1),
                'res': AdaResBlock1D(c * 2, c, cond_dim=cond_dim)
            }))
            prev = c

        self.final_conv = nn.Conv1d(channels[0], in_channels, kernel_size=1)

    def forward(self, x, t, goal, context, force):
        """
        x:       [B, in_channels, T_orig] (T=21)
        t:       [B] (индексы времени)
        goal:    [B, 3]
        context: [B, 25] (или [B, 5, 5])
        force:   [B, 3]
        """
        # 1. Автоматический Паддинг (Replication)
        orig_T = x.shape[-1]
        x, mask, last_point = pad_to_pow2(x)

        # 2. Подготовка условий
        t_e = self.time_emb(t)
        g_e = self.goal_emb(goal)
        c_e = self.context_emb(context) # Flat input expected
        f_e = self.force_emb(force)

        # 3. Fusion (concat всех 4-х условий)
        cond = self.cond_fusion(torch.cat([t_e, g_e, c_e, f_e], dim=-1))

        # 4. Проход по UNet
        skips = []
        h = x
        
        # Encoder
        for layer in self.down_blocks:
            h = layer['res'](h, cond)
            skips.append(h)
            h = layer['down'](h)

        # Mid
        h = self.mid_res1(h, cond)
        h = self.mid_res2(h, cond)

        # Decoder
        for layer in self.up_blocks:
            h = layer['up'](h)
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = layer['res'](h, cond)

        # 5. Финальная свертка и обрезка
        out = self.final_conv(h)
        
        # Возвращаем только исходную длину
        return out