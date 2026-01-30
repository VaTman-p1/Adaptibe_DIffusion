import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# -----------------------------------------------------------------------------
# 1. Позиционное кодирование времени (для диффузии)
# -----------------------------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# -----------------------------------------------------------------------------
# 2. Блоки FiLM и ResNet
# -----------------------------------------------------------------------------
class FiLM(nn.Module):
    def __init__(self, cond_dim, out_channels):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2)
        )

    def forward(self, h, cond):
        # h: [B, C, T], cond: [B, cond_dim]
        stats = self.cond_mlp(cond).unsqueeze(-1) # [B, 2*C, 1]
        scale, shift = torch.chunk(stats, 2, dim=1)
        return h * (1 + scale) + shift

class FiLMResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.Mish()
        )
        self.film = FiLM(cond_dim, out_ch)
        self.block2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8, out_ch),
            nn.Mish(),
            nn.Dropout(dropout)
        )
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond):
        h = self.block1(x)
        h = self.film(h, cond)
        h = self.block2(h)
        return h + self.residual(x)

# -----------------------------------------------------------------------------
# 3. Temporal History Encoder (Вместо простого Flatten)
# -----------------------------------------------------------------------------
class HistoryEncoder(nn.Module):
    """Извлекает признаки динамики из 5 последних стейтов"""
    def __init__(self, in_channels=5, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            # Вход: [B, 5 признаков, 5 шагов]
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.out = nn.Linear(128, out_dim)

    def forward(self, x):
        # Если x пришел как [B, 25], превращаем в [B, 5, 5]
        if len(x.shape) == 2:
            x = x.view(-1, 5, 5) 
        return self.out(self.net(x))

# -----------------------------------------------------------------------------
# 4. Основная UNet1D модель
# -----------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=5, channels=[64, 128, 256], cond_dim=128, dropout=0.1):
        super().__init__()
        
        # --- Эмбеддинги условий ---
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4),
            nn.Mish(),
            nn.Linear(cond_dim * 4, cond_dim)
        )
        self.goal_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.Mish(),
            nn.Linear(64, cond_dim)
        )
        self.history_enc = HistoryEncoder(in_channels=in_channels, out_dim=cond_dim)
        
        # Слияние всех условий в один вектор для FiLM
        self.cond_fusion = nn.Sequential(
            nn.Linear(cond_dim * 3, cond_dim * 2),
            nn.Mish(),
            nn.Linear(cond_dim * 2, cond_dim)
        )

        # --- Путь UNet ---
        channels = [in_channels]+channels

        # Encoder (Downsampling)
        self.downs = nn.ModuleList()
        for i in range(len(channels)-1):
            self.downs.append(nn.ModuleList([
                FiLMResBlock(channels[i], channels[i+1], cond_dim, dropout),
                FiLMResBlock(channels[i+1], channels[i+1], cond_dim, dropout),
                nn.Conv1d(channels[i+1], channels[i+1], 3, stride=2, padding=1)
            ]))

        # Bottleneck
        mid_dim = channels[-1]
        self.mid1 = FiLMResBlock(mid_dim, mid_dim, cond_dim, dropout)
        self.mid2 = FiLMResBlock(mid_dim, mid_dim, cond_dim, dropout)

        # Decoder (Upsampling)
        self.ups = nn.ModuleList()
        rev_channels = list(reversed(channels[1:]))
        for i in range(len(rev_channels) - 1):
            in_ch = rev_channels[i]
            out_ch = rev_channels[i+1]
            self.ups.append(nn.ModuleList([
                FiLMResBlock(in_ch + out_ch, out_ch, cond_dim, dropout),
                FiLMResBlock(out_ch, out_ch, cond_dim, dropout),
                nn.ConvTranspose1d(out_ch, out_ch, 4, stride=2, padding=1)
            ]))

        # Финальный выход
        self.final_conv = nn.Sequential(
            nn.Conv1d(channels[0], channels[0], 3, padding=1),
            nn.GroupNorm(8, channels[0]),
            nn.Mish(),
            nn.Conv1d(channels[0], in_channels, 1)
        )

    def forward(self, x, t, goal, history):
        """
        x: [B, 5, 32] - зашумленная траектория (уже дополненная до 32)
        t: [B] - таймстеп
        goal: [B, 3] - x, y, z цели
        history: [B, 25] - 5 прошлых стейтов
        """
        # 1. Собираем условие
        t_e = self.time_mlp(t)
        g_e = self.goal_mlp(goal)
        h_e = self.history_enc(history)
        cond = self.cond_fusion(torch.cat([t_e, g_e, h_e], dim=-1))

        # 2. UNet
        h = self.init_conv(x)
        skips = []

        # Encoder
        for res1, res2, down in self.downs:
            h = res1(h, cond)
            h = res2(h, cond)
            print(h.size())
            skips.append(h)
            h = down(h)
            print(h.size())

        # Mid
        h = self.mid1(h, cond)
        h = self.mid2(h, cond)

        # Decoder
        for res1, res2, up in self.ups:
            h = up(h)
            print(h.size)
            skip = skips.pop()
            
            h = torch.cat([h, skip], dim=1)
            h = res1(h, cond)
            h = res2(h, cond)

        return self.final_conv(h)