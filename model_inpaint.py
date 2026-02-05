import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from conv_blocks import SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock

# -----------------------------------------------------------------------------
# 1. Эмбеддер времени (без изменений)
# -----------------------------------------------------------------------------
class TimeEmbedder(nn.Module):
    def __init__(self, cond_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim)
        )

    def forward(self, t):
        return self.time_mlp(t)

# -----------------------------------------------------------------------------
# 2. Блок с аддитивным добавлением времени
# -----------------------------------------------------------------------------
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()

        self.block1 = Conv1dBlock(in_ch, out_ch, 3)
        self.block2 = Conv1dBlock(out_ch, out_ch, 3)

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_ch),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x) + self.time_mlp(t_emb)
        h = self.block2(h)
        return h + self.residual(x)

# -----------------------------------------------------------------------------
# 3. UNet1D (Конкатенация по каналам)
# -----------------------------------------------------------------------------
class UNet1D(nn.Module):
    def __init__(self, 
                 in_channels=5, 
                 goal_dim=3,
                 channels=[64, 128, 256], 
                 cond_dim=128):
        super().__init__()
        self.in_channels = in_channels
        self.goal_dim = goal_dim
        
        self.time_embedder = TimeEmbedder(cond_dim=cond_dim)
        
        # Ключевой момент: на вход подаем сумму каналов траектории и цели (5 + 3 = 8)
        input_dim = in_channels + goal_dim 
        
        all_dims = [input_dim] + list(channels)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # ENCODER
        self.down_blocks = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (len(channels) - 1)
            self.down_blocks.append(nn.ModuleList([
                ResBlock1D(dim_in, dim_out, cond_dim),
                ResBlock1D(dim_out, dim_out, cond_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # BOTTLENECK
        mid_dim = channels[-1]
        self.mid1 = ResBlock1D(mid_dim, mid_dim, cond_dim)
        self.mid2 = ResBlock1D(mid_dim, mid_dim, cond_dim)

        # DECODER
        self.up_blocks = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.up_blocks.append(nn.ModuleList([
                ResBlock1D(dim_out * 2, dim_in, cond_dim),
                ResBlock1D(dim_in, dim_in, cond_dim),
                Upsample1d(dim_in) if i < len(channels) - 1 else nn.Identity()
            ]))

        # FINAL: возвращаем обратно к 5 каналам (предсказание только для траектории)
        self.final_out = nn.Sequential(
            Conv1dBlock(channels[0], channels[0], 3),
            nn.Conv1d(channels[0], in_channels, 1)
        )

    def forward(self, x, t, goal):
        """
        x: [B, 5, L] - зашумленная траектория (L должно делиться на 8)
        t: [B]       - временной шаг диффузии
        goal: [B, 3] - координаты цели
        """
        # 1. Эмбеддинг времени
        t_emb = self.time_embedder(t)

        # 2. Растягиваем Goal по всей длине последовательности
        # [B, 3] -> [B, 3, 1] -> [B, 3, L]
        B, _, L = x.shape
        goal_feat = goal.unsqueeze(-1).expand(-1, -1, L)

        # 3. Склеиваем по каналам (dim=1)
        # Получаем тензор [B, 8, L]
        h = torch.cat([x, goal_feat], dim=1)

        # 4. Проход через UNet
        skips = []
        
        # Encoder
        for block1, block2, downsample in self.down_blocks:
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            skips.append(h)
            h = downsample(h)

        # Bottleneck
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        # Decoder
        for block1, block2, upsample in self.up_blocks:
            skip = skips.pop()
            
            # Проверка на случай нечетной длины после паддинга
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-1], mode='nearest')
                
            h = torch.cat([h, skip], dim=1)
            h = block1(h, t_emb)
            h = block2(h, t_emb)
            h = upsample(h)

        # 5. Результат: [B, 5, L]
        return self.final_out(h)