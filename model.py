from comet_ml import Experiment
import time
import random
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import numpy as np
import pandas as pd
import torch
import torch.nn as nn



class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(-torch.log(torch.tensor(10000.0, device=device)) * torch.arange(half, device=device) / (half - 1))
        angles = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return emb
    
class ConditionEmbedder(nn.Module):
    def __init__(self, cond_dim=128, context_size=25, num_cond = 3):
        super().__init__()
        
        # 1. Энкодер времени (стандарт для диффузии)
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4),
            nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim)
        )
        
        # 2. Энкодер цели (координаты XYZ)
        self.goal_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, cond_dim)
        )
        
        # 3. Энкодер контекста (препятствия/сцена)
        self.context_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context_size, 256),
            nn.SiLU(),
            nn.Linear(256, cond_dim)
        )
        
        # 4. Финальный Fusion MLP
        # Смешивает все эмбеды в один вектор нужной размерности
        self.fusion = nn.Sequential(
            nn.Linear(cond_dim * num_cond, cond_dim * 2),
            nn.SiLU(),
            nn.Linear(cond_dim * 2, cond_dim) 
        )

    def forward(self, t, goal, context):
        t_e = self.time_mlp(t)
        g_e = self.goal_mlp(goal)
        c_e = self.context_mlp(context)
        
        # Объединяем и перемешиваем
        combined = torch.cat([t_e, g_e, c_e], dim=-1)
        return self.fusion(combined)

class AdaGNResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim_total, dropout=0.1):
        super().__init__()
        # Вектор условий (t + goal + context + force) -> [B, 512]
        # Мы проецируем его сразу в параметры для нормализации
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim_total, out_channels * 2) 
        )
        def get_groups(channels):
            if channels % 8 == 0:
                return 8
            # Если совсем мало или странное число (как 5), 
            # используем 1 группу (фактически LayerNorm)
            return 1


        self.norm1 = nn.GroupNorm(get_groups(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)

        self.dropout = nn.Dropout(dropout)
        
        self.norm2 = nn.GroupNorm(get_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, fused_cond):
        # 1. Первый этап свертки
        h = self.norm1(x)
        h = self.conv1(self.act(h))

        # 2. Генерируем scale (gamma) и shift (beta) из t и остальных условий
        # Мы используем .unsqueeze(-1) для применения ко всей длине траектории
        cond_params = self.cond_mlp(fused_cond).unsqueeze(-1)
        scale, shift = torch.chunk(cond_params, 2, dim=1)
        
        # 3. Адаптивная нормализация (вместо обычного FiLM)
        h = h * (1 + scale) + shift
        h = self.dropout(h)

        # 4. Второй этап
        h = self.norm2(h)
        h = self.conv2(self.act(h))

        return h + self.skip(x)



class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=[64, 128, 256],
        cond_dim = 128, num_cond = 3
    ):
        super().__init__()

        self.channels = channels

        self.cond_embedder = ConditionEmbedder(cond_dim=cond_dim, num_cond=num_cond)

        # -------------------------------
        # ENCODER
        # -------------------------------
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        prev_c = in_channels
        for c in channels:
            self.down_blocks.append(
                AdaGNResBlock(prev_c, c, cond_dim)
            )
            self.downsamples.append(
                nn.Conv1d(c, c, 4, stride=2, padding=1)
            )
            prev_c = c

        # -------------------------------
        # BOTTLENECK
        # -------------------------------
        self.mid1 = AdaGNResBlock(channels[-1], channels[-1], cond_dim)
        self.mid2 = AdaGNResBlock(channels[-1], channels[-1], cond_dim)

        # -------------------------------
        # DECODER
        # -------------------------------
        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        in_layer = channels[-1]
        for c in reversed(channels):
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
                    nn.Conv1d(in_layer, c, 3, padding=1)
                )
            )
            self.up_blocks.append(
                AdaGNResBlock(c * 2, c, cond_dim)
            )
            in_layer=c

        # -------------------------------
        # FINAL OUT
        # -------------------------------
        self.out = nn.Conv1d(channels[0], in_channels, 1)

    def forward(self, x, t, goal, context):
        # print('.')
        fused_cond = self.cond_embedder(t, goal, context)

        # ---------------- ENCODER ----------------
        skips = []
        h = x
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, fused_cond)
            skips.append(h)
            h = down(h)

        # ---------------- MID ----------------
        h = self.mid1(h,fused_cond)
        h = self.mid2(h,fused_cond)
        
        # ---------------- DECODER ----------------
        for upsample, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = upsample(h)
            h = torch.cat([h, skip], dim=1)
            h = block(h, fused_cond)


        return self.out(h)
