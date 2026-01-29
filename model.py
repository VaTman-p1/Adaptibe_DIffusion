from comet_ml import Experiment
import time
import random
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import einops


from conv_blocks import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock
)


class FiLM(nn.Module):
    def __init__(self, cond_dim, out_channels):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2)
        )

    def forward(self, h, cond):
        scale, shift = torch.chunk(self.cond_mlp(cond), 2, dim=-1)
        h = h * (1 + scale[..., None]) + shift[..., None]
        return h


class ConditionEmbedder(nn.Module):
    def __init__(self, cond_dim=128, context_size=25):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4),
            nn.Mish(),
            nn.Linear(cond_dim * 4, cond_dim)
        )
        
        # 2. Энкодер цели (координаты XYZ)
        self.goal_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.Mish(),
            nn.Linear(64, cond_dim)
        )
        
        # 3. Энкодер контекста (препятствия/сцена)
        self.context_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context_size, 256),
            nn.Mish(),
            nn.Linear(256, cond_dim)
        )

    def forward(self, t, goal, context):
        t_e = self.time_mlp(t)
        g_e = self.goal_mlp(goal)
        c_e = self.context_mlp(context)
        
        return t_e, g_e, c_e

        
    


class FiLMResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()

        self.block1 = Conv1dBlock(in_ch, out_ch, 3)
        self.block2 = Conv1dBlock(out_ch, out_ch, 3)

        self.time_mlp = nn.Linear(embed_dim, out_ch)
        self.film = FiLM(embed_dim, out_ch)

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb):
        h = self.block1(x)
        h = h + self.time_mlp(emb)[..., None]
        h = self.film(h, emb)
        h = self.block2(h)
        return h + self.residual(x)



class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=[64, 128, 256],
        cond_dim = 128, num_cond = 2, 
    ):
        super().__init__()

        self.channels = channels
        self.in_channels = in_channels
        self.num_cond = num_cond
        self.cond_dim = cond_dim

        self.proj = nn.Linear(cond_dim*2 + cond_dim, cond_dim)

        self.cond_embedder = ConditionEmbedder(cond_dim=self.cond_dim)
        all_dims = [in_channels] + list(channels)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        # -------------------------------
        # ENCODER
        # -------------------------------
        self.down_blocks = nn.ModuleList()

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (len(channels) - 1)
            self.down_blocks.append(nn.ModuleList([
                FiLMResBlock(dim_in, dim_out, self.cond_dim),
                FiLMResBlock(dim_out, dim_out, self.cond_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
     

        # -------------------------------
        # BOTTLENECK
        # -------------------------------
        self.mid1 = FiLMResBlock(channels[-1], channels[-1], self.cond_dim)
        self.mid2 = FiLMResBlock(channels[-1], channels[-1], self.cond_dim)

        # -------------------------------
        # DECODER
        # -------------------------------
        self.up_blocks = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = i >= (len(in_out) - 1)

            self.up_blocks.append(nn.ModuleList([
                FiLMResBlock(dim_out * 2, dim_in, self.cond_dim),
                FiLMResBlock(dim_in, dim_in, self.cond_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        # -------------------------------
        # FINAL OUT
        # -------------------------------
        self.out = nn.Sequential(
            Conv1dBlock(channels[0], channels[0], 3),
            nn.Conv1d(channels[0], in_channels, 1)
        )

    def forward(self, x, t, goal, context):
        t_e, g_e, c_e = self.cond_embedder(t, goal, context)

        proj_cond = self.proj(torch.cat([t_e, g_e, c_e], dim=-1))

        # ---------------- ENCODER ----------------
        skips = []
        h = x
        for block1, block2, downsample in self.down_blocks:
            h = block1(h, proj_cond)
            h = block2(h, proj_cond)

            skips.append(h)
            h = downsample(h)

        # ---------------- MID ----------------
        h = self.mid1(h,proj_cond)
        h = self.mid2(h,proj_cond)


        
        # ---------------- DECODER ----------------
        for block1, block2, upsample in self.up_blocks:

            
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block1(h, proj_cond)
            h = block2(h, proj_cond)
            h = upsample(h)     


        return self.out(h)
    