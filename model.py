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
    def __init__(self, cond_dim=128, context_size=25):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(cond_dim),
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
    def __init__(self, in_channels, out_channels, conddim, condtype = 'concat'):
        super().__init__()
        self.condtype = condtype
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(conddim, out_channels * 2) 
        )
        def get_groups(channels):
            if channels % 8 == 0:
                return 8
            return 1
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        
        self.norm = nn.GroupNorm(get_groups(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        
        self.act = nn.Mish()
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):

        h = self.conv1(x)
        h = self.norm(h)
        h = self.act(h)
        
        # cond_params = self.cond_mlp(fused_cond).unsqueeze(-1)
        cond_params = self.cond_mlp(cond)
        if self.condtype == 'concat':
            scale, shift = torch.chunk(cond_params.unsqueeze(-1), 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            sum_cond = torch.sum(cond_params, dim = 1)
            scale, shift = torch.chunk(sum_cond.unsqueeze(-1), 2, dim=1)
            h = h * (1 + scale) + shift


        h = self.conv2(h)
        h = self.norm(h)
        h = self.act(h)

        return h + self.skip(x)



class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels=2,
        channels=[64, 128, 256],
        cond_dim = 128, num_cond = 3, 
        condtype = 'concat'
    ):
        super().__init__()

        self.channels = channels
        self.condtype = condtype
        self.in_channels = in_channels
        self.num_cond = num_cond
        self.cond_dim = cond_dim

        if self.condtype == 'concat':
            cond_dim = int(num_cond*self.cond_dim)

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
                FiLMResBlock(dim_in, dim_out, cond_dim), 
                nn.Conv1d(dim_out, dim_out, 3, stride=2, padding=1) if not is_last else nn.Identity()
            ]))
     

        # -------------------------------
        # BOTTLENECK
        # -------------------------------
        self.mid1 = FiLMResBlock(channels[-1], channels[-1], cond_dim, condtype=self.condtype)
        self.mid2 = FiLMResBlock(channels[-1], channels[-1], cond_dim)

        # -------------------------------
        # DECODER
        # -------------------------------
        self.up_blocks = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out)):

        
            is_first = (i == 0)
            self.up_blocks.append(
                nn.ModuleList([
                    nn.Upsample(scale_factor=2, mode='linear', align_corners=False) if not is_first else nn.Identity(),
                    FiLMResBlock(dim_out + dim_out, dim_in, cond_dim)
                ])
            )
        # -------------------------------
        # FINAL OUT
        # -------------------------------
        self.out = nn.Conv1d(in_channels, in_channels, 1)

    def forward(self, x, t, goal, context):
        t_e, g_e, c_e = self.cond_embedder(t, goal, context)
        if self.condtype == 'concat':
            fused_cond = torch.cat([t_e, g_e, c_e], dim=1)
        else:
            fused_cond = torch.stack([t_e, g_e, c_e], dim=1)

        # ---------------- ENCODER ----------------
        skips = []
        h = x
        for block, downsample in self.down_blocks:
            h = block(h, fused_cond)
            # print(h.size())
            skips.append(h)
            h = downsample(h)
            # print(h.size())
            # print('-'*50)
        # print('skip sizes:', [sk.size() for sk in skips])
        # ---------------- MID ----------------
        h = self.mid1(h,fused_cond)
        h = self.mid2(h,fused_cond)


        
        # ---------------- DECODER ----------------
        for upsample, block in self.up_blocks:
            h = upsample(h)
            skip = skips.pop()
            # print(h.size(), skip.size())
            h = torch.cat([h, skip], dim=1)
            h = block(h, fused_cond)


        return self.out(h)
