import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from conv_blocks import SinusoidalPosEmb, Downsample1d, Upsample1d, Conv1dBlock


# -----------------------------------------------------------------------------
# 2. Модуляция и Условия
# -----------------------------------------------------------------------------

class FiLM(nn.Module):
    def __init__(self, cond_dim, out_channels):
        super().__init__()
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2)
        )
        self.cond_mlp[-1].weight.data.zero_()
        self.cond_mlp[-1].bias.data.zero_()


    def forward(self, h, cond):
        scale, shift = torch.chunk(self.cond_mlp(cond), 2, dim=-1)
        h = h * (1 + scale[..., None]) + shift[..., None]
        return h

class HistoryEncoder(nn.Module):
    """Сверточный энкодер для 5 стейтов истории (вместо Flatten)"""
    def __init__(self, in_channels=5, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.out = nn.Linear(128, out_dim)

    def forward(self, x):
        x=x.permute(0,2,1)

        return self.out(self.net(x))

class ConditionEmbedder(nn.Module):
    def __init__(self, cond_dim=128, context_size=25, in_channels=5):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(cond_dim),
            nn.Linear(cond_dim, cond_dim * 4), nn.SiLU(),
            nn.Linear(cond_dim * 4, cond_dim)
        )
        self.goal_mlp = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, cond_dim)
        )
        self.context_encoder = HistoryEncoder(in_channels=in_channels, out_dim=cond_dim)

    def forward(self, t, goal, context):
        return self.time_mlp(t), self.goal_mlp(goal), self.context_encoder(context)

# -----------------------------------------------------------------------------
# 3. Основные Блоки UNet
# -----------------------------------------------------------------------------

class FiLMResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, embed_dim):
        super().__init__()

        self.block1 = Conv1dBlock(in_ch, out_ch, 3)
        self.block2 = Conv1dBlock(out_ch, out_ch, 3)
        self.film = FiLM(embed_dim, out_ch)

        self.time_mlp = nn.Sequential(
        nn.Mish(),
        nn.Linear(embed_dim, out_ch),
        Rearrange('batch t -> batch t 1'),
        )

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, emb, t):
        h = self.block1(x) + self.time_mlp(t)
        h = self.film(h, emb)
        h = self.block2(h)
        return h + self.residual(x)

class UNet1D(nn.Module):
    def __init__(self, 
                 in_channels=5, 
                 channels=[64, 128, 256], 
                 cond_dim=128):
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_embedder = ConditionEmbedder(cond_dim=cond_dim, in_channels=in_channels)
        self.cond_fusion = nn.Linear(cond_dim * 2, cond_dim)
        all_dims = [in_channels] + list(channels)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        # ENCODER
        self.down_blocks = nn.ModuleList()

        for i, (dim_in, dim_out) in enumerate(in_out):
            is_last = i >= (len(channels) - 1)
            self.down_blocks.append(nn.ModuleList([
                FiLMResBlock(dim_in, dim_out, self.cond_dim),
                FiLMResBlock(dim_out, dim_out, self.cond_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # BOTTLENECK
        mid_dim = channels[-1]
        self.mid1 = FiLMResBlock(mid_dim, mid_dim, cond_dim)
        self.mid2 = FiLMResBlock(mid_dim, mid_dim, cond_dim)

        # DECODER
        self.up_blocks = nn.ModuleList()
        for i, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = i >= (len(in_out) - 1)

            self.up_blocks.append(nn.ModuleList([
                FiLMResBlock(dim_out * 2, dim_in, self.cond_dim),
                FiLMResBlock(dim_in, dim_in, self.cond_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))


        # FINAL
        self.final_out = nn.Sequential(
            Conv1dBlock(channels[0], channels[0], 3),
            nn.Conv1d(channels[0], in_channels, 1)
        )

    def forward(self, x, t, goal, context):
        # 1. Общие условия
        t_e, g_e, c_e = self.cond_embedder(t, goal, context)
        proj_cond = self.cond_fusion(torch.cat([g_e, c_e], dim=-1))

        # 2. ENCODER
        skips = []
        h = x
      
        for block1, block2, downsample in self.down_blocks:
            h = block1(h, proj_cond, t_e)
            h = block2(h, proj_cond, t_e)
            # print(h.size())
            skips.append(h) # Сохраняем промежуточные стейты
            h = downsample(h)
            # print(h.size())

        # 3. MID
        h = self.mid1(h, proj_cond, t_e)
        h = self.mid2(h, proj_cond, t_e)

        # 4. DECODER
        for  block1, block2, upsample in self.up_blocks:
            # print('-'*10)
            # print(h.size())
            
            # print(h.size())
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block1(h, proj_cond, t_e)
            h = block2(h, proj_cond, t_e)
            h = upsample(h)

        return self.final_out(h)