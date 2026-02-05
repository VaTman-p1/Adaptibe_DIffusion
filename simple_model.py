import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.downsample = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_channels)

    def forward(self, x, emb):
        x = self.conv1(x)
        emb_out = self.emb_proj(emb)[:, :, None]
        x = x + emb_out
        x = self.conv2(x)
        return x, self.downsample(x)  # skip before down, downsampled

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim):
        super().__init__()
        self.conv1 = ConvBlock(in_channels *2, out_channels)  # after concat
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.emb_proj = nn.Linear(emb_dim, out_channels)

    def forward(self, x, skip, emb):
        # Interpolate to exactly match skip length (handles arbitrary sizes)
        x = F.interpolate(x, size=(skip.size(2),), mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        emb_out = self.emb_proj(emb)[:, :, None]
        x = x + emb_out
        x = self.conv2(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, channels, emb_dim):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = ConvBlock(channels, channels)
        self.emb_proj = nn.Linear(emb_dim, channels)

    def forward(self, x, emb):
        x = self.conv1(x)
        emb_out = self.emb_proj(emb)[:, :, None]
        x = x + emb_out
        x = self.conv2(x)
        return x

class GoalTimeConditionedUNet1D(nn.Module):
    def __init__(self, state_channels=5, goal_channels=3, base_channels=64, emb_dim=64, num_levels=3):
        super().__init__()
        self.state_channels = state_channels
        self.goal_channels = goal_channels
        in_channels = state_channels + goal_channels

        self.time_emb = SinusoidalEmbedding(emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

        # Initial conv
        self.init_conv = ConvBlock(in_channels, base_channels)

        # Down blocks
        self.down_blocks = nn.ModuleList()
        channels = base_channels
        for _ in range(num_levels):
            self.down_blocks.append(DownBlock(channels, channels * 2, emb_dim))
            channels *= 2

        # Bottleneck
        self.bottleneck = Bottleneck(channels, emb_dim)

        # Up blocks
        self.up_blocks = nn.ModuleList()
        for _ in range(num_levels):
            self.up_blocks.append(UpBlock(channels, channels // 2, emb_dim))
            channels //= 2

        # Final conv (outputs state_channels)
        self.final_conv = nn.Conv1d(base_channels, state_channels, kernel_size=1)

    def forward(self, noisy_seq, goal, t):
        
        L = noisy_seq.size(2)
        # noisy_seq: (B, state_channels, L), goal: (B, 1, goal_channels) or (B, goal_channels, 1)
  
        x = torch.cat([noisy_seq, goal.unsqueeze(-1).repeat(1, 1, L)], dim=1)  # (B, state_channels + goal_channels, L)

        emb = self.time_emb(t)
        emb = self.time_mlp(emb)

        x = self.init_conv(x)

        skips = []
        for down in self.down_blocks:
            skip, x = down(x, emb)
            skips.append(skip)

        x = self.bottleneck(x, emb)

        for up in self.up_blocks:
            skip = skips.pop()
            x = up(x, skip, emb)

        x = self.final_conv(x)
        return x  # (B, state_channels, L)

# model = GoalTimeConditionedUNet1D(
#     state_channels=5,
#     goal_channels=3,
#     base_channels=64,
#     emb_dim=64,
#     num_levels=3
# )

# B, L = 4, 256
# noisy = torch.randn(B, 5, L)
# goal = torch.randn(B, 3)
# t = torch.randint(0, 1000, (B,))

# out = model(noisy, goal, t)
# print(out.shape)          # should print: torch.Size([4, 5, 256])