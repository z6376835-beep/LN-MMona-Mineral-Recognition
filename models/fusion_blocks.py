import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channel, ratio=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class TopKSpatialFusion(nn.Module):
    def __init__(self, in_channels, k=30):
        super().__init__()
        self.k = k
        self.conv_score = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)

    def forward(self, x):
        b, c, h, w = x.size()
        scores = self.conv_score(x)
        scores_flat = scores.view(b, c, -1)
        _, topk_indices = torch.topk(scores_flat, k=self.k, dim=-1)
        mask = torch.zeros_like(scores_flat).scatter_(-1, topk_indices, 1.0)
        mask = mask.view(b, c, h, w)
        out = x * mask
        return out

INNER_DIM = 64

class MineralFusion(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=3, groups=in_features)
        self.fusion = TopKSpatialFusion(in_channels=in_features)
        self.SE = SEBlock(channel=in_features)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        nn.init.uniform_(self.alpha, 0.0, 1.0)
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.normal_(conv.weight, mean=0.0, std=0.02)

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)

        fused_small = conv1_x + conv2_x
        out_small = self.fusion(fused_small)

        alpha = torch.sigmoid(self.alpha)
        scaled_small = alpha * out_small
        scaled_large = (1 - alpha) * conv3_x

        x = scaled_small + scaled_large + identity
        identity = x
        x = self.SE(x)
        return x + identity
