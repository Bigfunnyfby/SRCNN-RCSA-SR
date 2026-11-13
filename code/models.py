# -*- coding: utf-8 -*-
"""Models: SRCNN baseline and RCSA-SR (Residual Channel Attention SR)."""
import torch
import torch.nn as nn


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(m.bias)


class SRCNN(nn.Module):
    """The 9-5-5 SRCNN baseline (Dong et al., TPAMI 2016)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 5, padding=2),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.net(x)


class SEBlock(nn.Module):
    def __init__(self, channels=64, reduction=8):
        super().__init__()
        hidden = max(4, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        w = torch.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        self.last_attention = w.detach()
        return x * w


class ResidualBlock(nn.Module):
    def __init__(self, channels=64, use_se=True, reduction=8, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.se = SEBlock(channels, reduction) if use_se else nn.Identity()
        self.res_scale = res_scale

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = self.se(out)
        return x + self.res_scale * out


class RCSASR(nn.Module):
    """Residual channel-attention SR network.

    Input is a bicubic-upsampled LR image; the network predicts the residual.
    """

    def __init__(self, channels=64, num_blocks=4, use_se=True, reduction=8,
                 res_scale=0.1):
        super().__init__()
        self.head = nn.Conv2d(1, channels, 5, padding=2)
        self.body = nn.Sequential(
            *[ResidualBlock(channels, use_se=use_se, reduction=reduction,
                            res_scale=res_scale)
              for _ in range(num_blocks)]
        )
        self.tail = nn.Conv2d(channels, 1, 5, padding=2)
        self.apply(init_weights)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        residual = self.tail(feat)
        return x + residual


def build_model(name, **kwargs):
    if name == "srcnn":
        return SRCNN()
    if name == "rcsa":
        return RCSASR(**kwargs)
    raise ValueError(name)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
