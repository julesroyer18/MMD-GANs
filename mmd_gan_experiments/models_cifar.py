from __future__ import annotations

import torch
from torch import nn


def weights_init(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias.data)


class DCGANGenerator(nn.Module):
    def __init__(self, z_dim: int = 128, base_channels: int = 64) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 8, base_channels * 4, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                base_channels * 4, base_channels * 2, 4, 2, 1, bias=False
            ),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

        self.apply(weights_init)

    # TRYING TO REPLACE BATCHNORM BY LAYERNORM

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)
        return self.net(z)


class _CriticBackbone(nn.Module):
    def __init__(self, base_channels: int = 64, final_channels: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, final_channels, 4, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x)
        return feat.view(x.shape[0], -1)


class DCGANFeatureCritic(nn.Module):
    def __init__(self, base_channels: int = 64, feature_dim: int = 128) -> None:
        super().__init__()
        self.backbone = _CriticBackbone(base_channels=base_channels)
        self.head = nn.Linear(512, feature_dim)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class DCGANScalarCritic(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        self.backbone = _CriticBackbone(base_channels=base_channels)
        self.head = nn.Linear(512, 1)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x)).view(-1)
