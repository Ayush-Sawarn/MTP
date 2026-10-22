"""AttRNet-Conv and AttRNet-FC from Gao et al. (PyTorch implementation)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    """Squeeze-excitation attention, using either FC or 1x1 Conv excitation."""

    def __init__(self, channels: int, kind: str, reduction: int = 2) -> None:
        super().__init__()
        if kind not in {"fc", "conv"}:
            raise ValueError("kind must be 'fc' or 'conv'")
        hidden = max(1, channels // reduction)
        self.kind = kind
        if kind == "fc":
            self.excitation = nn.Sequential(
                nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels)
            )
        else:
            self.excitation = nn.Sequential(
                nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True), nn.Conv2d(hidden, channels, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(x, 1)
        if self.kind == "fc":
            pooled = pooled.flatten(1)
            weights = torch.sigmoid(self.excitation(pooled)).unsqueeze(-1).unsqueeze(-1)
        else:
            weights = torch.sigmoid(self.excitation(pooled))
        return x * weights


class AttResBlock(nn.Module):
    """Three convolutions, channel attention on the third, then an identity skip.

    The drawing connects the first convolution's result to the residual add;
    that is implemented here (rather than skipping the whole block input).
    """

    def __init__(self, channels: int, attention: str) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.attention = ChannelAttention(channels, attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.conv1(x)
        y = self.conv2(shortcut)
        y = F.relu(y, inplace=True)
        y = self.conv3(y)
        return shortcut + self.attention(y)


class AttRNet(nn.Module):
    """Residual attention channel estimator.

    Inputs are real/imaginary LS pilot estimates shaped ``[B, 2, P_f, P_t]``;
    outputs are real/imaginary channel gains shaped ``[B, 2, 72, 14]``.
    """

    def __init__(
        self,
        attention: str = "fc",
        features: int = 8,
        blocks: int = 3,
        output_size: tuple[int, int] = (72, 14),
        output_kernel: tuple[int, int] = (36, 7),
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.stem = nn.Conv2d(2, features, 3, padding=1)
        self.blocks = nn.Sequential(*[AttResBlock(features, attention) for _ in range(blocks)])
        # The paper applies a 3x3 feature convolution after the AttResBlocks,
        # then upsamples and applies its 36x7 output convolution.
        self.refine = nn.Conv2d(features, features, 3, padding=1)
        # Explicit asymmetric padding retains spatial size for the paper's
        # even-height 36x7 kernel without PyTorch's slower padding='same' path.
        kh, kw = output_kernel
        left, right = (kw - 1) // 2, kw - 1 - (kw - 1) // 2
        top, bottom = (kh - 1) // 2, kh - 1 - (kh - 1) // 2
        self.head_padding = nn.ZeroPad2d((left, right, top, bottom))
        self.head = nn.Conv2d(features, 2, output_kernel)
        self.attention = attention

    def forward(self, ls_pilots: torch.Tensor) -> torch.Tensor:
        x = self.stem(ls_pilots)
        x = self.blocks(x)
        x = self.refine(x)
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        return self.head(self.head_padding(x))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
