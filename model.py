from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, in_ch: int, widths: tuple[int, ...]) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        prev = in_ch
        for width in widths:
            self.blocks.append(ConvBlock(prev, width, dropout=0.05))
            prev = width

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = []
        for i, block in enumerate(self.blocks):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = block(x)
            features.append(x)
        return features


class AttentionGate(nn.Module):
    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int) -> None:
        super().__init__()
        self.gate_proj = nn.Conv2d(gate_ch, inter_ch, 1, bias=False)
        self.skip_proj = nn.Conv2d(skip_ch, inter_ch, 1, bias=False)
        self.psi = nn.Conv2d(inter_ch, 1, 1)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        attn = torch.sigmoid(self.psi(F.relu(self.gate_proj(gate) + self.skip_proj(skip))))
        return skip * attn


class DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.attn = AttentionGate(in_ch, skip_ch, max(out_ch // 2, 8))
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout=0.1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        skip = self.attn(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class PhysicsGate(nn.Module):
    """Inject physics prior by down-weighting impossible regions."""

    def forward(self, x: torch.Tensor, gate_mask: torch.Tensor) -> torch.Tensor:
        gate = F.interpolate(gate_mask, size=x.shape[-2:], mode="nearest")
        return x * (0.25 + 0.75 * gate)


class AGMMUNet(nn.Module):
    """Attention-gated multi-modal U-Net for lunar ice probability mapping."""

    def __init__(self, radar_ch: int = 7, aux_ch: int = 4) -> None:
        super().__init__()
        widths = (32, 64, 128, 256)
        self.radar_encoder = Encoder(radar_ch, widths)
        self.aux_encoder = Encoder(aux_ch, widths)
        self.physics_gate = PhysicsGate()

        self.bottleneck = ConvBlock(widths[-1] * 2, 512, dropout=0.2)
        self.dec3 = DecoderBlock(512, widths[2] * 2, 256)
        self.dec2 = DecoderBlock(256, widths[1] * 2, 128)
        self.dec1 = DecoderBlock(128, widths[0] * 2, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, radar_x: torch.Tensor, aux_x: torch.Tensor, ice_gate_mask: torch.Tensor) -> torch.Tensor:
        radar_feats = self.radar_encoder(radar_x)
        aux_feats = self.aux_encoder(aux_x)

        fused_skips = [torch.cat([r, a], dim=1) for r, a in zip(radar_feats, aux_feats)]
        bottleneck = self.bottleneck(fused_skips[-1])
        bottleneck = self.physics_gate(bottleneck, ice_gate_mask)

        x = self.dec3(bottleneck, fused_skips[2])
        x = self.dec2(x, fused_skips[1])
        x = self.dec1(x, fused_skips[0])
        return torch.sigmoid(self.out(x))

