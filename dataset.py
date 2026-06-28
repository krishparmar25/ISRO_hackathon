from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class LunarPatchDataset(Dataset):
    """Random patch dataset with positive-patch oversampling."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        patch_size: int,
        length: int,
        positive_fraction: float = 0.6,
    ) -> None:
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.patch_size = patch_size
        self.length = length
        self.positive_fraction = positive_fraction

        positive_pixels = np.argwhere(self.labels > 0.5)
        self.positive_pixels = positive_pixels if len(positive_pixels) else None

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _, h, w = self.features.shape
        ps = self.patch_size

        use_positive = self.positive_pixels is not None and np.random.rand() < self.positive_fraction
        if use_positive:
            cy, cx = self.positive_pixels[np.random.randint(len(self.positive_pixels))]
            y0 = int(np.clip(cy - ps // 2, 0, h - ps))
            x0 = int(np.clip(cx - ps // 2, 0, w - ps))
        else:
            y0 = np.random.randint(0, max(1, h - ps + 1))
            x0 = np.random.randint(0, max(1, w - ps + 1))

        x = self.features[:, y0 : y0 + ps, x0 : x0 + ps]
        y = self.labels[y0 : y0 + ps, x0 : x0 + ps][None, :, :]

        radar = torch.from_numpy(x[:7])
        aux = torch.from_numpy(x[7:11])
        gate = torch.from_numpy(x[10:11])
        target = torch.from_numpy(y)
        return radar, aux, gate, target

