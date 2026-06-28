from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robust per-channel normalization using finite pixels only."""
    out = x.astype(np.float32, copy=True)
    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros_like(out, dtype=np.float32)

    values = out[finite]
    lo, hi = np.percentile(values, [2, 98])
    out = np.clip(out, lo, hi)
    out = (out - lo) / (hi - lo + eps)
    out[~finite] = 0.0
    return out.astype(np.float32)

