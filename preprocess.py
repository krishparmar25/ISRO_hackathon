from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter

from lunarice_net.config import ensure_parent_dir
from lunarice_net.utils import normalize_channel


POL_CHANNELS = ("hh", "hv", "vh", "vv")


def load_complex_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load one complex or real raster band."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = src.meta.copy()
    return arr.astype(np.complex64), meta


def load_dfsar_channels(paths: dict[str, str]) -> tuple[dict[str, np.ndarray], dict]:
    """Load HH, HV, VH, VV polarimetric channels."""
    channels: dict[str, np.ndarray] = {}
    reference_meta = None

    for pol in POL_CHANNELS:
        # TODO: In config.yaml, set this path to the matching PRADAN DFSAR file.
        arr, meta = load_complex_raster(paths[pol])
        channels[pol.upper()] = arr
        if reference_meta is None:
            reference_meta = meta

    if reference_meta is None:
        raise ValueError("No DFSAR channels were loaded.")
    return channels, reference_meta


def refined_lee_intensity(image: np.ndarray, window_size: int = 7) -> np.ndarray:
    """Lee-style filter for SAR intensity while preserving edges."""
    intensity = np.abs(image) ** 2
    local_mean = uniform_filter(intensity, window_size)
    local_mean_sq = uniform_filter(intensity**2, window_size)
    local_var = np.maximum(local_mean_sq - local_mean**2, 0.0)

    noise_var = np.percentile(local_var[np.isfinite(local_var)], 10)
    weight = local_var / (local_var + noise_var + 1e-9)
    return local_mean + weight * (intensity - local_mean)


def compute_stokes_parameters(channels: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """
    Compute approximate circular-basis Stokes parameters.

    Fixes the proposal bug where VH was used but never assigned.
    Use original complex SLC channels here, not filtered real intensity.
    """
    hh = channels["HH"]
    hv = channels["HV"]
    vh = channels["VH"]
    vv = channels["VV"]

    rr = 0.5 * (hh - 1j * hv - 1j * vh - vv)
    rl = 0.5 * (hh + 1j * hv - 1j * vh + vv)

    s0 = np.abs(rr) ** 2 + np.abs(rl) ** 2
    s1 = np.abs(rr) ** 2 - np.abs(rl) ** 2
    s2 = 2.0 * np.real(rr * np.conj(rl))
    s3 = -2.0 * np.imag(rr * np.conj(rl))
    return s0.astype(np.float32), s1.astype(np.float32), s2.astype(np.float32), s3.astype(np.float32)


def compute_cpr_dop(s0: np.ndarray, s1: np.ndarray, s2: np.ndarray, s3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute CPR and DOP from Stokes parameters."""
    eps = 1e-9
    opposite_circular = (s0 + s3) / 2.0
    same_circular = (s0 - s3) / 2.0

    cpr = same_circular / (opposite_circular + eps)
    dop = np.sqrt(s1**2 + s2**2 + s3**2) / (s0 + eps)

    cpr = np.clip(cpr, 0.0, 10.0)
    dop = np.clip(dop, 0.0, 1.0)
    return cpr.astype(np.float32), dop.astype(np.float32)


def read_real_raster(path: str | Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def coregister_to_reference(src_path: str | Path, reference_path: str | Path, output_path: str | Path) -> None:
    """Reproject a raster so it matches a reference raster grid."""
    ensure_parent_dir(output_path)
    with rasterio.open(reference_path) as ref, rasterio.open(src_path) as src:
        dst_meta = ref.meta.copy()
        dst_meta.update(dtype=src.dtypes[0], count=1)

        with rasterio.open(output_path, "w", **dst_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=Resampling.bilinear,
            )


def apply_ice_gate(
    cpr: np.ndarray,
    dop: np.ndarray,
    slope_deg: np.ndarray,
    temp_k: np.ndarray,
    cpr_threshold: float,
    dop_threshold: float,
    slope_threshold_deg: float,
    temperature_threshold_k: float,
) -> np.ndarray:
    """Physics-constrained candidate mask."""
    return (
        (cpr > cpr_threshold)
        & (dop < dop_threshold)
        & (slope_deg < slope_threshold_deg)
        & (temp_k < temperature_threshold_k)
    ).astype(np.float32)


def build_feature_stack(config: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Create model-ready channels from S-band, L-band, slope, temperature, and PSR mask."""
    raw = config["data"]["raw"]
    gate_cfg = config["physics_gate"]

    s_channels, meta = load_dfsar_channels(raw["s_band"])
    l_channels, _ = load_dfsar_channels(raw["l_band"])

    s0, s1, s2, s3 = compute_stokes_parameters(s_channels)
    l0, l1, l2, l3 = compute_stokes_parameters(l_channels)

    s_cpr, s_dop = compute_cpr_dop(s0, s1, s2, s3)
    l_cpr, l_dop = compute_cpr_dop(l0, l1, l2, l3)
    dual_freq_ratio = s_cpr / (l_cpr + 1e-6)

    # TODO: These rasters must be aligned to DFSAR grid before training.
    slope = read_real_raster(raw["lola_slope"])
    temp = read_real_raster(raw["diviner_temperature"])
    psr = read_real_raster(raw["psr_mask"]) if raw.get("psr_mask") else np.ones_like(s_cpr)

    ice_gate = apply_ice_gate(
        s_cpr,
        s_dop,
        slope,
        temp,
        gate_cfg["cpr_threshold"],
        gate_cfg["dop_threshold"],
        gate_cfg["slope_threshold_deg"],
        gate_cfg["temperature_threshold_k"],
    )

    radar = [
        normalize_channel(s_cpr),
        normalize_channel(s_dop),
        normalize_channel(l_cpr),
        normalize_channel(l_dop),
        normalize_channel(dual_freq_ratio),
        normalize_channel(s0),
        normalize_channel(l0),
    ]
    aux = [
        normalize_channel(slope),
        normalize_channel(temp),
        normalize_channel(psr),
        normalize_channel(ice_gate),
    ]

    feature_stack = np.stack(radar + aux, axis=0).astype(np.float32)
    return feature_stack, ice_gate.astype(np.float32), meta

