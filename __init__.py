from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_closing, label, uniform_filter


def _calibration_k(constants: dict[str, float]) -> float:
    for key in ("K", "CALIBRATION_CONSTANT", "RADIOMETRIC_CONSTANT", "SIGMA0_CONSTANT"):
        if key in constants and constants[key] != 0:
            return float(constants[key])
    return 1.0


def multilook_coherency(
    e_h: np.ndarray,
    e_v: np.ndarray,
    looks: tuple[int, int] = (5, 5),
    calibration_constants: dict[str, float] | None = None,
    incidence_angle_deg: np.ndarray | float | None = None,
) -> dict[str, np.ndarray]:
    """Compute multi-look coherency elements from complex DFSAR channels."""
    size = tuple(int(v) for v in looks)
    k = _calibration_k(calibration_constants or {})

    i_h = uniform_filter(np.abs(e_h) ** 2, size=size).astype(np.float32) / k
    i_v = uniform_filter(np.abs(e_v) ** 2, size=size).astype(np.float32) / k
    cross = e_h * np.conj(e_v)
    c_re = uniform_filter(np.real(cross), size=size).astype(np.float32) / k
    c_im = uniform_filter(np.imag(cross), size=size).astype(np.float32) / k

    if incidence_angle_deg is not None:
        correction = np.sin(np.deg2rad(incidence_angle_deg)).astype(np.float32)
        correction = np.maximum(correction, 0.05)
        i_h = i_h / correction
        i_v = i_v / correction
        c_re = c_re / correction
        c_im = c_im / correction

    return {"I_HH": i_h, "I_VV": i_v, "C_HV_real": c_re, "C_HV_imag": c_im}


def compute_stokes(coherency: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """RHC-transmit Stokes parameters from roadmap Module 2.2."""
    i_h = coherency["I_HH"]
    i_v = coherency["I_VV"]
    c_re = coherency["C_HV_real"]
    c_im = coherency["C_HV_imag"]

    s1 = i_h + i_v
    s2 = i_h - i_v
    s3 = 2.0 * c_re
    s4 = 2.0 * c_im
    return {
        "S1": s1.astype(np.float32),
        "S2": s2.astype(np.float32),
        "S3": s3.astype(np.float32),
        "S4": s4.astype(np.float32),
    }


def circular_components(s1: np.ndarray, s4: np.ndarray, invert_s4_sign: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return same-sense and opposite-sense circular backscatter."""
    if invert_s4_sign:
        sigma_sc = (s1 - s4) / 2.0
        sigma_oc = (s1 + s4) / 2.0
    else:
        sigma_sc = (s1 + s4) / 2.0
        sigma_oc = (s1 - s4) / 2.0
    return np.maximum(sigma_sc, 0).astype(np.float32), np.maximum(sigma_oc, 0).astype(np.float32)


def compute_cpr_dop(stokes: dict[str, np.ndarray], invert_s4_sign: bool = False) -> dict[str, np.ndarray]:
    """Compute CPR and DOP from Stokes parameters."""
    s1, s2, s3, s4 = stokes["S1"], stokes["S2"], stokes["S3"], stokes["S4"]
    sigma_sc, sigma_oc = circular_components(s1, s4, invert_s4_sign=invert_s4_sign)

    cpr = sigma_sc / (sigma_oc + 1e-10)
    dop = np.sqrt(np.maximum(s2**2 + s3**2 + s4**2, 0)) / (s1 + 1e-10)
    return {
        "sigma_sc": sigma_sc,
        "sigma_oc": sigma_oc,
        "CPR": np.clip(cpr, 0, 10).astype(np.float32),
        "DOP": np.clip(dop, 0, 1).astype(np.float32),
    }


def m_chi_decomposition(stokes: dict[str, np.ndarray], dop: np.ndarray) -> dict[str, np.ndarray]:
    """m-chi decomposition: even, odd/volume, and diffuse powers."""
    s1 = stokes["S1"]
    s4 = stokes["S4"]
    m = np.clip(dop, 0, 1)
    arg = np.clip(-s4 / (m * s1 + 1e-10), -1, 1)
    chi = 0.5 * np.arcsin(arg)
    sin_2chi = np.sin(2.0 * chi)

    p_even = (m * s1 / 2.0) * (1.0 - sin_2chi)
    p_odd = (m * s1 / 2.0) * (1.0 + sin_2chi)
    p_diffuse = (1.0 - m) * s1
    return {
        "chi": chi.astype(np.float32),
        "P_even": np.maximum(p_even, 0).astype(np.float32),
        "P_odd": np.maximum(p_odd, 0).astype(np.float32),
        "P_diffuse": np.maximum(p_diffuse, 0).astype(np.float32),
        "P_odd_fraction": (p_odd / (s1 + 1e-10)).clip(0, 1).astype(np.float32),
    }


def generate_ice_mask(
    cpr: np.ndarray,
    dop: np.ndarray,
    s1: np.ndarray,
    p_odd: np.ndarray,
    cpr_thresh: float = 1.0,
    dop_thresh: float = 0.5,
    volume_fraction_thresh: float = 0.4,
    noise_percentile: float = 5,
    min_ice_area_px: int = 25,
) -> np.ndarray:
    """Primary PS-8 ice candidate mask: high CPR, low DOP, strong volume scattering."""
    valid_power = s1[np.isfinite(s1) & (s1 > 0)]
    noise_floor = float(np.percentile(valid_power, noise_percentile)) if valid_power.size else 0.0
    f_vol = p_odd / (s1 + 1e-10)

    primary = (cpr > cpr_thresh) & (dop < dop_thresh) & (s1 > noise_floor)
    secondary = f_vol > volume_fraction_thresh
    mask = (primary & secondary).astype(np.uint8)
    mask = binary_closing(mask, structure=np.ones((3, 3))).astype(np.uint8)

    labeled, num_features = label(mask)
    for region_id in range(1, num_features + 1):
        if int(np.sum(labeled == region_id)) < int(min_ice_area_px):
            mask[labeled == region_id] = 0
    return mask.astype(np.uint8)

