from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, binary_opening

from lunarice_pds4.constants import MOON_RADIUS_M


def calibrate_ohrc(raw_dn: np.ndarray, constants: dict[str, float], solar_elevation_deg: float) -> np.ndarray:
    """Convert OHRC DN values to an approximate reflectance image."""
    dark = constants.get("DARK_CURRENT", 0.0)
    gain = constants.get("GAIN", 1.0)
    integration = max(constants.get("INTEGRATION_TIME", 1.0), 1e-10)
    solar_irradiance = constants.get("SOLAR_IRRADIANCE", 1361.0)

    radiance = (raw_dn.astype(np.float32) - dark) * gain / integration
    cos_sun = max(float(np.cos(np.deg2rad(solar_elevation_deg))), 0.01)
    reflectance = (np.pi * radiance) / (solar_irradiance * cos_sun)
    return np.clip(reflectance, 0, 1).astype(np.float32)


def detect_psr(reflectance: np.ndarray, psr_threshold: float = 0.005) -> np.ndarray:
    """Detect low-reflectance permanently shadowed candidate pixels."""
    psr_mask = (reflectance < psr_threshold).astype(np.uint8)
    return binary_opening(psr_mask, structure=np.ones((3, 3))).astype(np.uint8)


def _make_annulus(shape: tuple[int, int], cy: int, cx: int, radius: int, width: int) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return (distance >= radius - width / 2) & (distance <= radius + width / 2)


def _make_disk(shape: tuple[int, int], cy: int, cx: int, radius: int) -> np.ndarray:
    y, x = np.ogrid[: shape[0], : shape[1]]
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def detect_craters(
    reflectance: np.ndarray,
    geometry: dict | None,
    min_radius_m: float = 50,
    max_radius_m: float = 5000,
    num_scales: int = 24,
    fallback_m_per_pixel: float = 1.0,
) -> pd.DataFrame:
    """Detect crater rims with a multi-scale Hough transform."""
    from skimage.exposure import rescale_intensity
    from skimage.filters import sobel
    from skimage.transform import hough_circle, hough_circle_peaks

    m_per_pixel = fallback_m_per_pixel
    if geometry and "raw_df" in geometry:
        try:
            lat_range = abs(geometry["raw_df"]["LATITUDE"].max() - geometry["raw_df"]["LATITUDE"].min())
            m_per_pixel = max((lat_range * np.pi / 180.0 * MOON_RADIUS_M) / reflectance.shape[0], 1e-6)
        except Exception:
            pass

    r_min = max(5, int(min_radius_m / m_per_pixel))
    r_max = min(reflectance.shape[0] // 4, max(r_min + 1, int(max_radius_m / m_per_pixel)))
    radii = np.unique(np.linspace(r_min, r_max, num_scales, dtype=int))
    if len(radii) == 0:
        return pd.DataFrame()

    normalized = rescale_intensity(reflectance, out_range=(0, 1))
    edges = sobel(normalized)
    hough_res = hough_circle(edges, radii)
    threshold = 0.35 * float(hough_res.max()) if hough_res.size else 0.0
    accums, cx, cy, detected_radii = hough_circle_peaks(
        hough_res,
        radii,
        min_xdistance=max(2, r_min * 2),
        min_ydistance=max(2, r_min * 2),
        threshold=threshold,
    )

    records = []
    for acc, x, y, r in zip(accums, cx, cy, detected_radii):
        rim = _make_annulus(reflectance.shape, int(y), int(x), int(r), width=max(2, int(r) // 5))
        interior = _make_disk(reflectance.shape, int(y), int(x), int(r * 0.7))
        if rim.sum() == 0 or interior.sum() == 0:
            continue
        if float(reflectance[rim].mean()) <= float(reflectance[interior].mean()) * 1.1:
            continue

        lat = lon = np.nan
        if geometry:
            try:
                lat = float(geometry["lat_interp"](y, x))
                lon = float(geometry["lon_interp"](y, x))
            except Exception:
                pass
        records.append(
            {
                "pixel_row": int(y),
                "pixel_col": int(x),
                "radius_px": int(r),
                "radius_m": float(r * m_per_pixel),
                "center_lat": lat,
                "center_lon": lon,
                "confidence": float(acc),
                "interior_brightness": float(reflectance[interior].mean()),
            }
        )

    crater_df = pd.DataFrame(records)
    if len(crater_df):
        crater_df["depth_est_m"] = crater_df["radius_m"] * 0.2
        crater_df["area_m2"] = np.pi * crater_df["radius_m"] ** 2
    return crater_df


def tag_doubly_shadowed_craters(crater_df: pd.DataFrame, psr_mask: np.ndarray) -> pd.DataFrame:
    """Mark craters whose centers are in PSR and nested in a larger PSR crater."""
    if crater_df.empty:
        return crater_df
    crater_df = crater_df.copy()
    crater_df["is_psr"] = False
    crater_df["is_doubly_shadowed"] = False

    for idx, row in crater_df.iterrows():
        y, x = int(row["pixel_row"]), int(row["pixel_col"])
        if 0 <= y < psr_mask.shape[0] and 0 <= x < psr_mask.shape[1]:
            crater_df.loc[idx, "is_psr"] = bool(psr_mask[y, x])

    for idx, small in crater_df.iterrows():
        if not bool(small["is_psr"]):
            continue
        larger = crater_df[crater_df["radius_px"] > small["radius_px"] * 2]
        for _, big in larger.iterrows():
            distance = np.hypot(small["pixel_col"] - big["pixel_col"], small["pixel_row"] - big["pixel_row"])
            if distance < big["radius_px"] and bool(big["is_psr"]):
                crater_df.loc[idx, "is_doubly_shadowed"] = True
                break
    return crater_df


def estimate_slope_from_shading(reflectance: np.ndarray, solar_azimuth_deg: float) -> np.ndarray:
    """Single-image slope proxy used as a relative hazard layer."""
    from scipy.ndimage import sobel

    dy = sobel(reflectance, axis=0)
    dx = sobel(reflectance, axis=1)
    az = np.deg2rad(solar_azimuth_deg)
    slope = np.abs(dx * np.cos(az) + dy * np.sin(az))
    return (slope / (slope.max() + 1e-10)).astype(np.float32)


def detect_boulders(
    reflectance: np.ndarray,
    solar_azimuth_deg: float,
    solar_elevation_deg: float,
    min_height_m: float = 0.5,
    m_per_pixel: float = 0.3,
) -> pd.DataFrame:
    """Detect bright spots with shadow tails as boulder hazards."""
    from skimage.feature import peak_local_max

    candidates = peak_local_max(
        reflectance,
        min_distance=max(3, int(0.5 / max(m_per_pixel, 1e-6))),
        threshold_abs=np.percentile(reflectance, 90),
    )
    shadow_az = np.deg2rad(solar_azimuth_deg + 180.0)
    dx = np.cos(shadow_az)
    dy = -np.sin(shadow_az)

    records = []
    for row, col in candidates:
        tail = []
        for step in range(1, 10):
            rr = int(row + dy * step)
            cc = int(col + dx * step)
            if 0 <= rr < reflectance.shape[0] and 0 <= cc < reflectance.shape[1]:
                tail.append(float(reflectance[rr, cc]))
        if not tail:
            continue
        shadow_len_px = int(np.argmin(tail) + 1)
        height_m = shadow_len_px * m_per_pixel * np.tan(np.deg2rad(solar_elevation_deg))
        if height_m >= min_height_m:
            records.append(
                {
                    "pixel_row": int(row),
                    "pixel_col": int(col),
                    "height_m": float(height_m),
                    "shadow_len_px": shadow_len_px,
                    "is_landing_hazard": bool(height_m > 0.5),
                }
            )
    return pd.DataFrame(records)


def generate_hazard_map(
    slope_proxy: np.ndarray,
    crater_df: pd.DataFrame,
    boulder_df: pd.DataFrame,
    shape: tuple[int, int],
    slope_thresh: float = 0.6,
) -> np.ndarray:
    """Combine slope, crater rim, and boulder hazards."""
    hazard = np.zeros(shape, dtype=np.uint8)
    hazard[slope_proxy > slope_thresh] = 1

    for _, crater in crater_df.iterrows():
        rim = _make_annulus(
            shape,
            int(crater["pixel_row"]),
            int(crater["pixel_col"]),
            int(crater["radius_px"]),
            width=max(3, int(crater["radius_px"]) // 10),
        )
        hazard[rim] = 1

    boulder_mask = np.zeros(shape, dtype=bool)
    for _, boulder in boulder_df.iterrows():
        row, col = int(boulder["pixel_row"]), int(boulder["pixel_col"])
        if 0 <= row < shape[0] and 0 <= col < shape[1]:
            boulder_mask[row, col] = True
    hazard[binary_dilation(boulder_mask, structure=np.ones((17, 17)))] = 1
    return hazard.astype(np.uint8)

