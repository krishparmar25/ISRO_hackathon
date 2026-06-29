from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline, griddata

from lunarice_pds4.constants import MOON_RADIUS_M


def _standardize_geometry_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    variants = {
        "LINE": ["LINE", "LINES", "ROW", "PIXEL_ROW", "LINE_NUMBER"],
        "SAMPLE": ["SAMPLE", "SAMPLES", "COLUMN", "COL", "PIXEL_COLUMN", "SAMPLE_NUMBER"],
        "LATITUDE": ["LATITUDE", "LAT", "GEODETIC_LATITUDE"],
        "LONGITUDE": ["LONGITUDE", "LON", "EAST_LONGITUDE"],
        "INCIDENCE_ANGLE": ["INCIDENCE_ANGLE", "INC_ANGLE", "LOCAL_INCIDENCE_ANGLE"],
    }
    for canonical, names in variants.items():
        for name in names:
            if name in df.columns:
                df = df.rename(columns={name: canonical})
                break
    return df


def load_geometry(geometry_dir: str | Path, lines: int, samples: int) -> dict:
    """Load PRADAN geometry CSV files and build lat/lon interpolators."""
    geometry_dir = Path(geometry_dir)
    csv_files = sorted(geometry_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No geometry CSV files found in: {geometry_dir}")

    frames = []
    for path in csv_files:
        try:
            frames.append(pd.read_csv(path, comment="#"))
        except UnicodeDecodeError:
            frames.append(pd.read_csv(path, comment="#", encoding="latin1"))
    raw_df = pd.concat(frames, ignore_index=True)
    raw_df = _standardize_geometry_columns(raw_df)
    raw_df = raw_df.dropna(subset=["LINE", "SAMPLE", "LATITUDE", "LONGITUDE"])
    raw_df = raw_df.sort_values(["LINE", "SAMPLE"])

    line_vals = np.unique(raw_df["LINE"].astype(float).values)
    sample_vals = np.unique(raw_df["SAMPLE"].astype(float).values)
    points = raw_df[["LINE", "SAMPLE"]].astype(float).values

    def make_interp(column: str) -> Callable:
        values = raw_df[column].astype(float).values
        try:
            grid = raw_df.pivot_table(values=column, index="LINE", columns="SAMPLE").values
            spline = RectBivariateSpline(line_vals, sample_vals, grid, kx=min(3, len(line_vals) - 1), ky=min(3, len(sample_vals) - 1))
            return lambda rows, cols: spline.ev(np.asarray(rows), np.asarray(cols))
        except Exception:
            return lambda rows, cols: griddata(points, values, (np.asarray(rows), np.asarray(cols)), method="linear")

    geometry = {
        "raw_df": raw_df,
        "lat_interp": make_interp("LATITUDE"),
        "lon_interp": make_interp("LONGITUDE"),
        "inc_interp": make_interp("INCIDENCE_ANGLE") if "INCIDENCE_ANGLE" in raw_df.columns else None,
        "lines": lines,
        "samples": samples,
    }
    return geometry


def build_geotransform(geometry: dict, lines: int, samples: int) -> tuple[float, float, float, float, float, float]:
    """Approximate affine geotransform from geometry bounds."""
    df = geometry["raw_df"]
    lon_min = float(df["LONGITUDE"].min())
    lon_max = float(df["LONGITUDE"].max())
    lat_min = float(df["LATITUDE"].min())
    lat_max = float(df["LATITUDE"].max())
    pixel_width = (lon_max - lon_min) / max(samples - 1, 1)
    pixel_height = (lat_min - lat_max) / max(lines - 1, 1)
    return (lon_min, pixel_width, 0.0, lat_max, 0.0, pixel_height)


def estimate_m_per_pixel(geometry: dict, fallback: float = 1.0) -> float:
    """Estimate meters per pixel from geometry latitude span."""
    try:
        df = geometry["raw_df"]
        lat_span_deg = abs(float(df["LATITUDE"].max()) - float(df["LATITUDE"].min()))
        line_span = max(float(df["LINE"].max()) - float(df["LINE"].min()), 1.0)
        meters = lat_span_deg * np.pi / 180.0 * MOON_RADIUS_M
        return float(max(meters / line_span, 1e-6))
    except Exception:
        return float(fallback)


def incidence_grid(geometry: dict, shape: tuple[int, int]) -> np.ndarray | None:
    """Dense incidence-angle grid, if geometry CSV contains incidence angle."""
    if geometry.get("inc_interp") is None:
        return None
    rows, cols = np.indices(shape)
    return geometry["inc_interp"](rows, cols).astype(np.float32)


def resize_to_shape(arr: np.ndarray, target_shape: tuple[int, int], order: int = 1) -> np.ndarray:
    """Simple array resize for quick common-grid integration."""
    from scipy.ndimage import zoom

    if arr.shape == target_shape:
        return arr
    zoom_y = target_shape[0] / arr.shape[0]
    zoom_x = target_shape[1] / arr.shape[1]
    return zoom(arr, (zoom_y, zoom_x), order=order)

