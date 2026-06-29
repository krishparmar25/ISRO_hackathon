from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from lxml import etree

from lunarice_pds4.constants import BAND_ORDERINGS, PDS4_DTYPE_MAP


def _local_text(node: etree._Element, name: str) -> str | None:
    found = node.xpath(f".//*[local-name()='{name}']")
    if not found or found[0].text is None:
        return None
    return found[0].text.strip()


def _all_local_text(node: etree._Element, name: str) -> list[str]:
    values = []
    for item in node.xpath(f".//*[local-name()='{name}']"):
        if item.text and item.text.strip():
            values.append(item.text.strip())
    return values


def discover_scene(scene_root: str | Path) -> dict[str, list[Path]]:
    """Find PDS4 labels, geometry CSVs, calibration text, and binaries in a PRADAN folder."""
    root = Path(scene_root)
    if not root.exists():
        raise FileNotFoundError(f"PRADAN scene folder not found: {root}")

    labels = sorted(root.glob("data/*.xml")) or sorted(root.rglob("*.xml"))
    geometry_csvs = sorted((root / "geometry").glob("*.csv")) if (root / "geometry").exists() else []
    calibration_files = sorted((root / "miscellaneous").glob("*CALIBRATION*.txt"))
    calibration_files += sorted((root / "miscellaneous").glob("*calibration*.txt"))
    binaries = sorted(root.glob("data/*.dat")) + sorted(root.glob("data/*.img"))

    return {
        "labels": labels,
        "geometry_csvs": geometry_csvs,
        "calibration_files": sorted(set(calibration_files)),
        "binaries": binaries,
    }


def parse_pds4_label(xml_path: str | Path, scene_root: str | Path | None = None) -> dict[str, Any]:
    """Parse a PDS4 XML label and return enough metadata to read the binary file."""
    xml_path = Path(xml_path)
    tree = etree.parse(str(xml_path))
    root = tree.getroot()
    base_dir = xml_path.parent
    scene_root = Path(scene_root) if scene_root else xml_path.parents[1] if xml_path.parent.name.lower() == "data" else xml_path.parent

    file_names = _all_local_text(root, "file_name")
    data_name = None
    for name in file_names:
        if Path(name).suffix.lower() in {".dat", ".img", ".bin"}:
            data_name = name
            break
    if data_name is None and file_names:
        data_name = file_names[0]
    if data_name is None:
        raise ValueError(f"No file_name found in PDS4 label: {xml_path}")

    data_file = base_dir / data_name
    if not data_file.exists():
        matches = list(scene_root.rglob(Path(data_name).name))
        if matches:
            data_file = matches[0]

    header_bytes = 0
    header_nodes = root.xpath(".//*[local-name()='Header']")
    if header_nodes:
        length = _local_text(header_nodes[0], "object_length")
        header_bytes = int(float(length)) if length else 0

    array_nodes = root.xpath(
        ".//*[starts-with(local-name(), 'Array_') and contains(local-name(), 'Image')]"
    )
    if not array_nodes:
        raise ValueError(f"No PDS4 Array_*_Image object found in: {xml_path}")
    array_el = array_nodes[0]

    axes: dict[str, int] = {}
    axis_order: list[str] = []
    for ax in array_el.xpath(".//*[local-name()='Axis_Array']"):
        axis_name = _local_text(ax, "axis_name")
        elements = _local_text(ax, "elements")
        if axis_name and elements:
            key = axis_name.strip().upper()
            axes[key] = int(float(elements))
            axis_order.append(key)

    lines = axes.get("LINE", axes.get("LINES", axes.get("ROW", 1)))
    samples = axes.get("SAMPLE", axes.get("SAMPLES", axes.get("COLUMN", 1)))
    bands = axes.get("BAND", axes.get("BANDS", axes.get("PLANE", 1)))

    elem_nodes = array_el.xpath(".//*[local-name()='Element_Array']")
    elem = elem_nodes[0] if elem_nodes else array_el
    raw_dtype = _local_text(elem, "data_type") or "IEEE754LSBSingle"
    dtype = PDS4_DTYPE_MAP.get(raw_dtype, "<f4")
    scaling_factor = float(_local_text(elem, "scaling_factor") or 1.0)
    value_offset = float(_local_text(elem, "value_offset") or 0.0)

    name_upper = " ".join([xml_path.name, data_file.name, str(scene_root)]).upper()
    product_type = "DFSAR_SLC" if "DFSAR" in name_upper or data_file.suffix.lower() == ".dat" else "OHRC_IMAGE"

    meta: dict[str, Any] = {
        "xml_path": str(xml_path),
        "scene_root": str(scene_root),
        "data_file": str(data_file),
        "product_type": product_type,
        "lines": int(lines),
        "samples": int(samples),
        "bands": int(bands),
        "axis_order": axis_order,
        "raw_data_type": raw_dtype,
        "data_type": dtype,
        "header_bytes": int(header_bytes),
        "scaling_factor": scaling_factor,
        "value_offset": value_offset,
        "center_lat": None,
        "center_lon": None,
        "incidence_angle": None,
        "wavelength_cm": None,
        "polarization": None,
        "solar_azimuth": None,
        "solar_elevation": None,
    }

    lat_min = _local_text(root, "minimum_bounding_latitude")
    lat_max = _local_text(root, "maximum_bounding_latitude")
    lon_min = _local_text(root, "western_bounding_longitude")
    lon_max = _local_text(root, "eastern_bounding_longitude")
    if lat_min and lat_max and lon_min and lon_max:
        meta["center_lat"] = (float(lat_min) + float(lat_max)) / 2.0
        meta["center_lon"] = (float(lon_min) + float(lon_max)) / 2.0

    for field, keys in {
        "incidence_angle": ["incidence_angle", "local_incidence_angle"],
        "wavelength_cm": ["wavelength", "radar_wavelength"],
        "polarization": ["polarization_type", "polarization"],
        "solar_azimuth": ["Solar_Azimuth_Angle", "solar_azimuth"],
        "solar_elevation": ["Solar_Elevation_Angle", "solar_elevation"],
    }.items():
        for key in keys:
            value = _local_text(root, key)
            if value:
                try:
                    meta[field] = float(value)
                    if field == "wavelength_cm" and meta[field] < 1:
                        meta[field] = meta[field] * 100.0
                except ValueError:
                    meta[field] = value
                break

    return meta


def load_raw_array(meta: dict[str, Any]) -> np.ndarray:
    """Memory-map and scale a PDS4 binary product using parsed label metadata."""
    dtype = np.dtype(meta["data_type"])
    total = int(meta["lines"]) * int(meta["samples"]) * int(meta["bands"])
    raw = np.memmap(meta["data_file"], dtype=dtype, mode="r", offset=int(meta["header_bytes"]))

    if raw.size < total:
        raise ValueError(
            f"Binary file is smaller than label says: found {raw.size} elements, expected {total}"
        )

    if int(meta["bands"]) == 1:
        arr = np.asarray(raw[: int(meta["lines"]) * int(meta["samples"])]).reshape(
            int(meta["lines"]), int(meta["samples"])
        )
    else:
        arr = np.asarray(raw[:total]).reshape(
            int(meta["lines"]), int(meta["samples"]), int(meta["bands"])
        )

    if meta.get("scaling_factor", 1.0) != 1.0 or meta.get("value_offset", 0.0) != 0.0:
        arr = arr.astype(np.float32) * float(meta["scaling_factor"]) + float(meta["value_offset"])
    return arr


def reconstruct_complex_channels(arr: np.ndarray, band_order: str = "re_h_im_h_re_v_im_v") -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct DFSAR horizontal and vertical complex receive channels."""
    if np.iscomplexobj(arr):
        if arr.ndim == 2:
            return arr.astype(np.complex64), arr.astype(np.complex64)
        if arr.ndim == 3 and arr.shape[2] >= 2:
            return arr[..., 0].astype(np.complex64), arr[..., 1].astype(np.complex64)

    if arr.ndim == 3 and arr.shape[2] >= 4:
        if band_order not in BAND_ORDERINGS:
            raise ValueError(f"Unknown band_order: {band_order}")
        h_pair, v_pair = BAND_ORDERINGS[band_order]
        e_h = arr[..., h_pair[0]] + 1j * arr[..., h_pair[1]]
        e_v = arr[..., v_pair[0]] + 1j * arr[..., v_pair[1]]
        return e_h.astype(np.complex64), e_v.astype(np.complex64)

    if arr.ndim == 3 and arr.shape[2] == 2:
        warnings.warn("Only two real DFSAR bands found. Phase is unavailable; CPR/DOP will be approximate.")
        return arr[..., 0].astype(np.complex64), arr[..., 1].astype(np.complex64)

    raise ValueError(f"Unexpected DFSAR array shape: {arr.shape}")


def read_calibration_constants(calib_file: str | Path | None) -> dict[str, float]:
    """Read loose KEY=VALUE or KEY: VALUE calibration constants."""
    if not calib_file:
        return {}
    constants: dict[str, float] = {}
    pattern = re.compile(r"^\s*([A-Za-z0-9_ \-/]+)\s*[:=]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    with open(calib_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                key = re.sub(r"\s+", "_", match.group(1).strip()).upper()
                constants[key] = float(match.group(2))
    return constants


def validate_ingestion(meta: dict[str, Any], e_h: np.ndarray, e_v: np.ndarray) -> None:
    expected = (int(meta["lines"]), int(meta["samples"]))
    assert e_h.shape == expected, f"E_H shape mismatch: {e_h.shape} != {expected}"
    assert e_v.shape == expected, f"E_V shape mismatch: {e_v.shape} != {expected}"
    assert np.iscomplexobj(e_h), "E_H is not complex; check band reconstruction."
    total_power = float(np.nanmean(np.abs(e_h) ** 2 + np.abs(e_v) ** 2))
    assert total_power > 0, "Total power is zero; binary read likely failed."
    print(f"[OK] Ingestion: {meta['product_type']} shape={expected}, mean_power={total_power:.4e}")

