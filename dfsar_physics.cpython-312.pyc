from __future__ import annotations

from pathlib import Path

import numpy as np

from lunarice_pds4.config import load_config, resolve_path
from lunarice_pds4.dfsar_physics import (
    compute_cpr_dop,
    compute_stokes,
    generate_ice_mask,
    m_chi_decomposition,
    multilook_coherency,
)
from lunarice_pds4.geometry import build_geotransform, estimate_m_per_pixel, incidence_grid, load_geometry, resize_to_shape
from lunarice_pds4.ingestion import (
    discover_scene,
    load_raw_array,
    parse_pds4_label,
    read_calibration_constants,
    reconstruct_complex_channels,
    validate_ingestion,
)
from lunarice_pds4.ohrc_terrain import (
    calibrate_ohrc,
    detect_boulders,
    detect_craters,
    detect_psr,
    estimate_slope_from_shading,
    generate_hazard_map,
    tag_doubly_shadowed_craters,
)
from lunarice_pds4.products import write_analysis_stack, write_dashboard, write_rover_geojson, write_volume_csv


def _first_or_none(values):
    return values[0] if values else None


def validate_config(config_path: str | Path) -> None:
    """Check that configured PRADAN folders exist and contain expected files."""
    config_path = Path(config_path)
    cfg = load_config(config_path)
    base_dir = config_path.parent

    for key in ("dfsar_root", "ohrc_root"):
        scene_root = resolve_path(base_dir, cfg["data"]["pradan"].get(key))
        print(f"\n[{key}] {scene_root}")
        if not scene_root or not scene_root.exists():
            print("  MISSING - paste the PRADAN folder and update config.yaml")
            continue
        discovered = discover_scene(scene_root)
        print(f"  labels:       {len(discovered['labels'])}")
        print(f"  binaries:     {len(discovered['binaries'])}")
        print(f"  geometry csv: {len(discovered['geometry_csvs'])}")
        print(f"  calibration:  {len(discovered['calibration_files'])}")
        if discovered["labels"]:
            meta = parse_pds4_label(discovered["labels"][0], scene_root)
            print(f"  first label:  {Path(meta['xml_path']).name}")
            print(f"  product:      {meta['product_type']}")
            print(f"  shape:        lines={meta['lines']} samples={meta['samples']} bands={meta['bands']}")
            print(f"  data file:    {Path(meta['data_file']).name}")


def process_dfsar(scene_root: Path, cfg: dict) -> dict:
    discovered = discover_scene(scene_root)
    if not discovered["labels"]:
        raise FileNotFoundError(f"No DFSAR XML label found in {scene_root}")

    meta = parse_pds4_label(discovered["labels"][0], scene_root)
    raw = load_raw_array(meta)
    e_h, e_v = reconstruct_complex_channels(raw)
    validate_ingestion(meta, e_h, e_v)

    calibration = read_calibration_constants(_first_or_none(discovered["calibration_files"]))
    geometry = None
    incidence = meta.get("incidence_angle")
    geotransform = None
    m_per_pixel = 1.0
    if discovered["geometry_csvs"]:
        geometry = load_geometry(scene_root / "geometry", meta["lines"], meta["samples"])
        geotransform = build_geotransform(geometry, meta["lines"], meta["samples"])
        m_per_pixel = estimate_m_per_pixel(geometry, fallback=1.0)
        dense_incidence = incidence_grid(geometry, (meta["lines"], meta["samples"]))
        if dense_incidence is not None:
            incidence = dense_incidence

    looks_cfg = cfg["processing"]["multilook"]
    coherency = multilook_coherency(
        e_h,
        e_v,
        looks=(looks_cfg["azimuth_looks"], looks_cfg["range_looks"]),
        calibration_constants=calibration,
        incidence_angle_deg=incidence,
    )
    stokes = compute_stokes(coherency)
    cpr_dop = compute_cpr_dop(
        stokes,
        invert_s4_sign=bool(cfg["processing"]["circular_polarization"]["invert_s4_sign"]),
    )
    mchi = m_chi_decomposition(stokes, cpr_dop["DOP"])

    thresholds = cfg["processing"]["thresholds"]
    ice_mask = generate_ice_mask(
        cpr_dop["CPR"],
        cpr_dop["DOP"],
        stokes["S1"],
        mchi["P_odd"],
        cpr_thresh=thresholds["cpr"],
        dop_thresh=thresholds["dop"],
        volume_fraction_thresh=thresholds["volume_fraction"],
        min_ice_area_px=thresholds["min_ice_area_px"],
    )

    return {
        "meta": meta,
        "geometry": geometry,
        "geotransform": geotransform,
        "m_per_pixel": m_per_pixel,
        "stokes": stokes,
        "cpr_dop": cpr_dop,
        "mchi": mchi,
        "ice_mask": ice_mask,
    }


def process_ohrc(scene_root: Path, cfg: dict, target_shape: tuple[int, int]) -> dict:
    discovered = discover_scene(scene_root)
    if not discovered["labels"]:
        raise FileNotFoundError(f"No OHRC XML label found in {scene_root}")

    meta = parse_pds4_label(discovered["labels"][0], scene_root)
    raw = load_raw_array(meta)
    if raw.ndim == 3:
        raw = raw[..., 0]

    calibration = read_calibration_constants(_first_or_none(discovered["calibration_files"]))
    solar_azimuth = float(meta.get("solar_azimuth") or cfg["processing"]["ohrc"]["solar_azimuth_deg"])
    solar_elevation = float(meta.get("solar_elevation") or cfg["processing"]["ohrc"]["solar_elevation_deg"])
    reflectance = calibrate_ohrc(raw, calibration, solar_elevation)

    geometry = None
    m_per_pixel = float(cfg["processing"]["ohrc"]["fallback_m_per_pixel"])
    if discovered["geometry_csvs"]:
        geometry = load_geometry(scene_root / "geometry", meta["lines"], meta["samples"])
        m_per_pixel = estimate_m_per_pixel(geometry, fallback=m_per_pixel)

    thresholds = cfg["processing"]["thresholds"]
    psr_mask = detect_psr(reflectance, psr_threshold=thresholds["psr_reflectance"])
    crater_df = detect_craters(
        reflectance,
        geometry,
        fallback_m_per_pixel=m_per_pixel,
    )
    crater_df = tag_doubly_shadowed_craters(crater_df, psr_mask)
    slope = estimate_slope_from_shading(reflectance, solar_azimuth)
    boulder_df = detect_boulders(reflectance, solar_azimuth, solar_elevation, m_per_pixel=m_per_pixel)
    hazard = generate_hazard_map(
        slope,
        crater_df,
        boulder_df,
        reflectance.shape,
        slope_thresh=thresholds["slope_hazard"],
    )

    return {
        "meta": meta,
        "geometry": geometry,
        "m_per_pixel": m_per_pixel,
        "reflectance": reflectance,
        "psr_mask": psr_mask,
        "crater_df": crater_df,
        "boulder_df": boulder_df,
        "slope_proxy": slope,
        "hazard_map": hazard,
    }


def run_pipeline(config_path: str | Path) -> None:
    config_path = Path(config_path)
    cfg = load_config(config_path)
    base_dir = config_path.parent

    dfsar_root = resolve_path(base_dir, cfg["data"]["pradan"]["dfsar_root"])
    ohrc_root = resolve_path(base_dir, cfg["data"]["pradan"]["ohrc_root"])
    if dfsar_root is None:
        raise ValueError("config.yaml data.pradan.dfsar_root is empty.")
    if ohrc_root is None:
        raise ValueError("config.yaml data.pradan.ohrc_root is empty.")

    dfsar = process_dfsar(dfsar_root, cfg)
    shape = dfsar["stokes"]["S1"].shape
    ohrc = process_ohrc(ohrc_root, cfg, target_shape=shape)

    reflectance = resize_to_shape(ohrc["reflectance"], shape, order=1)
    psr_mask = resize_to_shape(ohrc["psr_mask"], shape, order=0).astype(np.uint8)
    hazard_map = resize_to_shape(ohrc["hazard_map"], shape, order=0).astype(np.uint8)
    doubly_shadowed = np.zeros(shape, dtype=np.uint8)
    if not ohrc["crater_df"].empty:
        for _, crater in ohrc["crater_df"].iterrows():
            if bool(crater.get("is_doubly_shadowed", False)):
                row = int(crater["pixel_row"] * shape[0] / ohrc["reflectance"].shape[0])
                col = int(crater["pixel_col"] * shape[1] / ohrc["reflectance"].shape[1])
                radius = max(1, int(crater["radius_px"] * shape[0] / ohrc["reflectance"].shape[0]))
                yy, xx = np.ogrid[: shape[0], : shape[1]]
                doubly_shadowed[(xx - col) ** 2 + (yy - row) ** 2 <= radius**2] = 1

    ice_confirmed = (
        (dfsar["ice_mask"] > 0)
        & (psr_mask > 0)
        & (doubly_shadowed > 0)
        & (hazard_map == 0)
    ).astype(np.uint8)

    p_even_fraction = dfsar["mchi"]["P_even"] / (dfsar["stokes"]["S1"] + 1e-10)
    layers = {
        "OHRC_reflectance": reflectance.astype(np.float32),
        "S1": dfsar["stokes"]["S1"].astype(np.float32),
        "CPR": dfsar["cpr_dop"]["CPR"].astype(np.float32),
        "DOP": dfsar["cpr_dop"]["DOP"].astype(np.float32),
        "P_odd_fraction": dfsar["mchi"]["P_odd_fraction"].astype(np.float32),
        "P_even_fraction": p_even_fraction.astype(np.float32),
        "ice_mask": dfsar["ice_mask"].astype(np.float32),
        "hazard_map": hazard_map.astype(np.float32),
        "psr_mask": psr_mask.astype(np.float32),
        "doubly_shadowed": doubly_shadowed.astype(np.float32),
        "ice_confirmed": ice_confirmed.astype(np.float32),
    }

    outputs = cfg["outputs"]
    write_analysis_stack(base_dir / outputs["analysis_stack_tif"], layers, dfsar["geotransform"])
    write_volume_csv(
        base_dir / outputs["volume_csv"],
        ohrc["crater_df"],
        ice_confirmed,
        dfsar["cpr_dop"]["CPR"],
        dfsar["cpr_dop"]["DOP"],
        dfsar["mchi"]["P_odd_fraction"],
        dfsar["m_per_pixel"],
    )
    write_rover_geojson(base_dir / outputs["rover_geojson"], ohrc["crater_df"])
    write_dashboard(base_dir / outputs["dashboard_png"], layers, ohrc["crater_df"])

    print("\n[OK] Pipeline complete")
    print(f"  stack:     {base_dir / outputs['analysis_stack_tif']}")
    print(f"  csv:       {base_dir / outputs['volume_csv']}")
    print(f"  geojson:   {base_dir / outputs['rover_geojson']}")
    print(f"  dashboard: {base_dir / outputs['dashboard_png']}")

