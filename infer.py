from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch

from lunarice_net.config import ensure_parent_dir, load_config
from lunarice_net.model import AGMMUNet
from lunarice_net.preprocess import build_feature_stack


def write_geotiff(path: str, array: np.ndarray, meta: dict) -> None:
    ensure_parent_dir(path)
    out_meta = meta.copy()
    out_meta.update(count=1, dtype="float32")
    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(array.astype(np.float32), 1)


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Enable dropout layers during inference for uncertainty estimation."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout2d):
            module.train()


@torch.no_grad()
def infer(config_path: str, checkpoint_path: str) -> None:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, _, meta = build_feature_stack(config)
    radar = torch.from_numpy(features[:7][None]).to(device)
    aux = torch.from_numpy(features[7:11][None]).to(device)
    gate = torch.from_numpy(features[10:11][None]).to(device)

    model = AGMMUNet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    passes = int(config["inference"]["mc_dropout_passes"])
    preds = []
    enable_mc_dropout(model)
    for _ in range(passes):
        preds.append(model(radar, aux, gate).cpu().numpy()[0, 0])

    pred_stack = np.stack(preds, axis=0)
    probability = pred_stack.mean(axis=0)
    uncertainty = pred_stack.std(axis=0)

    output_prob = config["inference"]["output_probability_tif"]
    output_unc = config["inference"]["output_uncertainty_tif"]
    Path(output_prob).parent.mkdir(parents=True, exist_ok=True)
    write_geotiff(output_prob, probability, meta)
    write_geotiff(output_unc, uncertainty, meta)

    print(f"Saved probability map: {output_prob}")
    print(f"Saved uncertainty map: {output_unc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    infer(args.config, args.checkpoint)


if __name__ == "__main__":
    main()

