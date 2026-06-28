from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lunarice_net.config import load_config
from lunarice_net.dataset import LunarPatchDataset
from lunarice_net.losses import LunarIceLoss
from lunarice_net.model import AGMMUNet
from lunarice_net.preprocess import build_feature_stack
from lunarice_net.utils import set_seed


def load_or_build_arrays(config: dict) -> tuple[np.ndarray, np.ndarray]:
    processed = config["data"]["processed"]
    feature_path = Path(processed["feature_stack"])
    label_path = Path(processed["label_mask"])

    if feature_path.exists() and label_path.exists():
        return np.load(feature_path), np.load(label_path)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    features, _, _ = build_feature_stack(config)

    # TODO: Replace train_mask in config.yaml with your label/proxy mask path.
    with rasterio.open(config["data"]["labels"]["train_mask"]) as src:
        labels = src.read(1).astype(np.float32)
    labels = (labels > 0).astype(np.float32)

    np.save(feature_path, features)
    np.save(label_path, labels)
    return features, labels


def train(config_path: str) -> None:
    config = load_config(config_path)
    set_seed(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, labels = load_or_build_arrays(config)

    train_cfg = config["training"]
    dataset = LunarPatchDataset(
        features,
        labels,
        patch_size=train_cfg["patch_size"],
        length=2000,
        positive_fraction=train_cfg["positive_patch_fraction"],
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )

    model = AGMMUNet().to(device)
    criterion = LunarIceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"], weight_decay=1e-4)

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(train_cfg["epochs"]):
        model.train()
        running = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{train_cfg['epochs']}")

        for radar, aux, gate, target in progress:
            radar = radar.to(device)
            aux = aux.to(device)
            gate = gate.to(device)
            target = target.to(device)

            pred = model(radar, aux, gate)
            loss = criterion(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        epoch_loss = running / max(1, len(loader))
        torch.save(model.state_dict(), checkpoint_dir / "last_model.pt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), checkpoint_dir / "best_model.pt")

        print(f"Epoch {epoch + 1}: loss={epoch_loss:.4f}, best={best_loss:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()

