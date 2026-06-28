from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_parent_dir(path: str | Path) -> None:
    """Create the parent folder for an output file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

