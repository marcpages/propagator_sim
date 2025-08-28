from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping
import yaml

from propagator.models import Fuel, FuelSystem


def fuels_from_yaml(path: str | Path) -> FuelSystem:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fuels_node = data.get("fuels")
    if not isinstance(fuels_node, Mapping):
        raise ValueError("YAML must contain 'fuels' (mapping)")
    # coerce IDs to int and build Fuel objects
    fuels: Dict[int, Fuel] = {}
    for k, v in fuels_node.items():
        try:
            fid = int(k)
        except Exception as e:
            raise ValueError(f"fuel ID '{k}' is not an integer") from e
        if not isinstance(v, Mapping):
            raise ValueError(f"fuel entry for ID {fid} must be a mapping")
        fuels[fid] = Fuel(**dict(v))
    # build the FuelSystem
    fs = FuelSystem(fuels=fuels)
    return fs
