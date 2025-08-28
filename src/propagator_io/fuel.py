from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping
import yaml

from propagator.models import Fuel, FuelSystem


def fuels_from_yaml(path: str | Path) -> FuelSystem:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    fuels_node = data.get("fuels")
    trans_node = data.get("transitions")
    if not isinstance(fuels_node, Mapping) or not isinstance(trans_node, Mapping):
        raise ValueError("YAML must contain 'fuels' (mapping) and 'transitions' (mapping)")
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
    # coerce transition IDs to int and values to float
    transition: Dict[int, Dict[int, float]] = {}
    for k, v in trans_node.items():
        try:
            from_id = int(k)
        except Exception as e:
            raise ValueError(f"transition row ID '{k}' is not an integer") from e
        if not isinstance(v, Mapping):
            raise ValueError(f"transition entry for ID {from_id} must be a mapping")
        transition[from_id] = {}
        for j, val in v.items():
            try:
                to_id = int(j)
            except Exception as e:
                raise ValueError(f"transition column ID '{j}' is not an integer") from e
            try:
                p = float(val)
            except Exception as e:
                raise ValueError(f"transition probability P[{from_id},{to_id}]='{val}' is not a float") from e
            transition[from_id][to_id] = p
    # build the FuelSystem
    fs = FuelSystem(fuels=fuels, transition=transition)
    return fs
