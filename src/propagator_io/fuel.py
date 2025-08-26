from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, Field, ConfigDict, model_validator, PrivateAttr


# -------------------- leaf model --------------------
class Fuel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="fuel label")
    v0: float = Field(..., ge=0.0, description="nominal rate of spread (m/min)")
    d0: float = Field(..., ge=0.0, description="dead fuel load (kg/m²)")
    d1: float = Field(..., ge=0.0, description="live fuel load (kg/m²)")
    hhv: float = Field(..., gt=0.0, description="higher heating value (kJ/kg)")
    umid: float = Field(..., ge=0.0, le=100.0, description="live fuel moisture (%)")
    do_spotting: bool = False
    burn: bool = True


# -------------------- system model --------------------
class FuelSystem(BaseModel):
    """
    Fuel catalog + transition matrix.
    Rows/cols of `transition` are aligned to `id_order` (sorted fuel IDs).
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    fuels: Dict[int, Fuel]
    transition: npt.NDArray[np.float64]

    # computed at load time; not part of the schema
    _id_order: List[int] = PrivateAttr(default_factory=list)
    _id_to_idx: Dict[int, int] = PrivateAttr(default_factory=dict)

    # ---------- validation ----------
    @model_validator(mode="after")
    def _check_and_index(self):
        n = len(self.fuels)
        if n == 0:
            raise ValueError("at least one fuel is required")

        # unique names
        names = [f.name for f in self.fuels.values()]
        if len(set(names)) != n:
            raise ValueError("fuel names must be unique")

        # set (or reset) alignment
        self._id_order = sorted(self.fuels.keys())
        self._id_to_idx = {fid: i for i, fid in enumerate(self._id_order)}

        P = np.asarray(self.transition, dtype=np.float64)
        if P.shape != (n, n):
            raise ValueError(f"transition shape {P.shape} != ({n}, {n})")

        if not np.all(np.isfinite(P)):
            raise ValueError("transition contains non-finite values")

        if np.any(P < 0.0) or np.any(P > 1.0):
            raise ValueError("transition probabilities must be in [0, 1]")

        # rows must be (approximately) stochastic
        row_sums = P.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-9):
            raise ValueError(f"each row of transition must sum to 1 (got {row_sums})")

        self.transition = P
        return self

    # ---------- helpers ----------
    @property
    def id_order(self) -> List[int]:
        return list(self._id_order)

    def idx_of(self, fuel_id: int) -> int:
        return self._id_to_idx[fuel_id]

    def name_of(self, fuel_id: int) -> str:
        return self.fuels[fuel_id].name

    def get_fuel_names(self) -> List[str]:
        return [self.fuels[i].name for i in self._id_order]

    def prob(self, from_id: int, to_id: int) -> float:
        i = self._id_to_idx[from_id]
        j = self._id_to_idx[to_id]
        return float(self.transition[i, j])

    # ---------- YAML I/O ----------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "FuelSystem":
        """
        YAML schema (clean & minimal):

        fuels:
          1:
            name: grassland
            v0: 100
            d0: 1.5
            d1: 0.0
            hhv: 18000
            umid: 60
            burn: false
          2:
            name: shrubland
            v0: 200
            d0: 2.0
            d1: 0.5
            hhv: 19000
            umid: 50

        transitions:
          1: {1: 0.7, 2: 0.3}
          2: {1: 0.4, 2: 0.6}
        """
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

        # fixed order by sorted IDs
        id_order = sorted(fuels.keys())
        idx_by_id = {fid: i for i, fid in enumerate(id_order)}
        n = len(id_order)
        P = np.zeros((n, n), dtype=np.float64)

        # fill P from nested maps; require exact ID set per row
        for row_id, row_map in trans_node.items():
            try:
                from_id = int(row_id)
            except Exception as e:
                raise ValueError(f"transition row key '{row_id}' is not an integer") from e
            if from_id not in idx_by_id:
                raise ValueError(f"transition row for unknown fuel ID {from_id}")
            if not isinstance(row_map, Mapping):
                raise ValueError(f"transition row {from_id} must be a mapping")

            i = idx_by_id[from_id]
            expected_to_ids = set(id_order)
            seen_to_ids = set()

            for to_id_str, p in row_map.items():
                try:
                    to_id = int(to_id_str)
                except Exception as e:
                    raise ValueError(f"transition column key '{to_id_str}' is not an integer") from e
                if to_id not in idx_by_id:
                    raise ValueError(f"transition to unknown fuel ID {to_id}")
                j = idx_by_id[to_id]
                P[i, j] = float(p)
                seen_to_ids.add(to_id)

            missing = expected_to_ids - seen_to_ids
            if missing:
                raise ValueError(f"transition row {from_id} missing columns for IDs {sorted(missing)}")

        fs = cls(fuels=fuels, transition=P)
        # populate private attrs for convenience (already done in validator too)
        fs._id_order = id_order
        fs._id_to_idx = idx_by_id
        return fs

    def to_yaml(self, path: str | Path) -> None:
        """Write the same schema produced by `from_yaml`."""
        path = Path(path)
        fuels_node = {fid: self.fuels[fid].model_dump() for fid in self._id_order}
        trans_node: Dict[int, Dict[int, float]] = {}
        for i, from_id in enumerate(self._id_order):
            trans_node[from_id] = {to_id: float(self.transition[i, j])
                                   for j, to_id in enumerate(self._id_order)}
        data = {"fuels": fuels_node, "transitions": trans_node}
        path.write_text(yaml.safe_dump(data, sort_keys=True, allow_unicode=True), encoding="utf-8")
