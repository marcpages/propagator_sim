from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, model_validator, PrivateAttr

from propagator.models import Fuel


# -------------------- system model --------------------
class FuelSystem(BaseModel):
    """
    Fuel catalog + transition matrix.
    """
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    fuels: Dict[int, Fuel]
    transition: Dict[int, Dict[int, float]]
    _n_fuels: int = PrivateAttr()
    _non_vegetated: int = PrivateAttr()

    # ---------- validation ----------
    @model_validator(mode="after")
    def _check_and_index(self):
        # checks on fuels
        n = len(self.fuels)
        self._n_fuels = n
        if n == 0:
            raise ValueError("at least one fuel is required")
        # the ids must be contiguous 1..n
        id_set = set(self.fuels.keys())
        expected_id_set = set(range(1, n + 1))
        if id_set != expected_id_set:
            raise ValueError(f"fuel IDs must be contiguous 1..{n}, got {sorted(id_set)}")
        # the ids must be ordered, if not reorder the dictionary
        id_order = sorted(self.fuels.keys())
        self.fuels = {fid: self.fuels[fid] for fid in id_order}
        # check if fuel with burn=False is unique and save the code
        non_veg_ids = [fid for fid, f in self.fuels.items() if not f.burn]
        if len(non_veg_ids) == 0:
            raise ValueError("at least one fuel must have burn=False")
        if len(non_veg_ids) > 1:
            raise ValueError(f"only one fuel can have burn=False, got {non_veg_ids}")
        self._non_vegetated = non_veg_ids[0]
        # checks on transition
        if len(self.transition.keys()) != n:
            raise ValueError(f"transition must have {n} rows")
        for k in self.transition.keys():
            if k not in expected_id_set:
                raise ValueError(f"transition contains unknown fuel ID {k}")
            row = self.transition[k]
            if len(row.keys()) != n:
                raise ValueError(f"transition row {k} must have {n} entries")
            for j in row.keys():
                if j not in expected_id_set:
                    raise ValueError(f"transition row {k} contains unknown fuel ID {j}")
                val = float(row[j])
                if not (0.0 <= val <= 1.0):
                    raise ValueError(f"transition probability P[{k},{j}]={val} must be in [0, 1]")
                self.transition[k][j] = val  # coerce to float
        return self

    # ---------- helpers ----------

    def get_prob(self, from_id: int, to_id: int) -> float:
        return float(self.transition[from_id][to_id])

    def get_prob_table(self) -> npt.NDArray[np.floating]:
        prob_table = np.zeros((self._n_fuels, self._n_fuels), dtype=np.float64)
        for j, from_id in enumerate(sorted(self.fuels.keys())):
            for i, to_id in enumerate(sorted(self.fuels.keys())):
                prob_table[i, j] = self.get_prob(from_id, to_id)
        return prob_table

    def get_fuels(self) -> List[Fuel]:
        return [self.fuels[fid] for fid in sorted(self.fuels.keys())]

    def get_non_vegetated(self) -> int:
        return self._non_vegetated

    # ---------- YAML I/O ----------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "FuelSystem":
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
        fs = cls(fuels=fuels, transition=transition)
        return fs


# --- FUEL SYSTEM LEGACY ---
FUEL_SYSTEM_LEGACY = FuelSystem(
    fuels={
        1: Fuel(name="broadleaves", v0=140, d0=1.5, d1=3, hhv=20000, humidity=60, spotting=False, burn=True),
        2: Fuel(name="shrubs", v0=140, d0=1, d1=3, hhv=21000, humidity=45, spotting=False, burn=True),
        3: Fuel(name="non-vegetated", v0=20, d0=0.1, d1=0, hhv=100, humidity=-9999, spotting=False, burn=False),
        4: Fuel(name="grassland", v0=120, d0=0.5, d1=0, hhv=17000, humidity=-9999, spotting=False, burn=True),
        5: Fuel(name="conifers", v0=200, d0=1, d1=4, hhv=21000, humidity=55, spotting=True, burn=True),
        6: Fuel(name="agro-forestry areas", v0=120, d0=0.5, d1=2, hhv=19000, humidity=60, spotting=False, burn=True),
        7: Fuel(name="non-fire prone forests", v0=60, d0=1, d1=2, hhv=18000, humidity=65, spotting=False, burn=True)
    },
    transition={
        # burning cell j: {neighbor cell i: probability j->i}
        1: {1: 0.3, 2: 0.375, 3: 0.005, 4: 0.45, 5: 0.225, 6: 0.25, 7: 0.075},
        2: {1: 0.375, 2: 0.375, 3: 0.005, 4: 0.475, 5: 0.325, 6: 0.25, 7: 0.1},
        3: {1: 0.005, 2: 0.005, 3: 0.005, 4: 0.005, 5: 0.005, 6: 0.005, 7: 0.005},
        4: {1: 0.25, 2: 0.35, 3: 0.005, 4: 0.475, 5: 0.1, 6: 0.3, 7: 0.075},
        5: {1: 0.275, 2: 0.4, 3: 0.005, 4: 0.475, 5: 0.35, 6: 0.475, 7: 0.275},
        6: {1: 0.25, 2: 0.3, 3: 0.005, 4: 0.375, 5: 0.2, 6: 0.35, 7: 0.075},
        7: {1: 0.25, 2: 0.375, 3: 0.005, 4: 0.475, 5: 0.35, 6: 0.25, 7: 0.075}
    },
)
