"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Protocol, Dict

import numpy as np
import numpy.typing as npt

# Integer coords array of shape (n, 3). We can’t encode the shape statically
# with stdlib typing, but we DO lock the dtype to integer families.
CoordsArray = npt.NDArray[np.integer]

# The payload shape we pass around
UpdateBatch = List[CoordsArray]

RNG = np.random.default_rng(12345)


class PropagatorError(Exception):
    """Domain-specific error raised by PROPAGATOR."""


@dataclass(frozen=True)
class Fuel():
    v0: float
    d0: float
    d1: float
    hhv: float
    humidity: float
    name: Optional[str] = None
    spotting: bool = False
    burn: bool = True


@dataclass
class FuelSystem():
    fuels: Dict[int, Fuel]
    transition: Dict[int, Dict[int, float]]
    _n_fuels: int = field(init=False)
    _non_vegetated: int = field(init=False)
    _ids_to_keys: Dict[int, int] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # checks on fuels
        n = len(self.fuels)
        self._n_fuels = n
        if n == 0:
            raise ValueError("at least one fuel is required")
        # the ids must be ordered, if not reorder the dictionary
        id_order = sorted(self.fuels.keys())
        self.fuels = {fid: self.fuels[fid] for fid in id_order}
        # set link ids-keys
        self._ids_to_keys = {id + 1: fid for id, fid in enumerate(self.fuels.keys())}
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
        expected_id_set = set(self.fuels.keys())
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

    def get_prob_table(self) -> npt.NDArray[np.floating]:
        prob_table = np.zeros((self._n_fuels, self._n_fuels), dtype=np.float64)
        for j, from_id in self._ids_to_keys.items():
            for i, to_id in self._ids_to_keys.items():
                prob_table[i-1, j-1] = self.transition[from_id][to_id]
        return prob_table

    def get_fuels(self) -> List[Fuel]:
        return [self.fuels[fid] for fid in self._ids_to_keys.values()]

    def get_non_vegetated(self) -> int:
        return self._non_vegetated

    def convert_fuel_array(self, fuel_in: npt.NDArray[np.integer]) -> npt.NDArray[np.integer]:
        shape = fuel_in.shape
        fuel_out = np.zeros(shape, dtype=np.int32)
        for fid, k in self._ids_to_keys.items():
            fuel_out[fuel_in == k] = fid
        return fuel_out


@dataclass(frozen=True)
class Ignitions:
    time: int
    coords: CoordsArray


@dataclass(frozen=True)
class BoundaryConditions:
    """Boundary conditions applied at or after a given time.

    - time: Simulation time the conditions refer to.
    - ignitions: Boolean mask of new ignition points (True ignites).
    - moisture: Fuel moisture map (%), same shape as vegetation.
    - wind_dir: Wind direction map (radians, mathematical convention).
    - wind_speed: Wind speed map (km/h).
    - additional_moisture: Extra moisture to add to fuel (%), can be sparse.
    - vegetation_changes: Raster of vegetation type overrides (NaN to skip).
    """

    time: int
    moisture: Optional[npt.NDArray[np.floating]] = None
    wind_dir: Optional[npt.NDArray[np.floating]] = None
    wind_speed: Optional[npt.NDArray[np.floating]] = None
    ignition_mask: Optional[npt.NDArray[np.bool_]] = None
    additional_moisture: Optional[npt.NDArray[np.floating]] = None
    vegetation_changes: Optional[npt.NDArray[np.floating]] = None


@dataclass(frozen=True)
class PropagatorStats:
    """Summary statistics for the current simulation state."""

    n_active: int
    area_mean: float
    area_50: float
    area_75: float
    area_90: float

    def to_dict(self, c_time: int, ref_date: datetime) -> dict[str, float | int | str]:
        return dict(
            c_time=c_time,
            ref_date=ref_date.isoformat(),
            n_active=self.n_active,
            area_mean=self.area_mean,
            area_50=self.area_50,
            area_75=self.area_75,
            area_90=self.area_90,
        )


@dataclass(frozen=True)
class PropagatorOutput:
    """Snapshot of simulation outputs at a given time step."""

    time: int
    fire_probability: npt.NDArray[np.floating]
    ros_mean: npt.NDArray[np.floating]
    ros_max: npt.NDArray[np.floating]
    fli_mean: npt.NDArray[np.floating]
    fli_max: npt.NDArray[np.floating]
    stats: PropagatorStats


class PTimeFn(Protocol):
    """Callable protocol for rate-of-spread time functions.

    Accepts model-specific positional arguments; returns a tuple with
    transition times and rate of spread arrays.

    Returns
    -------
    tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]
        A pair of arrays (time_minutes, ros_m_per_min).
    """

    def __call__(
        self, *args, **kwargs
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ...


class PMoistFn(Protocol):
    """Callable protocol for moisture probability correction functions.

    Parameters
    ----------
    moist : npt.NDArray[np.floating]
        Moisture content array.

    Returns
    -------
    npt.NDArray[np.floating]
        Probability multiplier in [0, 1].
    """

    def __call__(self, moist: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]: ...
