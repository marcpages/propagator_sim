"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Protocol, Dict, Union, Tuple

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
    spread_probability: Dict[int, float]
    name: Optional[str] = None
    spotting: bool = False
    prob_ign_by_embers: float = 0.0
    burn: bool = True


@dataclass
class FuelSystem():
    fuels: Dict[int, Fuel]
    # derived attributes (filled in __post_init__)
    _n_fuels: int = field(init=False)
    _non_vegetated: int = field(init=False)
    # caches for fast vectorized lookups (filled in __post_init__)
    _ids_sorted: np.ndarray = field(init=False, repr=False)
    _id2idx: Dict[int, int] = field(init=False, repr=False)
    _v0_arr: np.ndarray = field(init=False, repr=False)
    _d0_arr: np.ndarray = field(init=False, repr=False)
    _d1_arr: np.ndarray = field(init=False, repr=False)
    _hhv_arr: np.ndarray = field(init=False, repr=False)
    _hum_arr: np.ndarray = field(init=False, repr=False)
    _embers_arr: np.ndarray = field(init=False, repr=False)
    _P: np.ndarray = field(init=False, repr=False)  # transition matrix

    def __post_init__(self):
        # checks on fuels
        n = len(self.fuels)
        self._n_fuels = n
        if n == 0:
            raise ValueError("at least one fuel is required")
        # check if fuel with burn=False is unique and save the code
        non_veg_ids = [fid for fid, f in self.fuels.items() if not f.burn]
        if len(non_veg_ids) == 0:
            raise ValueError("at least one fuel must have burn=False")
        if len(non_veg_ids) > 1:
            raise ValueError(f"only one fuel can have burn=False, got {non_veg_ids}")
        self._non_vegetated = non_veg_ids[0]
        # checks on transition probabilities
        expected_id_set = set(self.fuels.keys())
        for k, fuel in self.fuels.items():
            probs = fuel.spread_probability
            if len(probs) == 0:
                raise ValueError(f"fuel ID {k} must have set transition probabilities")
            if len(probs.keys()) != n:
                raise ValueError(f"fuel ID {k} must have {n} transition probabilities")
            for kk, p in probs.items():
                if kk not in expected_id_set:
                    raise ValueError(f"fuel ID {k} has unknown transition probability on fuel ID {kk}")
                if not (0.0 <= p <= 1.0):
                    raise ValueError(f"fuel ID {k} has invalid transition probability P[{k},{kk}]={probs[kk]}, must be in [0, 1]")
        # ---- build vectorization caches ----
        self._ids_sorted = np.array(sorted(self.fuels.keys()), dtype=int)
        self._id2idx = {fid: i for i, fid in enumerate(self._ids_sorted)}
        self._v0_arr = np.array([self.fuels[i].v0 for i in self._ids_sorted], dtype=float)
        self._d0_arr = np.array([self.fuels[i].d0 for i in self._ids_sorted], dtype=float)
        self._d1_arr = np.array([self.fuels[i].d1 for i in self._ids_sorted], dtype=float)
        self._hhv_arr = np.array([self.fuels[i].hhv for i in self._ids_sorted], dtype=float)
        self._hum_arr = np.array([self.fuels[i].humidity for i in self._ids_sorted], dtype=float)
        self._embers_arr = np.array([self.fuels[i].prob_ign_by_embers for i in self._ids_sorted], dtype=float)
        # full probability matrix P[i,j] with IDs mapped to compact indices
        self._P = np.empty((n, n), dtype=float)
        for j, fid_j in enumerate(self._ids_sorted):
            probs = self.fuels[fid_j].spread_probability
            self._P[:, j] = [probs[fid_i] for fid_i in self._ids_sorted]
        return self

    def get_keys(self) -> set[int]:
        return set(self.fuels.keys())

    def get_n_fuels(self) -> int:
        return self._n_fuels

    def get_non_vegetated(self) -> int:
        return self._non_vegetated

    def which_spotting(self) -> set[int]:
        return set(fid for fid, f in self.fuels.items() if f.spotting)

    # ---------- helpers ----------
    def _ids_to_indices(
        self, ids_arr: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.integer]:
        """Map fuel IDs to compact 0..n-1 indices (validates IDs)."""
        flat = ids_arr.ravel()
        missing = [int(x) for x in flat if int(x) not in self._id2idx]
        if missing:
            raise ValueError(f"unknown fuel ID(s): {sorted(set(missing))}")
        mapped = np.fromiter((self._id2idx[int(x)] for x in flat),
                             count=flat.size, dtype=int)
        return mapped.reshape(ids_arr.shape)

    # ---------- public getters (vectorized) ----------
    def get_transition_probability(
        self, from_id: npt.NDArray[np.integer], to_id: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]:
        fi = self._ids_to_indices(from_id)
        ti = self._ids_to_indices(to_id)
        # elementwise selection with broadcasting (e.g., scalar vs array)
        fi, ti = np.broadcast_arrays(fi, ti)
        return self._P[fi, ti]

    def get_v0(
        self, fuel_id: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]:
        idx = self._ids_to_indices(fuel_id)
        return self._v0_arr[idx]

    def get_d0(
        self, fuel_id: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]:
        idx = self._ids_to_indices(fuel_id)
        return self._d0_arr[idx]

    def get_d1(
        self, fuel_id: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]:
        idx = self._ids_to_indices(fuel_id)
        return self._d1_arr[idx]

    def get_hhv(
        self, fuel_id: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]:
        idx = self._ids_to_indices(fuel_id)
        return self._hhv_arr[idx]

    def get_humidity(
        self, fuel_id: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]:
        idx = self._ids_to_indices(fuel_id)
        return self._hum_arr[idx]

    def get_prob_ign_by_embers(
        self, fuel_id: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]:
        idx = self._ids_to_indices(fuel_id)
        return self._embers_arr[idx]


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
