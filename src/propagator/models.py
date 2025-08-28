"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Protocol, Dict, Any

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

    def __post_init__(self):
        # checks on fuels
        n = len(self.fuels)
        self._n_fuels = n
        if n == 0:
            raise ValueError("at least one fuel is required")
        # check if a key is zero, if so make it an error
        if 0 in self.fuels:
            raise ValueError("fuel IDs must be positive integers, got 0")
        # check if fuel with burn=False is unique and save the code
        non_veg_ids = [fid for fid, f in self.fuels.items() if not f.burn]
        if len(non_veg_ids) == 0:
            raise ValueError("at least one fuel must have burn=False")
        if len(non_veg_ids) > 1:
            raise ValueError(f"only one fuel can have burn=False, \
                got {non_veg_ids}")
        self._non_vegetated = non_veg_ids[0]
        # checks on transition probabilities
        expected_id_set = set(self.fuels.keys())
        for k, fuel in self.fuels.items():
            probs = fuel.spread_probability
            if len(probs) == 0:
                raise ValueError(f"fuel ID {k} must have set \
                    transition probabilities")
            if len(probs.keys()) != n:
                raise ValueError(f"fuel ID {k} must have {n} \
                    transition probabilities")
            for kk, p in probs.items():
                if kk not in expected_id_set:
                    raise ValueError(f"fuel ID {k} has unknown \
                        transition probability on fuel ID {kk}")
                if not (0.0 <= p <= 1.0):
                    raise ValueError(f"fuel ID {k} has invalid \
                        transition probability P[{k},{kk}]={probs[kk]}, \
                            must be in [0, 1]")
        return self

    def get_keys(self) -> set[int]:
        return set(self.fuels.keys())

    def get_n_fuels(self) -> int:
        return self._n_fuels

    def get_non_vegetated(self) -> int:
        return self._non_vegetated

    def which_spotting(self) -> set[int]:
        return set(fid for fid, f in self.fuels.items() if f.spotting)

    # ---------- public getters ----------
    # function for which I give an array of IDs and get an array of Fuel
    def get_fuels(
        self, ids_arr: npt.NDArray[np.integer]
    ) -> npt.NDArray[Any]:
        """Map fuel IDs array to Fuel objects array (validates IDs)."""
        flat = ids_arr.ravel()
        missing = [int(x) for x in flat if int(x) not in self.fuels]
        if missing:
            raise ValueError(f"unknown fuel ID(s): {sorted(set(missing))}")
        mapped = np.fromiter((self.fuels[int(x)] for x in flat),
                             count=flat.size, dtype=object)
        return mapped.reshape(ids_arr.shape)

    def get_fuel(self, fuel_id: int) -> Fuel:
        fuel = self.fuels.get(fuel_id, None)
        if fuel is None:
            raise ValueError(f"unknown fuel ID: {fuel_id}")
        return fuel

    def get_transition_probabilities(
        self,
        from_ids: npt.NDArray[np.integer], to_ids: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]:
        """Get array of transition probabilities P[from_id, to_id]."""
        if from_ids.shape != to_ids.shape:
            raise ValueError("from_ids and to_ids must have the same shape")
        flat_from = from_ids.ravel()
        flat_to = to_ids.ravel()
        missing = [int(x) for x in flat_from if int(x) not in self.fuels]
        if missing:
            raise ValueError(f"unknown fuel-from ID(s): \
                {sorted(set(missing))}")
        missing = [int(x) for x in flat_to if int(x) not in self.fuels]
        if missing:
            raise ValueError(f"unknown fuel-to ID(s): \
                {sorted(set(missing))}")
        probs = np.fromiter(
            (self.fuels[int(f)].spread_probability[int(t)]
             for f, t in zip(flat_from, flat_to)),
            count=flat_from.size,
            dtype=np.float64,
        )
        return probs.reshape(from_ids.shape)


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

    def to_dict(
        self, c_time: int, ref_date: datetime
    ) -> dict[str, float | int | str]:
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

    def __call__(self, moist: npt.NDArray[np.floating]
                 ) -> npt.NDArray[np.floating]: ...
