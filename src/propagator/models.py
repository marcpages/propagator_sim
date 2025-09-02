"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Protocol, Any

import numpy as np
import numpy.typing as npt
from numba.experimental import jitclass
from numba import types
from numba.typed import Dict


# Integer coords array of shape (n, 3). We can’t encode the shape statically
# with stdlib typing, but we DO lock the dtype to integer families.
CoordsTuple = tuple[int, int, int]

# The payload shape we pass around
UpdateBatch = List[CoordsTuple]

RNG = np.random.default_rng(12345)


class PropagatorError(Exception):
    """Domain-specific error raised by PROPAGATOR."""


spec = [
    ("v0", types.float64),
    ("d0", types.float64),
    ("d1", types.float64),
    ("hhv", types.float64),
    ("humidity", types.float64),
    ("spotting", types.boolean),
    ("prob_ign_by_embers", types.float64),
    ("burn", types.boolean),
    ("name", types.string)
]
# v0: float
# d0: float
# d1: float
# hhv: float
# humidity: float
# spread_probability: Dict
# spotting: bool
# prob_ign_by_embers: float
# burn: bool
# name: str


@jitclass(spec)
class Fuel:
    def __init__(
        self,
        v0: float,
        d0: float,
        d1: float,
        hhv: float,
        humidity: float,
        name: str,
        spotting: bool = False,
        prob_ign_by_embers: float = 0.0,
        burn: bool = True
    ):
        self.v0 = v0
        self.d0 = d0
        self.d1 = d1
        self.hhv = hhv
        self.humidity = humidity
        self.spotting = spotting
        self.prob_ign_by_embers = prob_ign_by_embers
        self.burn = burn
        self.name = name


spec = [
    ("fuels_id", types.DictType(types.int64, types.int64)),
    ("v0", types.float64[:]),
    ("d0", types.float64[:]),
    ("d1", types.float64[:]),
    ("hhv", types.float64[:]),
    ("humidity", types.float64[:]),
    # ("spread_probability", types.DictType(
    #     types.int64, types.float64[:]
    # )),
    ("spread_probability", types.float64[:, :]),
    ("spotting", types.boolean[:]),
    ("prob_ign_by_embers", types.float64[:]),
    ("burn", types.boolean[:]),
    ("name", types.DictType(types.int64, types.string)),
    ("_non_vegetated", types.int64)
]


@jitclass(spec)
class FuelSystem:

    def __init__(self, n_fuels: int):
        self.fuels_id = Dict.empty(
            key_type=types.int64,
            value_type=types.int64
        )
        self.v0 = np.zeros(n_fuels, dtype=np.float64)
        self.d0 = np.zeros(n_fuels, dtype=np.float64)
        self.d1 = np.zeros(n_fuels, dtype=np.float64)
        self.hhv = np.zeros(n_fuels, dtype=np.float64)
        self.humidity = np.zeros(n_fuels, dtype=np.float64)
        self.spread_probability = np.zeros(
            (n_fuels, n_fuels), dtype=np.float64
        )
        self.spotting = np.zeros(n_fuels, dtype=np.bool_)
        self.prob_ign_by_embers = np.zeros(n_fuels, dtype=np.float64)
        self.burn = np.ones(n_fuels, dtype=np.bool_)
        self.name = Dict.empty(
            key_type=types.int64,
            value_type=types.string
        )
        self._non_vegetated = -1

    def get_non_vegetated(self) -> int:
        return self._non_vegetated

    # def which_spotting(self) -> set[int]:
    #     return set(fid for fid, f in self.spotting.items() if f)

    # ---------- public getters ----------
    def get_transition_probability(self, from_id: int, to_id: int) -> float:
        if from_id not in self.fuels_id or to_id not in self.fuels_id:
            raise PropagatorError(
                f"Fuel IDs {from_id} or {to_id} do not exist."
            )
        i = self.fuels_id[from_id]
        j = self.fuels_id[to_id]
        return self.spread_probability[i, j]  # type: ignore

    def add_fuel(
        self,
        fuel_id: int,
        name: str,
        v0: float,
        d0: float,
        d1: float,
        hhv: float,
        humidity: float,
        spotting: bool = False,
        prob_ign_by_embers: float = 0.0,
        burn: bool = True
    ) -> None:
        n = len(self.fuels_id.keys())
        if fuel_id in self.fuels_id:
            raise PropagatorError(f"Fuel ID {fuel_id} already exists.")
        self.fuels_id[fuel_id] = n
        self.v0[n] = v0
        self.d0[n] = d0
        self.d1[n] = d1
        self.hhv[n] = hhv
        self.humidity[n] = humidity
        self.spotting[n] = spotting
        self.prob_ign_by_embers[n] = prob_ign_by_embers
        self.burn[n] = burn
        self.name[n] = name
        if not burn:
            self._non_vegetated = fuel_id

    def add_transition_probability(
        self, from_id: int, to_id: int, prob: float
    ) -> None:
        if from_id not in self.fuels_id or to_id not in self.fuels_id:
            raise PropagatorError(
                f"Fuel IDs {from_id} or {to_id} do not exist."
            )
        i = self.fuels_id[from_id]
        j = self.fuels_id[to_id]
        self.spread_probability[i, j] = prob

    def get_fuel(self, fuel_id: int) -> Fuel:
        if fuel_id not in self.fuels_id:
            raise PropagatorError(f"Fuel ID {fuel_id} does not exist.")
        i = self.fuels_id[fuel_id]
        return Fuel(
            self.v0[i],  # type: ignore
            self.d0[i],  # type: ignore
            self.d1[i],  # type: ignore
            self.hhv[i],  # type: ignore
            self.humidity[i],  # type: ignore
            self.name[i],  # type: ignore
            self.spotting[i],  # type: ignore
            self.prob_ign_by_embers[i],  # type: ignore
            self.burn[i]  # type: ignore
        )


def fuelsystem_from_dict(fuels: dict[int, dict]) -> FuelSystem:
    n_fuels = len(fuels)
    fuelsystem = FuelSystem(n_fuels)
    for k, fuel in fuels.items():
        fuelsystem.add_fuel(
            k,
            fuel["name"],
            fuel["v0"],
            fuel["d0"],
            fuel["d1"],
            fuel["hhv"],
            fuel["humidity"],
            fuel.get("spotting", False),
            fuel.get("prob_ign_by_embers", 0.0),
            fuel.get("burn", True)
        )
    for from_id, fuel in fuels.items():
        for to_id, prob in fuel["spread_probability"].items():
            fuelsystem.add_transition_probability(from_id, to_id, prob)
    return fuelsystem


@dataclass(frozen=True)
class Ignitions:
    time: int
    coords: CoordsTuple


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

    def __call__(self, *args, **kwargs) -> tuple[float, float]: ...


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

    def __call__(self, moist: float) -> float: ...
