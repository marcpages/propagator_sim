"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Protocol

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

    def to_dict(self, c_time: int, ref_date: datetime) -> dict[str, float|int|str]:
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
    fireline_int_mean: npt.NDArray[np.floating]
    fireline_int_max: npt.NDArray[np.floating]  
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
