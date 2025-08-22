from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal
from pathlib import Path
from pydantic import (BaseModel, Field,
                      field_validator, model_validator)

# ---- project utils ----------------------------------------------------------
from propagator.functions import get_p_moist_fn, get_p_time_fn

from propagator_io.boundary_conditions import BoundaryConditionsInput
from propagator_io.geometry import _coerce_geometry_list, Geometry


class PropagatorConfigurationLegacy(BaseModel):
    """Propagator configuration"""

    mode: Literal["tileset", "geotiff"] = Field(
        "tileset",
        description="Mode of static data load: 'tileset' for automatic, "
                    "'geotiff' for giving DEM and FUEL in input."
                    "[default: tileset]",
    )
    dem: Optional[Path] = Field(
        None,
        description="Path to DEM file (GeoTIFF), "
                    "required in 'geotiff' mode",
    )
    fuel: Optional[Path] = Field(
        None,
        description="Path to FUEL file (GeoTIFF), "
                    "required in 'geotiff' mode",
    )
    output: Path = Field(
        ...,
        description="Path to output folder where results will be saved",
    )
    record: bool = Field(
        False,
        description="Export run logs",
    )
    realizations: int = Field(
        1, ge=1,
        description="Number of realizations"
    )
    init_date: datetime = Field(
        default_factory=datetime.now,
        description="Datetime of the simulated event"
    )
    time_resolution: int = Field(
        60, gt=0,
        description="Simulation resolution [minutes]"
    )
    time_limit: int = Field(
        1440, gt=0,
        description="Simulation limit [minutes]"
    )
    epsg: int = Field(
        4326,
        description="EPSG of geometries"
    )
    ignitions: Optional[List[Geometry]] = Field(
        None,
        description="List of ignitions at simulation start (time=0)."
    )
    boundary_conditions: List[BoundaryConditionsInput] = Field(
        default_factory=list,
        description="List of boundary conditions"
    )
    do_spotting: bool = Field(
        False,
        description="Spotting option"
    )
    ros_model: str = Field(
        "default",
        description="ROS model name"
    )
    prob_moist_model: str = Field(
        "default",
        description="Moisture model name"
    )
    p_time_fn: Optional[object] = Field(
        default=None,
        exclude=True
    )
    p_moist_fn: Optional[object] = Field(
        default=None,
        exclude=True
    )

    # ---------- checks ----------
    @field_validator("init_date", mode="before")
    @classmethod
    def _parse_init_date(cls, v: str | datetime) -> datetime:
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            fmt_ok = ("%Y%m%d%H%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")
            for fmt in fmt_ok:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            # if no format matched, raise error
            raise ValueError(f"init_date string not recognized: {v!r}. "
                             f"Expected formats: {fmt_ok}")
        # if v is neither str nor datetime, raise error
        raise TypeError(f"init_date must be a datetime or string, \
            got {type(v).__name__}")

    @field_validator("ignitions", mode="before")
    @classmethod
    def _coerce_top_ignitions(cls, v: None | List[str]
                              ) -> Optional[List[Geometry]]:
        return _coerce_geometry_list(v, allowed={"point", "line", "polygon"},
                                     field_name="ignitions")

    # ---------- cross-field checks & friendly console messages ----------
    @model_validator(mode="after")
    def _post_setup(self):
        # geotiff mode: DEM/FUEL required and must exist
        if self.mode == "geotiff":
            if not self.dem:
                raise ValueError("DEM path must be set in 'geotiff' mode")
            if not self.dem.is_file():
                raise ValueError("DEM file not found")

            if not self.fuel:
                raise ValueError("FUEL path must be set in 'geotiff' mode")
            if not self.fuel.is_file():
                raise ValueError("FUEL file not found")

        # set the functions
        self.p_time_fn = get_p_time_fn(self.ros_model)
        self.p_moist_fn = get_p_moist_fn(self.prob_moist_model)
        if self.p_time_fn is None:
            raise ValueError(f"Unknown ROS model: {self.ros_model}")
        if self.p_moist_fn is None:
            raise ValueError(f"Unknown moisture model: \
                {self.prob_moist_model}")

        # sort the boundary conditions
        self.boundary_conditions.sort(key=lambda bc: bc.time)

        # check if boundary condition is empty
        if len(self.boundary_conditions) == 0:
            raise ValueError("boundary_conditions must not be empty.")

        # check if time == 0 is present
        t0_bc = next((bc for bc in self.boundary_conditions if bc.time == 0),
                     None)
        if t0_bc is None:
            raise ValueError(
                "boundary_conditions must include an entry with time = 0.")

        # add initial ignitions (if present) to the firt boundary condition
        if self.ignitions:
            if t0_bc.ignitions is None:
                t0_bc.ignitions = []
            t0_bc.ignitions.extend(self.ignitions)
            # # single source of truth: clear at top-level
            # self.ignitions = None

        # now, check if t0 has an ignition > must have, otherwise error
        if not t0_bc.ignitions or len(t0_bc.ignitions) == 0:
            raise ValueError(
                "Initial ignitions must be provided either at top-level or in "
                "the first boundary condition (time=0)."
            )

        # check if there are repetitions in boundary conditions
        times = [bc.time for bc in self.boundary_conditions]
        if len(times) != len(set(times)):
            raise ValueError("boundary_conditions have duplicate times.")

        return self
