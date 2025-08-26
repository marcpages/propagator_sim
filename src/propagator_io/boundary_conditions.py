from __future__ import annotations

from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from propagator.propagator import (
    BoundaryConditions,
)

# ---- project utils ----------------------------------------------------------
from propagator.utils import normalize
from propagator_io.actions import NO_FUEL_ACTION, NO_MOIST_ACTION, Action, parse_actions
from propagator_io.geo import GeographicInfo
from propagator_io.geometry import (
    DEFAULT_EPSG_GEOMETRY,
    Geometry,
    GeometryParser,
    rasterize_geometries,
)


# ---- wind helpers -----------------------------------------------------------
def meteo_deg_to_model_rad(deg: float) -> float:
    return normalize((180.0 - float(deg) + 90.0) * np.pi / 180.0)


# ---- simulation inputs ------------------------------------------------------
class TimedInput(BaseModel):
    """Single time-step boundary conditions."""

    model_config = ConfigDict(extra="allow")

    time: int = Field(0, description="minutes from simulation start")

    # Weather conditions
    w_dir: float = Field(
        default_factory=lambda: meteo_deg_to_model_rad(0.0),
        description="wind direction in radians",
    )  # ????
    w_speed: float = Field(0.0, description="wind speed in km/h")
    moisture: float = Field(
        0.0, ge=0.0, le=100.0, description="fuel moisture in percent (0-100)"
    )

    actions: Optional[list[Action]] = None

    # Optional per-step ignitions (POINT/LINE/POLYGON)
    ignitions: Optional[List[Geometry]] = None

    @field_validator("w_speed", mode="before")
    @classmethod
    def _coerce_speed(cls, v):
        if v is None:
            return 0.0
        return float(v)

    @field_validator("w_dir", mode="before")
    @classmethod
    def _coerce_wdir(cls, v):
        if v is None:
            return meteo_deg_to_model_rad(0.0)
        x = float(v)
        if x > 2 * np.pi or x < -2 * np.pi:
            return meteo_deg_to_model_rad(x)
        return normalize(x)

    @field_validator("time")
    @classmethod
    def _time_nonnegative(cls, v):
        if v < 0:
            raise ValueError("time must be >= 0")
        return v

    @model_validator(mode="before")
    @classmethod
    def _lift_legacy_actions(cls, data: dict, info):
        if not isinstance(data, dict):
            return data
        epsg = (info.context or {}).get("epsg", DEFAULT_EPSG_GEOMETRY)
        #  legacy ignitions parsing
        if "ignitions" in data:
            v = data["ignitions"]
            if isinstance(v, list) and (not v or isinstance(v[0], str)):
                data["ignitions"] = GeometryParser.parse_geometry_list(
                    v, allowed={"point", "line", "polygon"}, epsg=epsg
                )
        # let actions.py parse and normalize legacy fields
        new_actions, consumed = parse_actions(data, epsg=epsg)
        if len(new_actions) != 0:
            # append to any already-provided "actions"
            data["actions"] = list(data.get("actions", [])) + new_actions
            # remove consumed legacy keys so they don't error as "extra"
            for k in consumed:
                data.pop(k, None)
        return data

    def get_boundary_conditions(self, geo_info: GeographicInfo) -> BoundaryConditions:
        # rasterize weather conditions > so far given as scalars
        w_speed_arr = np.ones(geo_info.shape) * self.w_speed
        w_dir_arr = np.ones(geo_info.shape) * self.w_dir
        moisture_arr = np.ones(geo_info.shape) * self.moisture
        ignition_mask = None
        additional_moisture = None
        vegetation_changes = None

        if self.ignitions is not None:
            ignition_mask = rasterize_geometries(
                geometries=self.ignitions,
                geo_info=geo_info,
                default_value=1,  # set 1 for ignited pixels
                dtype="uint8",
                merge_alg="replace",
            )

        if self.actions is not None:
            additional_moisture = np.full(
                geo_info.shape, fill_value=NO_MOIST_ACTION, dtype=float
            )
            vegetation_changes = np.full(
                geo_info.shape, fill_value=NO_FUEL_ACTION, dtype=float
            )
            for action in self.actions:
                moist_action, fuel_action = action.rasterize_action(
                    geo_info,
                    non_vegetated=0.0,  # hardcoded for now
                )
                # accumulate actions
                additional_moisture += moist_action
                # substitute fuel actions (last one wins)
                np.putmask(
                    vegetation_changes, fuel_action != NO_FUEL_ACTION, fuel_action
                )

        # convert info in Propagator BoundaryConditions
        return BoundaryConditions(
            time=self.time,
            wind_speed=w_speed_arr,
            wind_dir=w_dir_arr,
            moisture=moisture_arr,
            ignition_mask=ignition_mask,
            additional_moisture=additional_moisture,
            vegetation_changes=vegetation_changes,
        )
