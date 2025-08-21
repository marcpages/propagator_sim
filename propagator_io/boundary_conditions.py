from __future__ import annotations
from typing import List, Optional
import numpy as np
from pydantic import (BaseModel, ConfigDict, Field, field_validator)

from propagator_io.geometry import Geometry, GeoLine, _coerce_geometry_list

# ---- project utils ----------------------------------------------------------
from propagator.utils import normalize


# ---- wind helpers -----------------------------------------------------------
def meteo_deg_to_model_rad(deg: float) -> float:
    return normalize((180.0 - float(deg) + 90.0) * np.pi / 180.0)


# ---- simulation inputs ------------------------------------------------------
class BoundaryConditionsInput(BaseModel):
    """Single time-step boundary conditions."""
    model_config = ConfigDict(extra="allow")

    time: int = Field(0, description="minutes from simulation start")

    # Weather conditions
    w_dir: float = Field(default_factory=lambda: meteo_deg_to_model_rad(0.0),
                         description="wind direction in radians")  # ????
    w_speed: float = Field(0.0, description="wind speed in km/h")
    moisture: float = Field(0.0, ge=0.0, le=100.0,
                            description="fuel moisture in percent (0-100)")

    # Per-step actions (LINE-only)
    waterline_action: Optional[List[GeoLine]] = None
    canadair: Optional[List[GeoLine]] = None
    helicopter: Optional[List[GeoLine]] = None
    heavy_action: Optional[List[GeoLine]] = None

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
        if x > 2*np.pi or x < -2*np.pi:
            return meteo_deg_to_model_rad(x)
        return normalize(x)

    @field_validator("time")
    @classmethod
    def _time_nonnegative(cls, v):
        if v < 0:
            raise ValueError("time must be >= 0")
        return v

    @field_validator("waterline_action", "canadair",
                     "helicopter", "heavy_action",
                     mode="before")
    @classmethod
    def _coerce_lines(cls, v, info):
        return _coerce_geometry_list(v, allowed={"line"},
                                     field_name=info.field_name)

    @field_validator("ignitions", mode="before")
    @classmethod
    def _coerce_perstep_ignitions(cls, v):
        return _coerce_geometry_list(v,
                                     allowed={"point", "line", "polygon"},
                                     field_name="ignitions")
