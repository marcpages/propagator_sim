from __future__ import annotations
from typing import List, Optional
import numpy as np
from pydantic import (BaseModel, ConfigDict, Field,
                      field_validator, model_validator)

from propagator_io.geometry import (
    GeometryParser, Geometry, GeoLine,
    DEFAULT_EPSG_GEOMETRY
)

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

    @model_validator(mode="before")
    @classmethod
    def _parse_geometry_fields(cls, data: dict, info):
        if not isinstance(data, dict):
            return data
        epsg = (info.context or {}).get("epsg", DEFAULT_EPSG_GEOMETRY)
        # field -> allowed kinds
        spec = {
            "ignitions": {"point", "line", "polygon"},
            "waterline_action": {"line"},
            "canadair": {"line"},
            "helicopter": {"line"},
            "heavy_action": {"line"},
        }
        for key, allowed in spec.items():
            val = data.get(key)
            if isinstance(val, list) and (not val or isinstance(val[0], str)):
                data[key] = GeometryParser.parse_geometry_list(
                    val, allowed=allowed, epsg=epsg
                )
        return data
