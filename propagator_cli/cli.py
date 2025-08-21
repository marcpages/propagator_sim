from __future__ import annotations

from typing import Optional, Literal
from pathlib import Path
import json
from pydantic import (Field, field_validator)
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---- project utils ----------------------------------------------------------
from propagator_io.configuration import PropagatorConfigurationLegacy


# --- CLI configuration -------------------------------------------------------
class PropagatorCLILegacy(BaseSettings):
    model_config = SettingsConfigDict(cli_parse_args=True)

    config: Path = Field(
        ...,
        description="Path to configuration file (JSON)"
    )
    mode: Literal["tileset", "geotiff"] = Field(
        "tileset",
        description="Mode of static data load: 'tileset' for automatic, "
                    "'geotiff' for giving DEM and FUEL in input."
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

    # ---------- checks ----------
    @field_validator("config", mode="before")
    @classmethod
    def _check_config_file(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        # check if the file exists
        if not v.is_file():
            raise ValueError("Configuration file not found.")
        return v

    def build_configuration(self) -> PropagatorConfigurationLegacy:
        """Merge CLI config and JSON config into one validated object.
        NOTE: CLI config override JSON config in case of overlapping"""
        with open(self.config) as f:
            json_cfg = json.load(f)
        # CLI values override JSON if both are provided
        return PropagatorConfigurationLegacy(**json_cfg,
                                             **self.model_dump())
