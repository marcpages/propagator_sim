import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from propagator_io.configuration import PropagatorConfigurationLegacy

from .console import info_msg, ok_msg, setup_console


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



def main():
    simulation_time = datetime.now()

    info_msg("Initializing CLI...")
    cli = PropagatorCLILegacy() # type: ignore
    ok_msg("CLI initialized")

    if cli.record:
        basename = f"propagator_run_{simulation_time.strftime('%Y%m%d_%H%M%S')}"
        setup_console(
            record_path=cli.output,
            basename=basename
        )
    else:
        setup_console()
    ok_msg("Console initialized")

    info_msg("Loading configuration from JSON file...")
    cfg = cli.build_configuration()
    ok_msg("Configuration loaded")
    print(cfg.model_dump_json(indent=2))


# %%
if __name__ == "__main__":
    main()
