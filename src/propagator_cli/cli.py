import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from propagator_io.configuration import PropagatorConfigurationLegacy
from propagator_io.loader.geotiff import PropagatorDataFromGeotiffs
from propagator.propagator import Propagator
import numpy as np

from propagator_cli.console import info_msg, ok_msg, setup_console


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
    # pydantic-settings is taking care of it
    cli = PropagatorCLILegacy()  # type: ignore
    ok_msg("CLI initialized")
    print(cli.model_dump())

    if cli.record:
        basename = f"propagator_run_{
            simulation_time.strftime('%Y%m%d_%H%M%S')}"
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

    v0 = np.loadtxt("example/v0_table.txt")
    prob_table = np.loadtxt("example/prob_table.txt")
    p_veg = np.loadtxt("example/p_vegetation.txt")

    if cfg.dem is None or cfg.fuel is None:
        raise ValueError("DEM and FUEL files must be provided in 'geotiff' mode")
    
    # loader geographic information
    loader = PropagatorDataFromGeotiffs(
        dem_file=str(cfg.dem),
        veg_file=str(cfg.fuel),
    )

    # Load the data
    dem = loader.get_dem()
    veg = loader.get_veg()
    geo_info = loader.get_geo_info()

    args = dict()
    if cfg.p_time_fn is not None:
        args.update(dict(p_time_fn=cfg.p_time_fn))
    if cfg.p_moist_fn is not None:
        args.update(dict(p_moist_fn=cfg.p_moist_fn))

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=cfg.realizations,
        ros_0=v0,
        probability_table=prob_table,
        veg_parameters=p_veg,
        do_spotting=cfg.do_spotting,
        **args
    )

    boundary_conditions_list = cfg.get_boundary_conditions(geo_info)
    for boundary_condition in boundary_conditions_list:
        simulator.set_boundary_conditions(boundary_condition)

    while True:
        next_time = simulator.next_time()
        if next_time is None:
            break

        info_msg(f"Current time: {simulator.time}")
        simulator.step()
        info_msg(f"New time: {simulator.time}")

        if simulator.time % cfg.time_resolution == 0:
            _output = simulator.get_output()
            # Save the output to the specified folder
            ...

        if simulator.time > cfg.time_limit:
            break

# %%
if __name__ == "__main__":
    main()
