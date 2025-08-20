from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Literal
from pathlib import Path
import json
from pydantic import (ValidationError, ConfigDict, Field,
                      field_validator, model_validator)
from pydantic_cli import Cmd

# ---- project utils ----------------------------------------------------------
from propagator.functions import get_p_moist_fn, get_p_time_fn

from propagator_cli.boundary_conditions import BoundaryConditionsInput
from propagator_cli.geometry import _coerce_geometry_list, Geometry
from propagator_cli.console import (
    ok, warn, print_validation_errors, print_config_summary,
    update_export_destination,

)


# --- CLI configuration -------------------------------------------------------
class PropagatorCLILegacy(Cmd):
    """Configuration options coming from CLI only."""
    model_config = ConfigDict(extra="allow")

    # --- configuration file ---
    config_file: Path = Field(
        None,
        description="Path to configuration file (JSON)",
        cli=('-c', '--config-file'),
    )

    # --- static inputs mode ---
    mode: Literal["tileset", "geotiff"] = Field(
        "tileset",
        description="Mode of static data load: 'tileset' for automatic, "
                    "'geotiff' for giving DEM and FUEL in input."
                    "[default: tileset]",
        cli=('-m', '--mode'),
    )
    dem_path: Optional[Path] = Field(
        None,
        description="Path to DEM file (GeoTIFF), "
                    "required in 'geotiff' mode",
        cli=('-d', '--dem'),
    )
    fuel_path: Optional[Path] = Field(
        None,
        description="Path to FUEL file (GeoTIFF), "
                    "required in 'geotiff' mode",
        cli=('-f', '--fuel'),
    )

    # --- output folder ---
    output_folder: Path = Field(
        None,
        description="Path to output folder where results will be saved",
        cli=('-o', '--output-folder')
    )

    # --- save log ---
    record: bool = Field(
        False,
        description="Export run logs",
        cli=('--record',),
    )

    # ---------- coercion ----------
    @field_validator("config_file", mode="before")
    @classmethod
    def _coerce_config_path(cls, v):
        if v is None:
            raise ValueError("Configuration file must be set")
        return Path(v).expanduser()

    @field_validator("dem_path", "fuel_path", mode="before")
    @classmethod
    def _coerce_optional_path(cls, v):
        return None if v in (None, "") else Path(v).expanduser()

    @field_validator("output_folder", mode="before")
    @classmethod
    def _coerce_output_path(cls, v):
        if v is None:
            raise ValueError("Output folder must be set")
        return Path(v).expanduser()

    # ---------- field checks ----------
    @field_validator("config_file")
    @classmethod
    def file_must_exist(cls, v: Path) -> Path:
        if not v.is_file():
            raise ValueError("Configuration file not found.")
        return v

    # ---------- cross-field checks & friendly console messages ----------
    @model_validator(mode="after")
    def _post_setup(self):
        # geotiff mode: DEM/FUEL required and must exist
        if self.mode == "geotiff":
            if not self.dem_path:
                raise ValueError("DEM path must be set in 'geotiff' mode")
            if not self.dem_path.is_file():
                raise ValueError("DEM file not found")

            if not self.fuel_path:
                raise ValueError("FUEL path must be set in 'geotiff' mode")
            if not self.fuel_path.is_file():
                raise ValueError("FUEL file not found")

        # tileset mode: politely warn if paths were passed (they’re ignored)
        elif self.mode == "tileset":
            if self.dem_path or self.fuel_path:
                warn(
                    "'tileset' mode: DEM and FUEL paths were\
                        provided but will be ignored"
                )

        # ensure output folder exists (create if needed)
        if not self.output_folder.exists():
            warn("Output folder not present, it will be created")
            self.output_folder.mkdir(parents=True, exist_ok=True)

        return self

    def run(self):
        """Run PROPAGATOR."""
        # point the atexit export to the final output folder
        update_export_destination(
            output_folder=self.output_folder,
            basename="propagator_run",
            append_timestamp=True,
            enabled=self.record,         # honor CLI --no-record
            export_html=True,
        )

        # now build the merged configuration
        try:
            cfg = PropagatorConfigurationLegacy.from_sources(self)
        except ValidationError as ve:
            print_validation_errors(ve)
            return 1

        ok("Configuration loaded successfully.")
        print_config_summary(cfg)

        # TODO: run propagator here

        # run propagator with the configuration


# --- Full configuration (CLI + JSON) -----------------------------------------
class PropagatorConfigurationLegacy(PropagatorCLILegacy):
    """Global run configuration + time series of boundary conditions."""
    # keep extra="allow" inherited

    # --- run parameters ---
    realizations: int = Field(1, ge=1,
                              description="Number of realizations")
    init_date: datetime = Field(default_factory=datetime.now,
                                description="Datetime of the simulated event")
    time_resolution: int = Field(60, gt=0,
                                 description="Simulation resolution [minute]")
    time_limit: int = Field(1440, gt=0,
                            description="Simulation limit [minute]")
    geometry_epsg: int = Field(4326, description="EPSG of geometries")

    # boundary conditions & ignitions (placeholders)
    ignitions: Optional[List[Geometry]] = None
    boundary_conditions: List[BoundaryConditionsInput] = \
        Field(default_factory=list,
              description="List of boundary conditions")

    # other settings
    do_spotting: bool = Field(False, description="Spotting option")
    ros_model: str = Field("default", description="ROS model name")
    prob_moist_model: str = Field("default", description="Moisture model name")

    # computed (not serialized)
    p_time_fn: Optional[object] = Field(default=None, exclude=True)
    p_moist_fn: Optional[object] = Field(default=None, exclude=True)

    # --- custom loader ---
    @classmethod
    def from_sources(cls, cli: PropagatorCLILegacy
                     ) -> "PropagatorConfigurationLegacy":
        """Merge CLI config and JSON config into one validated object.
        NOTE: CLI config override JSON config in case of overlapping"""
        with open(cli.config_file) as f:
            json_cfg = json.load(f)
        # CLI values override JSON if both are provided
        return cls(**json_cfg, **cli.model_dump())

    @field_validator("init_date", mode="before")
    @classmethod
    def _parse_init_date(cls, v):
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
    def _coerce_top_ignitions(cls, v):
        return _coerce_geometry_list(v, allowed={"point", "line", "polygon"},
                                     field_name="ignitions")

    @model_validator(mode="after")
    def _post_setup(self):

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
