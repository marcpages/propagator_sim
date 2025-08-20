from __future__ import annotations
import atexit
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
from pydantic import ValidationError

from rich.console import Console
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.traceback import install as rich_traceback_install

# Pretty tracebacks once for the whole app
rich_traceback_install(show_locals=False)


# ---- Console singleton ----
_console: Optional[Console] = None


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console()  # not recording by default; we flip it on in main
    return _console


# ---- Export config stored globally, used by atexit handler ----
@dataclass
class _ExportConf:
    enabled: bool = True
    output_folder: Path = Path(".")
    basename: str = "propagator_run"
    append_timestamp: bool = True
    export_html: bool = True


_export_conf: _ExportConf = _ExportConf()
_export_registered: bool = False


def _export_once() -> None:
    """
    Called at process exit by atexit; writes whatever is recorded.
    If recording was never turned on, nothing is exported.
    """
    c = get_console()
    if not _export_conf.enabled or not c.record:
        return

    # build filename
    base = _export_conf.basename
    if _export_conf.append_timestamp:
        base = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir = _export_conf.output_folder
    outdir.mkdir(parents=True, exist_ok=True)

    if _export_conf.export_html:
        (outdir / f"{base}.html").write_text(c.export_html(inline_styles=True),
                                             encoding="utf-8")


def enable_bootstrap_recording() -> None:
    """Turn on recording right now (early, before CLI is parsed)."""
    get_console().record = True


def enable_bootstrap_export(
    *,
    output_folder: str | Path = ".",
    basename: str = "propagator_boot",
    append_timestamp: bool = True,
    export_html: bool = True,
    enabled: bool = True,
) -> None:
    """
    Register a single atexit exporter.
    You can retarget it later (once CLI is parsed).
    """
    global _export_registered
    _export_conf.enabled = enabled
    _export_conf.output_folder = Path(output_folder)
    _export_conf.basename = basename
    _export_conf.append_timestamp = append_timestamp
    _export_conf.export_html = export_html

    if not _export_registered:
        atexit.register(_export_once)
        _export_registered = True


def update_export_destination(
    *,
    output_folder: Optional[str | Path] = None,
    basename: Optional[str] = None,
    append_timestamp: Optional[bool] = None,
    export_html: Optional[bool] = None,
    enabled: Optional[bool] = None,
) -> None:
    """
    Call this as soon as CLI parsing succeeds to redirect the final export
    to the *real* output folder.
    """
    if output_folder is not None:
        _export_conf.output_folder = Path(output_folder)
    if basename is not None:
        _export_conf.basename = basename
    if append_timestamp is not None:
        _export_conf.append_timestamp = append_timestamp
    if export_html is not None:
        _export_conf.export_html = export_html
    if enabled is not None:
        _export_conf.enabled = enabled


# --- MESSAGES ---
def info(message: str) -> None:
    get_console().print(Text(message, style="black"))


def ok(message: str) -> None:
    get_console().print(Text(message, style="bold green"))


def warn(message: str) -> None:
    get_console().print(Text(message, style="bold yellow"))


def die(message: str, code: int = 2) -> None:
    get_console().print(Panel.fit(Text(message, style="bold red"),
                                  border_style="red"))
    raise SystemExit(code)


def print_validation_errors(ve: ValidationError) -> None:
    table = Table(title="Validation errors", box=box.SIMPLE_HEAVY)
    table.add_column("Location", style="yellow", no_wrap=True)
    table.add_column("Message", style="red")
    table.add_column("Type", style="magenta", no_wrap=True)
    for err in ve.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        table.add_row(loc or "-", err.get("msg", ""), err.get("type", ""))
    get_console().print(table)


def print_config_summary(cfg) -> None:
    """Pretty summary for PropagatorConfiguration"""
    c = get_console()

    top = Table(box=box.MINIMAL_DOUBLE_HEAD, title="PropagatorConfiguration")
    top.add_column("Field", style="cyan", no_wrap=True)
    top.add_column("Value", style="white")
    top.add_row("mode", cfg.mode)
    top.add_row("output_folder", str(cfg.output_folder))
    if cfg.mode == "geotiff":
        top.add_row("dem_path", str(cfg.dem_path))
        top.add_row("fuel_path", str(cfg.fuel_path))
    top.add_row("geometry_epsg", str(cfg.geometry_epsg))
    top.add_row("realizations", str(cfg.realizations))
    top.add_row("time_resolution [min]", str(cfg.time_resolution))
    top.add_row("time_limit [min]", str(cfg.time_limit))
    top.add_row("ros_model", cfg.ros_model)
    top.add_row("prob_moist_model", cfg.prob_moist_model)
    c.print(top)

    bc = Table(title="Boundary conditions", box=box.SIMPLE)
    bc.add_column("t [min]", justify="right", style="cyan", no_wrap=True)
    bc.add_column("w_dir [rad]", justify="right")
    bc.add_column("w_speed [km/h]", justify="right")
    bc.add_column("moisture [%]", justify="right")
    bc.add_column("ignitions", justify="left")
    for step in cfg.boundary_conditions:
        kinds = ",".join(g.kind for g in (step.ignitions or [])) or "-"
        bc.add_row(
            str(step.time),
            f"{step.w_dir:.5f}",
            f"{step.w_speed:.2f}",
            f"{step.moisture:.2f}",
            kinds,
        )
    c.print(bc)
