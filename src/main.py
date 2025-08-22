# %%
from datetime import datetime
try:
    from propagator_cli.cli import PropagatorCLILegacy
    from propagator_cli.console import (
        setup_console,
        info_msg, ok_msg, warn_msg, error_msg
    )
except ModuleNotFoundError:
    # Support running without installation by adding src to sys.path
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from propagator_cli.cli import PropagatorCLILegacy  # type: ignore
    from propagator_cli.console import (  # type: ignore
        setup_console,
        info_msg, ok_msg, warn_msg, error_msg
    )


# %%
def main():
    simulation_time = datetime.now()

    info_msg("Initializing CLI...")
    cli = PropagatorCLILegacy()
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
