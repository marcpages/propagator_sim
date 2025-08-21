# %%
from datetime import datetime
from propagator_cli.cli import PropagatorCLILegacy
from propagator_cli.console import (
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
