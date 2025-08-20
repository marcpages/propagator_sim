# %%
from pydantic_cli import run_and_exit
from propagator_cli.cli import PropagatorCLI
from propagator_cli.console import (enable_bootstrap_recording,
                                    enable_bootstrap_export)


# %%
def main():
    # record from process start + set a temporary destination
    enable_bootstrap_recording()
    enable_bootstrap_export(output_folder=".",
                            basename="propagator_boot",
                            append_timestamp=True)

    run_and_exit(PropagatorCLI)


# %%
if __name__ == "__main__":
    main()
