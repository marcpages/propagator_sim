# %%
from datetime import datetime
from propagator_cli.cli import PropagatorCLILegacy
from propagator_cli.console import (
    setup_console,
    info_msg, ok_msg, warn_msg, error_msg
)
import numpy as np
from propagator_io.loader.geotiff import PropagatorDataFromGeotiffs
from propagator.propagator import (
    Propagator,
    PropagatorActions,
    PropagatorBoundaryConditions,
)



# %%
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

    v0 = np.loadtxt("v0_table.txt")
    prob_table = np.loadtxt("prob_table.txt")
    p_veg = np.loadtxt("p_vegetation.txt")

    # loader geographic information
    loader = PropagatorDataFromGeotiffs(
        dem_file=cfg.dem,
        veg_file=cfg.fuel,
    )

    # Load the data
    dem = loader.get_dem()
    veg = loader.get_veg()
    geo_info = loader.get_geo_info()

    simulator = Propagator(
        dem=dem,
        veg=veg,
        realizations=cfg.realizations,
        ros_0=v0,
        probability_table=prob_table,
        veg_parameters=p_veg,
        do_spotting=cfg.do_spotting,
        p_time_fn=cfg.p_time_fn,
        p_moist_fn=cfg.p_moist_fn,
    )

    boundary_conditions_list = cfg.get_propagator_bcs(geo_info)
    bc_0 = boundary_conditions_list[0]
    print(bc_0.ignitions.sum())
    # actions_list = cfg.get_propagator_actions(geo_info)

    while True:
        next_time = simulator.next_time()
        if next_time is None:
            break

        info_msg(f"Supposed Next time: {next_time}")

        if len(boundary_conditions_list) > 0:
            boundary_conditions = boundary_conditions_list[0]
            if boundary_conditions.time <= next_time:
                simulator.set_boundary_conditions(boundary_conditions)
                boundary_conditions_list.pop(0)

        # if len(actions_list) > 0:
        #     actions = actions_list[0]
        #     if actions.time <= next_time:
        #         simulator.apply_actions(actions)
        #         actions_list.pop(0)

        info_msg(f"Current time: {simulator.time}")
        simulator.step()
        info_msg(f"New time: {simulator.time}")

        if simulator.time % cfg.time_resolution == 0:
            output = simulator.get_output()
            # Save the output to the specified folder
            ...

        if simulator.time > cfg.time_limit:
            break


# %%
if __name__ == "__main__":
    main()
