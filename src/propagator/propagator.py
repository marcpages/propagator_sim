"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from propagator.constants import (
    FUEL_SYSTEM_LEGACY,
)
from propagator.functions import (
    get_p_moist_fn,
    get_p_time_fn,
    next_updates_fn,
)
from propagator.models import (
    BoundaryConditions,
    FuelSystem,
    PMoistFn,
    PropagatorOutput,
    PropagatorStats,
    PTimeFn,
    UpdateBatch,
    UpdateBatchWithTime,
)
from propagator.scheduler import Scheduler


@dataclass
class Propagator:
    """Stochastic cellular wildfire spread simulator.

    PROPAGATOR evolves a binary fire state over a regular grid for a
    configurable number of realizations. Spread depends on vegetation,
    topography and environmental drivers (wind, moisture) through
    pluggable probability and travel-time models.
    """

    # domain parameters for the simulation

    # input
    veg: npt.NDArray[np.integer]
    dem: npt.NDArray[np.floating]
    realizations: int
    do_spotting: bool

    # set fuels
    fuels: FuelSystem = field(default_factory=lambda: FUEL_SYSTEM_LEGACY)

    # selected simulation functions
    p_time_fn: PTimeFn = field(default=get_p_time_fn("default"))
    p_moist_fn: PMoistFn = field(default=get_p_moist_fn("default"))

    # scheduler object
    scheduler: Scheduler = field(init=False)

    # simulation state
    time: int = field(init=False, default=0)
    fire: npt.NDArray[np.int8] = field(init=False)
    ros: npt.NDArray[np.float32] = field(init=False)
    fireline_int: npt.NDArray[np.float32] = field(init=False)
    moisture: npt.NDArray[np.floating] = field(init=False)
    wind_dir: npt.NDArray[np.floating] = field(init=False)
    wind_speed: npt.NDArray[np.floating] = field(init=False)
    actions_moisture: npt.NDArray[np.floating] | None = field(
        default=None, init=False
    )  # additional moisture due to fighting actions
    # (ideally it should decay over time)

    def __post_init__(self):
        """Allocate internal state arrays based
        on the vegetation grid shape."""
        shape = self.veg.shape
        self.scheduler = Scheduler(realizations=self.realizations)
        self.fire = np.zeros(shape + (self.realizations,), dtype=np.int8)
        self.ros = np.zeros(shape + (self.realizations,), dtype=np.float32)
        self.fireline_int = np.zeros(
            shape + (self.realizations,), dtype=np.float32
        )

    # def set_ignitions(self, ignitions: Ignitions) -> None:
    #     """
    #     Apply ignitions to the state of the simulation.
    #     """
    #     self.scheduler.push_ignitions(ignitions)
    #     for p in ignitions.coords:
    #         self.fire[p[0], p[1], p[2]] = 0

    def compute_fire_probability(self) -> npt.NDArray[np.floating]:
        """Return mean burn probability across realizations for each cell.

        Returns
        -------
        numpy.ndarray
            2D array with values in [0, 1].
        """
        values = np.mean(self.fire, axis=2).astype(np.float32)
        return values

    def compute_ros_max(self) -> npt.NDArray[np.floating]:
        """Return per-cell maximum Rate of Spread across realizations.

        Returns
        -------
        numpy.ndarray
            2D array with max RoS per cell.
        """
        RoS_max = np.max(self.ros, axis=2).astype(np.float32)
        return RoS_max

    def compute_ros_mean(self) -> npt.NDArray[np.floating]:
        """Return per-cell mean Rate of Spread, ignoring zeros as no-spread.

        Returns
        -------
        numpy.ndarray
            2D array with mean RoS per cell.
        """
        RoS_m = np.where(self.ros > 0, self.ros, np.nan)
        RoS_mean = np.nanmean(RoS_m, axis=2).astype(np.float32)
        RoS_mean = np.where(RoS_mean > 0, RoS_mean, 0)
        return RoS_mean

    def compute_fireline_int_max(self) -> npt.NDArray[np.floating]:
        """Return per-cell maximum fireline intensity across realizations.

        Returns
        -------
        numpy.ndarray
            2D array of max intensity values.
        """
        fl_I_max = np.nanmax(self.fireline_int, axis=2).astype(np.float32)
        return fl_I_max

    def compute_fireline_int_mean(self) -> npt.NDArray[np.floating]:
        """Return per-cell mean fireline intensity,
        ignoring zeros as no-spread.

        Returns
        -------
        numpy.ndarray
            2D array of mean intensity values.
        """
        fl_I_m = np.where(self.fireline_int > 0, self.fireline_int, np.nan)
        fl_I_mean = np.nanmean(fl_I_m, axis=2).astype(np.float32)
        fl_I_mean = np.where(fl_I_mean > 0, fl_I_mean, 0)
        return fl_I_mean

    def compute_stats(
        self, values: npt.NDArray[np.floating]
    ) -> PropagatorStats:
        """Compute simple area-based stats and number of active fronts.

        Parameters
        ----------
        values : numpy.ndarray
            Fire probability map in [0, 1].

        Returns
        -------
        PropagatorStats
            Dataclass with counters and area summaries.
        """
        n_active = len(self.scheduler.active().tolist())
        # cell_area = float() * float(self.step_y) / 10000.0
        cell_area = 1
        area_mean = float(np.sum(values) * cell_area)
        area_50 = float(np.sum(values >= 0.5) * cell_area)
        area_75 = float(np.sum(values >= 0.75) * cell_area)
        area_90 = float(np.sum(values >= 0.90) * cell_area)

        return PropagatorStats(
            n_active=n_active,
            area_mean=area_mean,
            area_50=area_50,
            area_75=area_75,
            area_90=area_90,
        )

    def set_boundary_conditions(
        self, boundary_condition: BoundaryConditions
    ) -> None:
        """Externally apply boundary conditions at desired time.

        Parameters
        ----------
        boundary_condition : PropagatorBoundaryConditions
            Conditions to apply.
        """
        if int(self.time) > boundary_condition.time:
            raise ValueError(
                "Boundary conditions cannot be applied in the past.\
                Please check the time of the boundary conditions."
            )
        self.scheduler.add_boundary_conditions(boundary_condition)

    # def compute_spotting(
    #     self,
    #     veg_type: npt.NDArray[np.integer],
    #     update: npt.NDArray[np.integer],
    # ) -> tuple[
    #     npt.NDArray[np.integer],
    #     npt.NDArray[np.integer],
    #     npt.NDArray[np.integer],
    #     npt.NDArray[np.floating],
    # ]:
    #     """Compute ember landing cells and their
    # transition times (if enabled).

    #     Parameters
    #     ----------
    #     veg_type : numpy.ndarray
    #         Vegetation type at updated cells.
    #     update : numpy.ndarray
    #         Updates as stacked rows [r, c, t].

    #     Returns
    #     -------
    #     tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
    #         Row indices, col indices, realization indices, and transition
    #         times for spotted ignitions.
    #     """
    #     moisture = self.get_moisture()
    #     # only cells that have veg as emitters for spotting are considered
    #     spotting_mask = np.isin(veg_type, list(self.fuels.which_spotting()))
    #     spotting_r, spotting_c, spotting_t = (
    #         update[spotting_mask, 0],
    #         update[spotting_mask, 1],
    #         update[spotting_mask, 2],
    #     )
    #     # calculate number of embers per emitter > Poisson distribution
    #     num_embers = RNG.poisson(LAMBDA_SPOTTING, size=spotting_r.shape)
    #     # create list of source points for each ember
    #     spotting_arr_r = spotting_r.repeat(repeats=num_embers)
    #     spotting_arr_c = spotting_c.repeat(repeats=num_embers)
    #     spotting_arr_t = spotting_t.repeat(repeats=num_embers)
    #     # calculate angle > uniform distribution
    #     ember_angle = RNG.uniform(0, 2.0 * np.pi, size=spotting_arr_r.shape)
    #     # calculate distance > depends on wind speed and direction
    #     # NOTE: it is computed considering wind speed and direction
    #     # of the cell of origin of the ember
    #     ember_distance = fire_spotting(
    #         ember_angle,
    #         self.wind_dir[spotting_arr_r, spotting_arr_c],
    #         self.wind_speed[spotting_arr_r, spotting_arr_c],
    #     )
    #     # filter out short embers
    #     idx_long_embers = ember_distance > 2 * CELLSIZE
    #     spotting_arr_r = spotting_arr_r[idx_long_embers]
    #     spotting_arr_c = spotting_arr_c[idx_long_embers]
    #     spotting_arr_t = spotting_arr_t[idx_long_embers]
    #     ember_angle = ember_angle[idx_long_embers]
    #     ember_distance = ember_distance[idx_long_embers]
    #     # calculate landing locations
    #     # vertical delta [meters]
    #     delta_r = ember_distance * np.cos(ember_angle)
    #     # horizontal delta [meters]
    #     delta_c = ember_distance * np.sin(ember_angle)
    #     nb_spot_r = delta_r / CELLSIZE  # number of vertical cells
    #     nb_spot_r = nb_spot_r.astype(int)
    #     nb_spot_c = delta_c / CELLSIZE  # number of horizontal cells
    #     nb_spot_c = nb_spot_c.astype(int)
    #     # vertical location of the cell to be ignited by the ember
    #     nr_spot = spotting_arr_r + nb_spot_r
    #     # horizontal location of the cell to be ignited by the ember
    #     nc_spot = spotting_arr_c + nb_spot_c
    #     nt_spot = spotting_arr_t
    #     # if I surpass the bounds, I stick to them.
    #     # This way I don't have to reshape anything.
    #     shape = self.fire.shape
    #     nr_spot[nr_spot > shape[0] - 1] = shape[0] - 1
    #     nc_spot[nc_spot > shape[1] - 1] = shape[1] - 1
    #     nr_spot[nr_spot < 0] = 0
    #     nc_spot[nc_spot < 0] = 0
    #     # we want to put another probabilistic filter in order
    #     # to assess the success of ember ignition.
    #     # Formula (10) of Alexandridis et al IJWLF 2011
    #     # P_c = P_c0 (1 + P_cd), where P_c0 constant probability of ignition
    #     # by spotting and P_cd is a correction factor that
    #     # depends on vegetation type and density > set on the fuels system
    #     fuels_to = self.fuels.get_fuels(self.veg[nr_spot, nc_spot])
    #     P_c = P_C0 * (1 + np.array([f.prob_ign_by_embers for f in fuels_to]))
    #     success_spot_mask = RNG.uniform(size=P_c.shape) < P_c
    #     nr_spot = nr_spot[success_spot_mask]
    #     nc_spot = nc_spot[success_spot_mask]
    #     nt_spot = nt_spot[success_spot_mask]
    #     # the following function evalates the time that the embers
    #     # will need to burn the entire cell they land into
    #     fuels_to = self.fuels.get_fuels(self.veg[nr_spot, nc_spot])
    #     v0 = np.array([f.v0 for f in fuels_to]) / 60  # ros [m/min]
    #     transition_time_spot, _ros_spot = self.p_time_fn(
    #         v0,
    #         # dh=0 (no slope) to simplify the phenomenon
    #         self.dem[nr_spot, nc_spot],
    #         self.dem[nr_spot, nc_spot],
    #         np.zeros(nr_spot.shape),  # angle to
    #         CELLSIZE * np.ones(nr_spot.shape),
    #         moisture[nr_spot, nc_spot],
    #         # wind in the cell where the ember lands
    #         self.wind_dir[nr_spot, nc_spot],
    #         self.wind_speed[nr_spot, nc_spot],
    #     )
    #     return nr_spot, nc_spot, nt_spot, transition_time_spot

    def apply_updates(
        self,
        updates: UpdateBatch,
    ):
        """Apply a batch of burning updates and schedule new ones.
        Parameters
        ----------
        updates : list[numpy.ndarray] | numpy.ndarray
            Coordinates to activate as burning, each as
            [row, col, realization].
        Returns
        -------
        list[tuple[float, numpy.ndarray]]
            Pairs of (time, array[n, 3]) for future updates to schedule.
        """
        moisture = self.get_moisture()

        must_be_updated = self.fire[updates.rows, updates.cols, updates.realizations] == 0
        rows = updates.rows[must_be_updated]
        cols = updates.cols[must_be_updated]
        realizations = updates.realizations[must_be_updated]
        ros = updates.rates_of_spread[must_be_updated]
        fireline_intensity = updates.fireline_intensities[must_be_updated]

        self.fire[rows, cols, realizations] = 1
        self.ros[rows, cols, realizations] = ros
        self.fireline_int[rows, cols, realizations] = fireline_intensity

        new_updates_tuple = next_updates_fn(
            rows,
            cols,
            realizations,
            self.time,
            self.veg,
            self.dem,
            self.fire,
            moisture,
            self.wind_dir,
            self.wind_speed,
            self.fuels,
        )

        next_updates = UpdateBatchWithTime.from_tuple(new_updates_tuple)
        self.scheduler.push_updates(next_updates)

    def decay_actions_moisture(
        self, time_delta: int, decay_factor: float = 0.01
    ) -> None:
        """
        Decay the actions moisture over time.

        Args:
            time_delta (int): Elapsed simulation time since last step.
            decay_factor (float): Per-unit-time fractional decay in [0, 1].
        """
        if self.actions_moisture is None:
            return
        k = np.clip(decay_factor, 0, 1)
        self.actions_moisture *= (1 - k) ** max(time_delta, 0)

    def get_moisture(self) -> npt.NDArray[np.floating]:
        """
        Get the fuel moisture at the current time step.

        Returns:
            np.ndarray: Base moisture plus action-derived increments,
            clipped to [0, 100].
        """
        if self.actions_moisture is None:
            return self.moisture

        moisture = self.moisture + self.actions_moisture
        moisture = np.clip(moisture, 0, 100)

        return moisture

    def step(
        self,
    ) -> None:
        """Advance the simulation to the next scheduled
        time and update state."""
        time, scheduler_event = self.scheduler.pop()

        time_delta = time - self.time
        self.time = time
        self.decay_actions_moisture(time_delta)

        if scheduler_event.moisture is not None:
            self.moisture = scheduler_event.moisture

        if scheduler_event.additional_moisture is not None:
            if self.actions_moisture is None:
                self.actions_moisture = np.zeros_like(self.moisture)
            self.actions_moisture += scheduler_event.additional_moisture
            self.actions_moisture = np.clip(self.actions_moisture, 0, 100)

        if scheduler_event.wind_dir is not None:
            self.wind_dir = scheduler_event.wind_dir

        if scheduler_event.wind_speed is not None:
            self.wind_speed = scheduler_event.wind_speed

        if scheduler_event.vegetation_changes is not None:
            # mutate vegetation where needed
            mask = ~np.isnan(scheduler_event.vegetation_changes)
            self.veg[mask] = scheduler_event.vegetation_changes[mask]

        if scheduler_event.updates is not None:
            self.apply_updates(scheduler_event.updates)

    def get_output(self) -> PropagatorOutput:
        """Assemble the current outputs and summary stats into a dataclass.

        Returns:
            PropagatorOutput: Snapshot of fire probability,
            RoS, intensity, stats.
        """
        fire_probability = self.compute_fire_probability()
        ros_max = self.compute_ros_max()
        ros_mean = self.compute_ros_mean()
        fireline_intensity_max = self.compute_fireline_int_max()
        fireline_intensity_mean = self.compute_fireline_int_mean()
        stats = self.compute_stats(fire_probability)

        return PropagatorOutput(
            time=self.time,
            fire_probability=fire_probability,
            ros_mean=ros_mean,
            ros_max=ros_max,
            fli_mean=fireline_intensity_mean,
            fli_max=fireline_intensity_max,
            stats=stats,
        )

    def next_time(self) -> int | None:
        """
        Get the next time step.

        Returns:
            int | None: 0 at initialization; None if no more events; otherwise
            the next scheduled simulation time.
        """
        if len(self.scheduler) == 0:
            return None

        return self.scheduler.next_time()
