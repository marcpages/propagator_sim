"""Core wildfire propagation engine.

This module defines the main simulation primitives and the `Propagator` class
that evolves a fire state over a grid using wind, slope, vegetation, and
moisture inputs. Public dataclasses capture boundary conditions, actions,
summary statistics, and output snapshots suitable for CLI and IO layers.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from numpy import tile

from propagator.constants import (
    CELLSIZE,
    LAMBDA_SPOTTING,
    NEIGHBOURS_ANGLE,
    NEIGHBOURS_ARRAY,
    NEIGHBOURS_DISTANCE,
    P_C0,
    FUEL_SYSTEM_LEGACY
)
from propagator.functions import (
    fire_spotting,
    fireline_intensity,
    get_p_moist_fn,
    get_p_time_fn,
    lhv_canopy,
    lhv_dead_fuel,
    w_h_effect_on_probability,
)
from propagator.models import (
    FuelSystem,
    BoundaryConditions,
    Ignitions,
    PMoistFn,
    PropagatorOutput,
    PropagatorStats,
    PTimeFn,
    UpdateBatch,
)
from propagator.scheduler import Scheduler

RNG = np.random.default_rng(12345)


@dataclass
class Propagator:
    """Stochastic cellular wildfire spread simulator.

    PROPAGATOR evolves a binary fire state over a regular grid for a
    configurable number of realizations. Spread depends on vegetation, topography
    and environmental drivers (wind, moisture) through pluggable probability and
    travel-time models.
    """

    # domain parameters for the simulation

    # input
    veg: npt.NDArray[np.integer]
    dem: npt.NDArray[np.floating]
    realizations: int
    do_spotting: bool

    # set fuels
    fuels: FuelSystem = field(default_factory=FUEL_SYSTEM_LEGACY)

    # selected simulation functions
    p_time_fn: PTimeFn = field(default=get_p_time_fn("default"))
    p_moist_fn: PMoistFn = field(default=get_p_moist_fn("default"))

    # scheduler object
    scheduler: Scheduler = field(init=False)

    # simulation state
    time: int = field(init=False, default=0)
    fire: npt.NDArray[np.int8] = field(init=False)
    ros: npt.NDArray[np.float16] = field(init=False)
    fireline_int: npt.NDArray[np.float16] = field(init=False)
    moisture: npt.NDArray[np.floating] = field(init=False)
    wind_dir: npt.NDArray[np.floating] = field(init=False)
    wind_speed: npt.NDArray[np.floating] = field(init=False)
    actions_moisture: npt.NDArray[np.floating] | None = field(
        init=False
    )  # additional moisture due to fighting actions (ideally it should decay over time)

    def __post_init__(self):
        """Allocate internal state arrays based on the vegetation grid shape."""
        shape = self.veg.shape
        self.scheduler = Scheduler(realizations=self.realizations)
        self.fire = np.zeros(shape + (self.realizations,), dtype=np.int8)
        self.ros = np.zeros(shape + (self.realizations,), dtype=np.float16)
        self.fireline_int = np.zeros(shape + (self.realizations,), dtype=np.float16)
        self.actions_moisture = np.zeros(shape, dtype=np.float16)
        # check if unique values in veg (apart of 0) are in fuels keys
        veg_types = np.unique(self.veg)
        for vt in veg_types:
            if vt != 0 and vt not in self.fuels.get_keys():
                raise ValueError(
                    f"vegetation type {vt} found in veg raster is not present in fuels keys {self.fuels.get_keys()}"
                )

    def set_ignitions(self, ignitions: Ignitions) -> None:
        """
        Apply ignitions to the state of the simulation.
        """
        self.scheduler.push_ignitions(ignitions)
        for p in ignitions.coords:
            self.fire[p[0], p[1], p[2]] = 0

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
        """Return per-cell mean fireline intensity, ignoring zeros as no-spread.

        Returns
        -------
        numpy.ndarray
            2D array of mean intensity values.
        """
        fl_I_m = np.where(self.fireline_int > 0, self.fireline_int, np.nan)
        fl_I_mean = np.nanmean(fl_I_m, axis=2).astype(np.float32)
        fl_I_mean = np.where(fl_I_mean > 0, fl_I_mean, 0)
        return fl_I_mean

    def compute_stats(self, values: npt.NDArray[np.floating]) -> PropagatorStats:
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

    def propagation_probability(
        self,
        dem_from: npt.NDArray[np.floating],
        dem_to: npt.NDArray[np.floating],
        veg_from: npt.NDArray[np.integer],
        veg_to: npt.NDArray[np.integer],
        angle_to: npt.NDArray[np.floating],
        dist_to: npt.NDArray[np.floating],
        moist: npt.NDArray[np.floating],
        w_dir: npt.NDArray[np.floating],
        w_speed: npt.NDArray[np.floating],
    ) -> npt.NDArray[np.floating]:
        """Compute spread probability from one cell to a neighbor.

        Combines vegetation-to-vegetation base probability with wind/slope
        modulation and moisture attenuation. The output is in [0, 1].

        Parameters
        ----------
        dem_from, dem_to : numpy.ndarray
            Elevation at source and neighbor cells.
        veg_from, veg_to : numpy.ndarray
            Vegetation types (1-based) at source and neighbor cells.
        angle_to : numpy.ndarray
            Direction to neighbor (radians).
        dist_to : numpy.ndarray
            Lattice distance to neighbor (cells).
        moist : numpy.ndarray
            Moisture values (%).
        w_dir : numpy.ndarray
            Wind direction (radians).
        w_speed : numpy.ndarray
            Wind speed (km/h).

        Returns
        -------
        numpy.ndarray
            Probability per neighbor in [0, 1].
        """
        dh = dem_to - dem_from
        alpha_wh = w_h_effect_on_probability(angle_to, w_speed, w_dir, dh, dist_to)
        alpha_wh = np.maximum(alpha_wh, 0)  # prevent alpha < 0

        p_moist = self.p_moist_fn(moist)
        p_moist = np.clip(p_moist, 0, 1.0)
        p_veg = self.fuels.get_transition_probability(veg_from, veg_to)
        probability = 1 - (1 - p_veg) ** alpha_wh
        probability = np.clip(probability * p_moist, 0, 1.0)

        return probability

    def set_boundary_conditions(self, boundary_condition: BoundaryConditions) -> None:
        """Externally apply boundary conditions at desired time.

        Parameters
        ----------
        boundary_condition : PropagatorBoundaryConditions
            Conditions to apply.
        """
        if self.time > boundary_condition.time:
            raise ValueError(
                "Boundary conditions cannot be applied in the past. Please check the time of the boundary conditions."
            )
        self.scheduler.add_boundary_conditions(boundary_condition)

    def compute_spotting(
        self,
        veg_type: npt.NDArray[np.integer],
        update: npt.NDArray[np.integer],
    ) -> tuple[
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
        npt.NDArray[np.integer],
        npt.NDArray[np.floating],
    ]:
        """Compute ember landing cells and their transition times (if enabled).

        Parameters
        ----------
        veg_type : numpy.ndarray
            Vegetation type at updated cells.
        update : numpy.ndarray
            Updates as stacked rows [r, c, t].

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            Row indices, col indices, realization indices, and transition
            times for spotted ignitions.
        """
        moisture = self.get_moisture()
        # only cells that have veg as emitters for spotting are considered
        spotting_mask = np.isin(veg_type, list(self.fuels.which_spotting()))
        spotting_r, spotting_c, spotting_t = (
            update[spotting_mask, 0],
            update[spotting_mask, 1],
            update[spotting_mask, 2],
        )
        # calculate number of embers per emitter > Poisson distribution
        num_embers = RNG.poisson(LAMBDA_SPOTTING, size=spotting_r.shape)
        # create list of source points for each ember
        spotting_arr_r = spotting_r.repeat(repeats=num_embers)
        spotting_arr_c = spotting_c.repeat(repeats=num_embers)
        spotting_arr_t = spotting_t.repeat(repeats=num_embers)
        # calculate angle > uniform distribution
        ember_angle = RNG.uniform(0, 2.0 * np.pi, size=spotting_arr_r.shape)
        # calculate distance > depends on wind speed and direction
        # NOTE: it is computed considering wind speed and direction
        # of the cell of origin of the ember
        ember_distance = fire_spotting(
            ember_angle,
            self.wind_dir[spotting_arr_r, spotting_arr_c],
            self.wind_speed[spotting_arr_r, spotting_arr_c])
        # filter out short embers
        idx_long_embers = ember_distance > 2 * CELLSIZE
        spotting_arr_r = spotting_arr_r[idx_long_embers]
        spotting_arr_c = spotting_arr_c[idx_long_embers]
        spotting_arr_t = spotting_arr_t[idx_long_embers]
        ember_angle = ember_angle[idx_long_embers]
        ember_distance = ember_distance[idx_long_embers]
        # calculate landing locations
        # vertical delta [meters]
        delta_r = ember_distance * np.cos(ember_angle)
        # horizontal delta [meters]
        delta_c = ember_distance * np.sin(ember_angle)
        nb_spot_r = delta_r / CELLSIZE  # number of vertical cells
        nb_spot_r = nb_spot_r.astype(int)
        nb_spot_c = delta_c / CELLSIZE  # number of horizontal cells
        nb_spot_c = nb_spot_c.astype(int)
        # vertical location of the cell to be ignited by the ember
        nr_spot = spotting_arr_r + nb_spot_r
        # horizontal location of the cell to be ignited by the ember
        nc_spot = spotting_arr_c + nb_spot_c
        nt_spot = spotting_arr_t
        # if I surpass the bounds, I stick to them.
        # This way I don't have to reshape anything.
        shape = self.fire.shape
        nr_spot[nr_spot > shape[0] - 1] = shape[0] - 1
        nc_spot[nc_spot > shape[1] - 1] = shape[1] - 1
        nr_spot[nr_spot < 0] = 0
        nc_spot[nc_spot < 0] = 0
        # we want to put another probabilistic filter in order
        # to assess the success of ember ignition.
        # Formula (10) of Alexandridis et al IJWLF 2011
        # P_c = P_c0 (1 + P_cd), where P_c0 constant probability of ignition
        # by spotting and P_cd is a correction factor that
        # depends on vegetation type and density > set on the fuels system
        P_c = P_C0 * (
            1 + self.fuels.get_prob_ign_by_embers(self.veg[nr_spot, nc_spot]))
        success_spot_mask = RNG.uniform(size=P_c.shape) < P_c
        nr_spot = nr_spot[success_spot_mask]
        nc_spot = nc_spot[success_spot_mask]
        nt_spot = nt_spot[success_spot_mask]
        # the following function evalates the time that the embers
        # will need to burn the entire cell they land into
        veg_to = self.veg[nr_spot, nc_spot]
        v0 = self.fuels.get_v0(veg_to) / 60  # ros [m/min]
        transition_time_spot, _ros_spot = self.p_time_fn(
            v0,
            # dh=0 (no slope) to simplify the phenomenon
            self.dem[nr_spot, nc_spot],
            self.dem[nr_spot, nc_spot],
            np.zeros(nr_spot.shape),  # angle to
            CELLSIZE * np.ones(nr_spot.shape),
            moisture[nr_spot, nc_spot],
            # wind in the cell where the ember lands
            self.wind_dir[nr_spot, nc_spot],
            self.wind_speed[nr_spot, nc_spot],
        )
        return nr_spot, nc_spot, nt_spot, transition_time_spot

    def apply_updates(self, updates: UpdateBatch) -> list[Ignitions]:
        """Apply a batch of burning updates and schedule new ones.

        Parameters
        ----------
        updates : list[numpy.ndarray] | numpy.ndarray
            Coordinates to activate as burning, each as [row, col, realization].

        Returns
        -------
        list[tuple[float, numpy.ndarray]]
            Pairs of (time, array[n, 3]) for future updates to schedule.
        """
        moisture = self.get_moisture()

        # coordinates of the next updates
        update = np.vstack(updates)
        veg_type = self.veg[update[:, 0], update[:, 1]]
        mask = np.logical_and(
            veg_type != 0, self.fire[update[:, 0], update[:, 1], update[:, 2]] == 0
        )

        rows_from, cols_from, realization = (
            update[mask, 0],
            update[mask, 1],
            update[mask, 2],
        )
        self.fire[rows_from, cols_from, realization] = 1

        # # veg type modified due to the heavy fighting actions
        # heavy_acts = bc.get(HEAVY_ACTION_RASTER_TAG, None)
        # if heavy_acts:
        #     for heavyy in heavy_acts:
        #         # da scegliere se mettere a 0 (impossibile che propaghi) 3 (non veg, quindi prova a propagare ma non riesce) o 7(faggete, quindi propaga con bassissima probabilità)
        #         self.veg[heavyy[0], heavyy[1]] = 0

        neighbours_num = NEIGHBOURS_ARRAY.shape[0]
        from_num = rows_from.shape[0]

        neighbours_row_offsets = tile(NEIGHBOURS_ARRAY[:, 0], from_num)
        neighbours_col_offsets = tile(NEIGHBOURS_ARRAY[:, 1], from_num)

        rows_to = rows_from.repeat(neighbours_num) + neighbours_row_offsets
        cols_to = cols_from.repeat(neighbours_num) + neighbours_col_offsets
        nt = realization.repeat(neighbours_num)

        # let's apply a random noise to wind direction and speed for all the cells
        # [TODO] wind dir and speed are not scalar anymore! rewrite
        # w_dir_r = (self.wind_dir + (pi / 16) * (0.5 - RNG.random(from_num))).repeat(
        #     nb_num
        # )
        # w_speed_r = (self.wind_speed * (1.2 - 0.4 * RNG.random(from_num))).repeat(
        #     nb_num
        # )
        w_dir_r = self.wind_dir[rows_from, cols_from].repeat(neighbours_num)
        w_speed_r = self.wind_speed[rows_from, cols_from].repeat(neighbours_num)
        moisture_r = moisture[rows_to, cols_to]

        dem_from = self.dem[rows_from, cols_from].repeat(neighbours_num)
        veg_from = self.veg[rows_from, cols_from].repeat(neighbours_num)
        veg_to = self.veg[rows_to, cols_to]
        dem_to = self.dem[rows_to, cols_to]

        angle_to = NEIGHBOURS_ANGLE[
            neighbours_row_offsets + 1, neighbours_col_offsets + 1
        ]
        dist_to = NEIGHBOURS_DISTANCE[
            neighbours_row_offsets + 1, neighbours_col_offsets + 1
        ]

        # keep only pixels where fire can spread
        valid_fire_mask = np.logical_and(
            self.fire[rows_to, cols_to, nt] == 0, veg_to != 0
        )
        dem_from = dem_from[valid_fire_mask]
        veg_from = veg_from[valid_fire_mask]
        dem_to = dem_to[valid_fire_mask]
        veg_to = veg_to[valid_fire_mask]
        angle_to = angle_to[valid_fire_mask]
        dist_to = dist_to[valid_fire_mask]
        w_speed_r = w_speed_r[valid_fire_mask]
        w_dir_r = w_dir_r[valid_fire_mask]
        moisture_r = moisture_r[valid_fire_mask]
        rows_to = rows_to[valid_fire_mask]
        cols_to = cols_to[valid_fire_mask]
        nt = nt[valid_fire_mask]

        # get the probability for all the pixels
        p_prob = self.propagation_probability(
            dem_from,
            dem_to,
            veg_from,
            veg_to,
            angle_to,
            dist_to,
            moisture_r,
            w_dir_r,
            w_speed_r,
        )

        # try the propagation
        occurrences = p_prob > RNG.random(p_prob.shape[0])

        dem_from = dem_from[occurrences]
        veg_from = veg_from[occurrences]
        dem_to = dem_to[occurrences]
        veg_to = veg_to[occurrences]
        angle_to = angle_to[occurrences]
        dist_to = dist_to[occurrences]
        w_speed_r = w_speed_r[occurrences]
        w_dir_r = w_dir_r[occurrences]
        moisture_r = moisture_r[occurrences]
        rows_to = rows_to[occurrences]
        cols_to = cols_to[occurrences]
        nt = nt[occurrences]

        # get the propagation time for the propagating pixels
        # transition_time = self.p_time(dem_from[p], dem_to[p],
        v0 = self.fuels.get_v0(veg_from) / 60
        transition_time, ros = self.p_time_fn(
            v0,
            dem_from,
            dem_to,
            angle_to,
            dist_to,
            moisture_r,
            w_dir_r,
            w_speed_r,
        )
        d0 = self.fuels.get_d0(veg_to)
        d1 = self.fuels.get_d1(veg_to)
        hhv = self.fuels.get_hhv(veg_to)
        humidity = self.fuels.get_humidity(veg_to)

        # evaluate LHV of dead fuel
        lhv_dead_fuel_value = lhv_dead_fuel(hhv, moisture_r)
        # evaluate LHV of the canopy
        lhv_canopy_value = lhv_canopy(hhv, humidity)
        # evaluate fireline intensity
        fireline_intensity_value = fireline_intensity(
            d0, d1, ros, lhv_dead_fuel_value, lhv_canopy_value
        )

        self.fireline_int[rows_to, cols_to, nt] = fireline_intensity_value
        self.ros[rows_to, cols_to, nt] = ros

        if self.do_spotting:
            nr_spot, nc_spot, nt_spot, transition_time_spot = self.compute_spotting(
                veg_type, update
            )

            # row-coordinates of the "spotted cells" added to the other ones
            rows_to = np.append(rows_to, nr_spot)
            # column-coordinates of the "spotted cells" added to the other ones
            cols_to = np.append(cols_to, nc_spot)
            # time propagation of "spotted cells" added to the other ones
            nt = np.append(nt, nt_spot)
            transition_time = np.append(transition_time, transition_time_spot)

        TICK_PRECISION = 10
        prop_tick = np.rint((self.time + transition_time) * TICK_PRECISION)

        def extract_updates(tick: np.int64):
            # idx = np.nonzero(prop_time == t)
            idx = np.nonzero(prop_tick == tick)
            stacked = np.stack((rows_to[idx], cols_to[idx], nt[idx]), axis=1)
            return stacked

        # schedule the new updates
        unique_ticks = np.unique(prop_tick)
        new_updates = [
            Ignitions(t / TICK_PRECISION, extract_updates(t)) for t in unique_ticks
        ]

        return new_updates

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
            np.ndarray: Base moisture plus action-derived increments, clipped to
            [0, 100].
        """
        if self.actions_moisture is None:
            return self.moisture

        moisture = self.moisture + self.actions_moisture
        moisture = np.clip(moisture, 0, 100)

        return moisture

    def step(
        self,
    ) -> None:
        """Advance the simulation to the next scheduled time and update state."""
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

        new_updates = self.apply_updates(scheduler_event.coords)

        for new_ignitions in new_updates:
            self.scheduler.push_ignitions(new_ignitions)

    def get_output(self) -> PropagatorOutput:
        """Assemble the current outputs and summary stats into a dataclass.

        Returns:
            PropagatorOutput: Snapshot of fire probability, RoS, intensity, stats.
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
