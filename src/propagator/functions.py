"""Spread, probability, and intensity model functions.

This module contains pluggable formulations for rate-of-spread, probability
modulators for wind/slope/moisture, fire spotting distance, and fireline
intensity utilities used by the core propagator.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt
from numba import jit
from numpy.random import random

from propagator.constants import (
    C_MOIST,
    CELLSIZE,
    D1,
    D2,
    D3,
    D4,
    D5,
    FIRE_SPOTTING_DISTANCE_COEFFICIENT,
    M1,
    M2,
    M3,
    M4,
    NEIGHBOURS,
    NEIGHBOURS_ANGLE,
    NEIGHBOURS_DISTANCE,
    NO_FUEL,
    ROTHERMEL_ALPHA1,
    ROTHERMEL_ALPHA2,
    SPOTTING_RN_MEAN,
    SPOTTING_RN_STD,
    WANG_BETA1,
    WANG_BETA2,
    WANG_BETA3,
    A,
    Q,
)
from propagator.models import (
    Fuel,
    FuelSystem,
    PMoistFn,
    PTimeFn,
    UpdateBatchTuple,
)

type ROS_model_literal = Literal["default", "wang", "rothermel"]
type Moisture_model_literal = Literal[
    "default", "new_formulation", "rothermel"
]


@jit(cache=True)
def clip(x: float, min: float, max: float) -> float:
    """Clip x to the range [min, max]."""
    if x < min:
        return min
    if x > max:
        return max
    return x


@jit(cache=True)
def normalize(angle_to_norm: float) -> float:
    """Normalize an angle to the interval [-pi, pi)."""
    return (angle_to_norm + np.pi) % (2 * np.pi) - np.pi  # type: ignore[return-value]


def load_parameters(
    probability_file: str | None = None,
    v0_file: str | None = None,
    p_vegetation: str | None = None,
) -> None:
    """Override default vegetation parameters from text files.

    - probability_file: Path to vegetation-to-vegetation probability table.
    - v0_file: Path to base ROS per vegetation type.
    - p_vegetation: Path to ignition probabilities per vegetation type.
    """
    global v0, prob_table, p_veg
    if v0_file:
        v0 = np.loadtxt(v0_file)
    if probability_file:
        prob_table = np.loadtxt(probability_file)
    if p_vegetation:
        p_veg = np.loadtxt(p_vegetation)


def get_p_time_fn(ros_model_code: ROS_model_literal) -> PTimeFn:
    """Select a rate-of-spread model by code.

    Returns a function with signature `(v0, dem_from, dem_to,
    angle_to, dist, moist, w_dir, w_speed) -> (time, ros)`.
    """
    match ros_model_code:
        case "default":
            return p_time_standard
        case "wang":
            return p_time_wang
        case "rothermel":
            return p_time_rothermel

    raise ValueError(f"Unknown ros_model_code: {ros_model_code!r}")


def get_p_moist_fn(moist_model_code: Moisture_model_literal) -> PMoistFn:
    """Select a moisture probability correction by code."""
    match moist_model_code:
        case "default":
            return moist_proba_correction_1
        case "new_formulation":
            return moist_proba_correction_1
        case "rothermel":
            return moist_proba_correction_2

    raise ValueError(f"Unknown moist_model_code: {moist_model_code!r}")


def p_time_rothermel(
    v0: npt.NDArray[np.integer],
    dem_from: float,
    dem_to: float,
    angle_to: float,
    dist: float,
    moist: float,
    w_dir: float,
    w_speed: float,
) -> tuple[float, float]:
    """Propagation time and ROS according to Rothermel-like scaling.

    Parameters
    ----------
    dem_from, dem_to : numpy.ndarray
        Elevation at source and neighbor cells.
    angle_to : numpy.ndarray
        Direction to neighbor (radians).
    dist : numpy.ndarray
        Lattice distance to neighbor (cells).
    moist : numpy.ndarray
        Moisture values (%).
    w_dir : numpy.ndarray
        Wind direction (radians).
    w_speed : numpy.ndarray
        Wind speed (km/h).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (transition time [min], ROS [m/min]).
    """
    # velocità di base modulata con la densità(tempo di attraversamento)
    dh = dem_to - dem_from

    real_dist = np.sqrt((CELLSIZE * dist) ** 2 + dh**2)

    # wind component in propagation direction
    w_proj = np.cos(w_dir - angle_to)
    # wind speed in the direction of propagation
    w_spd = (w_speed * w_proj) / 3.6

    teta_s_rad = np.arctan(dh / CELLSIZE * dist)  # slope angle [rad]
    teta_s = np.degrees(teta_s_rad)  # slope angle [°]

    # flame angle measured from the vertical
    # in the direction of fire spread [rad]
    teta_f_rad = np.arctan(0.4226 * w_spd)
    teta_f = np.degrees(teta_f_rad)  # flame angle [°]

    sf = np.exp(ROTHERMEL_ALPHA1 * teta_s)  # slope factor
    sf_clip = clip(sf, 0.01, 10)  # slope factor clipped at 10
    wf = np.exp(ROTHERMEL_ALPHA2 * teta_f)  # wind factor
    wf_rescaled = wf / 13  # wind factor rescaled to have 10 as max value
    wf_clip = clip(wf_rescaled, 1, 20)  # max value is 20, min is 1

    v_wh_pre = (
        v0 * sf_clip * wf_clip
    )  # Rate of Spread evaluate with Rothermel's model
    moist_eff = np.exp(C_MOIST * moist)  # moisture effect

    # v_wh = clip(v_wh_pre, 0.01, 100) #adoptable RoS
    v_wh = clip(v_wh_pre * moist_eff, 0.01, 100)  # adoptable RoS [m/min]

    t = real_dist / v_wh

    return t, v_wh
    # return t


@jit(cache=True)
def p_time_wang(
    v0: float,
    dh: float,
    angle_to: float,
    dist: float,
    moist: float,
    w_dir: float,
    w_speed: float,
) -> tuple[float, float]:
    """Propagation time and ROS according to Wang et al.

    Parameters
    ----------
    v0 : numpy.ndarray
        Base ROS vector per vegetation type.
    dem_from, dem_to : numpy.ndarray
        Elevation at source and neighbor cells.
    angle_to : numpy.ndarray
        Direction to neighbor (radians).
    dist : numpy.ndarray
        Lattice distance to neighbor (cells).
    moist : numpy.ndarray
        Moisture values (%).
    w_dir : numpy.ndarray
        Wind direction (radians).
    w_speed : numpy.ndarray
        Wind speed (km/h).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (transition time [min], ROS [m/min]).
    """
    # velocità di base modulata con la densità(tempo di attraversamento)

    real_dist = np.sqrt((CELLSIZE * dist) ** 2 + dh**2)

    # wind component in propagation direction
    w_proj = np.cos(w_dir - angle_to)
    # wind speed in the direction of propagation
    w_spd = (w_speed * w_proj) / 3.6

    teta_s_rad = np.arctan(dh / CELLSIZE * dist)  # slope angle [rad]
    teta_s_pos = np.absolute(teta_s_rad)  # absolute values of slope angle
    # +1 if fire spreads upslope, -1 if fire spreads downslope
    p_reverse = np.sign(dh)

    wf = np.exp(WANG_BETA1 * w_spd)  # wind factor
    wf_clip = clip(wf, 0.01, 10)  # clipped at 10
    sf = np.exp(
        p_reverse * WANG_BETA2 * np.tan(teta_s_pos) ** WANG_BETA3
    )  # slope factor
    sf_clip = clip(sf, 0.01, 10)

    # Rate of Spread evaluate with Wang Zhengfei's model
    v_wh_pre = v0 * wf_clip * sf_clip
    moist_eff = np.exp(C_MOIST * moist)  # moisture effect

    # v_wh = clip(v_wh_pre, 0.01, 100) #adoptable RoS
    v_wh = clip(v_wh_pre * moist_eff, 0.01, 100)  # adoptable RoS [m/min]

    t = real_dist / v_wh

    return t, v_wh


def p_time_standard(
    v0: float,
    dem_from: float,
    dem_to: float,
    angle_to: float,
    dist: float,
    moist: float,
    w_dir: float,
    w_speed: float,
) -> tuple[float, float]:
    """Baseline propagation time and ROS with combined wind-slope factor.

    Parameters
    ----------
    v0 : numpy.ndarray
        Base ROS vector per vegetation type.
    dem_from, dem_to : numpy.ndarray
        Elevation at source and neighbor cells.
    angle_to : numpy.ndarray
        Direction to neighbor (radians).
    dist : numpy.ndarray
        Lattice distance to neighbor (cells).
    moist : numpy.ndarray
        Moisture values (%).
    w_dir : numpy.ndarray
        Wind direction (radians).
    w_speed : numpy.ndarray
        Wind speed (km/h).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        (transition time [min], ROS [m/min]).
    """
    dh = dem_to - dem_from
    wh = w_h_effect(angle_to, w_speed, w_dir, dh, dist)
    moist_eff = np.exp(C_MOIST * moist)  # moisture effect

    v_wh = clip(v0 * wh * moist_eff, 0.01, 100)

    real_dist = np.sqrt((CELLSIZE * dist) ** 2 + dh**2)
    t = real_dist / v_wh
    return t, v_wh


@jit(cache=True)
def w_h_effect(
    angle_to: float,
    w_speed: float,
    w_dir: float,
    dh: float,
    dist: float,
) -> float:
    """Combined wind and slope multiplicative factor on ROS.

    Returns
    -------
    numpy.ndarray
        Dimensionless multiplier applied to base ROS.
    """
    w_effect_module = (
        A + (D1 * (D2 * np.tanh((w_speed / D3) - D4))) + (w_speed / D5)
    )
    a = (w_effect_module - 1) / 4
    w_effect_on_direction = (
        (a + 1) * (1 - a**2) / (1 - a * np.cos(normalize(w_dir - angle_to)))
    )
    slope = dh / (CELLSIZE * dist)
    h_effect = 2 ** (np.tanh((slope * 3) ** 2.0 * np.sign(slope)))

    w_h = h_effect * w_effect_on_direction
    return w_h


@jit(cache=True)
def w_h_effect_on_probability(
    angle_to: float,
    w_speed: float,
    w_dir: float,
    dh: float,
    dist_to: float,
) -> float:
    """Scale the wind/slope factor for use as probability exponent.

    Returns
    -------
    numpy.ndarray
        Positive factor used as an exponent on the vegetation probability term;
        values > 1 increase spread, < 1 decrease it.
    """
    w_speed_norm = clip(w_speed, 0, 60)
    wh_orig = w_h_effect(angle_to, w_speed_norm, w_dir, dh, dist_to)
    wh = wh_orig - 1.0
    if wh > 0:
        wh = wh / 2.13
    elif wh < 0:
        wh = wh / 1.12
    wh += 1.0
    return wh


@jit(cache=True)
def moist_proba_correction_1(
    moist: float,
) -> float:
    """
    Moisture correction to the transition probability p_{i,j}.

    Uses a 5th-degree polynomial in x = moist/Mx, with Mx = 0.3
    (Trucchia et al., Fire 2020).
    """
    Mx = 0.3
    x = clip(moist, 0.0, 1.0) / Mx
    p_moist = (
        (-11.507 * x**5)
        + (22.963 * x**4)
        + (-17.331 * x**3)
        + (6.598 * x**2)
        + (-1.7211 * x)
        + 1.0003
    )
    p_moist = clip(p_moist, 0.0, 1.0)
    return p_moist


def moist_proba_correction_2(
    moist: float,
) -> float:
    """
    Moisture correction to p_{i,j}
    (older formulation, Baghino; Trucchia et al., 2020).
    Parameters come from constants.
    """
    p_moist = M1 * moist**3 + M2 * moist**2 + M3 * moist + M4
    return p_moist


def fire_spotting(
    angle_to: npt.NDArray[np.floating],
    w_dir: npt.NDArray[np.floating],
    w_speed: npt.NDArray[np.floating],
) -> npt.NDArray[np.floating]:
    """Evaluate spotting distance using Alexandridis' formulation."""
    r_n = np.random.normal(
        SPOTTING_RN_MEAN, SPOTTING_RN_STD, size=angle_to.shape
    )  # main thrust of the ember: sampled from a
    # Gaussian Distribution (Alexandridis et al, 2008 and 2011)
    w_speed_ms = w_speed / 3.6  # wind speed [m/s]
    # Alexandridis' formulation for spotting distance
    d_p = r_n * np.exp(
        w_speed_ms
        * FIRE_SPOTTING_DISTANCE_COEFFICIENT
        * (np.cos(w_dir - angle_to) - 1)
    )
    return d_p


# functions useful for evaluating the fire line intensity


@jit(cache=True)
def lhv_dead_fuel(
    hhv: float,
    dffm: float,
) -> float:
    """Lower heating value of dead fuels given higher
    heating value and FFMC."""
    lhv = hhv * (1.0 - (dffm / 100.0)) - Q * (dffm / 100.0)
    return lhv


@jit(cache=True)
def lhv_canopy(
    hhv: float,
    hum: float,
) -> float:
    """Lower heating value of canopy fuels given humidity (percent)."""
    lhv = hhv * (1.0 - (hum / 100.0)) - Q * (hum / 100.0)
    if np.isnan(lhv):
        lhv = 0
    return lhv


@jit(cache=True)
def fireline_intensity(
    d0: float,
    d1: float,
    ros: float,
    lhv_dead_fuel: float,
    lhv_canopy: float,
    rg: float | None = None,
) -> float:
    """Estimate fireline intensity (kW/m) from fuel loads and ROS.

    Supports an optional `rg` fraction to blend surface/canopy contributions.
    """
    # if rg is not None:
    #     rg_idx = ~np.isnan(rg)
    #     d1_idx = (not np.isclose(d1, 0.0, rtol=1e-09, atol=1e-09)) & rg_idx
    #     d0_idx = (np.isclose(d1, 0.0, rtol=1e-09, atol=1e-09)) & rg_idx
    #     intensity[d0_idx] = (
    #         ros[d0_idx]
    #         * ((lhv_dead_fuel[d0_idx] * d0[d0_idx] * (1.0 - rg[d0_idx])) / 2)
    #         / 60.0
    #     )
    #     intensity[d1_idx] = (
    #         ros[d1_idx]
    #         * (
    #             (
    #                 lhv_dead_fuel[d1_idx] * d0[d1_idx]
    #                 + lhv_canopy[d1_idx] * (d1[d1_idx] * (1 - rg[d1_idx]))
    #             )
    #             / 2
    #         )
    #         / 60.0
    #     )
    #     intensity[~rg_idx] = (
    #         ros[~rg_idx]
    #         * (
    #             (
    #                 lhv_dead_fuel[~rg_idx] * d0[~rg_idx]
    #                 + lhv_canopy[~rg_idx] * d1[~rg_idx]
    #             )
    #             / 2
    #         )
    #         / 60.0
    #     )
    # else:
    # divided by 60 instead of 3600 because RoS is required in m/s and
    # it is given in m/min (so it has to be divided by 60)
    intensity = ros * (lhv_dead_fuel * d0 + lhv_canopy * d1) / 60.0
    return intensity


@jit(cache=True, nopython=True, fastmath=True)
def get_probability_to_neighbour(
    angle_to: float,
    dist_to: float,
    w_dir_r: float,
    w_speed_r: float,
    moisture_r: float,
    dh: float,
    transition_probability: float,
) -> float:
    # get the probability for all the pixels
    moisture_effect = moist_proba_correction_1(moisture_r)  # type: ignore
    alpha_wh = w_h_effect_on_probability(
        angle_to, w_speed_r, w_dir_r, dh, dist_to
    )

    alpha_wh = np.maximum(alpha_wh, 0)  # prevent alpha < 0
    p_prob = 1 - (1 - transition_probability) ** alpha_wh
    p_prob = clip(p_prob * moisture_effect, 0, 1.0)
    # try the propagation
    return p_prob


@jit(cache=True, nopython=True, fastmath=True)
def calculate_fire_behavior(
    fuel_from: Fuel,
    fuel_to: Fuel,
    dh: float,
    dist_to: float,
    angle_to: float,
    moisture_r: float,
    w_dir_r: float,
    w_speed_r: float,
) -> tuple[int, float, float]:
    # get the propagation time for the propagating pixels
    # transition_time = p_time(dem_from[p], dem_to[p],

    transition_time, ros_value = p_time_wang(
        fuel_from.v0 / 60,  # m/min!!!
        dh,
        angle_to,
        dist_to,
        moisture_r,  # type: ignore
        w_dir_r,
        w_speed_r,
    )
    transition_time = int(transition_time)
    if transition_time < 1:
        transition_time = 1

    # evaluate LHV of dead fuel
    lhv_dead_fuel_value = lhv_dead_fuel(fuel_to.hhv, moisture_r)  # type: ignore
    # evaluate LHV of the canopy
    lhv_canopy_value = lhv_canopy(fuel_to.hhv, fuel_to.humidity)
    # evaluate fireline intensity
    fireline_intensity_value = fireline_intensity(
        fuel_to.d0,
        fuel_to.d1,
        ros_value,
        lhv_dead_fuel_value,
        lhv_canopy_value,
    )
    return transition_time, ros_value, fireline_intensity_value


# if do_spotting:
#     nr_spot, nc_spot, nt_spot, transition_time_spot =
# compute_spotting(
#         veg_type, update
#     )
#     # row-coordinates of the "spotted cells"
# added to the other ones
#     rows_to = np.append(rows_to, nr_spot)
#     # column-coordinates of the "spotted cells"
# added to the other ones
#     cols_to = np.append(cols_to, nc_spot)
#     # time propagation of "spotted cells"
# added to the other ones
#     nt = np.append(nt, nt_spot)
#     transition_time = np.append(transition_time,
# transition_time_spot)

# schedule the new updates


@jit(cache=True, parallel=False, nopython=True, fastmath=True)
def apply_single_update(
    row: int,
    col: int,
    veg: npt.NDArray[np.integer],
    dem: npt.NDArray[np.floating],
    fire: npt.NDArray[np.int8],
    moisture: npt.NDArray[np.floating],
    wind_dir: npt.NDArray[np.floating],
    wind_speed: npt.NDArray[np.floating],
    fuels: FuelSystem,
) -> list[tuple[int, int, int, float, float]]:
    fire_spread_updates = []

    dem_from = dem[row, col]
    veg_from = veg[row, col]
    w_dir_r = wind_dir[row, col]
    w_speed_r = wind_speed[row, col]

    for neighbour, dist_to, angle_to in zip(
        NEIGHBOURS, NEIGHBOURS_DISTANCE, NEIGHBOURS_ANGLE
    ):
        row_to = row + neighbour[0]
        col_to = col + neighbour[1]
        veg_to = veg[row_to, col_to]

        # keep only pixels where fire can spread
        if fire[row_to, col_to] != 0 or veg_to == NO_FUEL:
            continue

        dh = dem[row_to, col_to] - dem_from
        moisture_r = moisture[row_to, col_to]
        transition_probability = fuels.get_transition_probability(
            veg_from,
            veg_to,  # type: ignore
        )

        p_prob = get_probability_to_neighbour(
            angle_to,
            dist_to,
            w_dir_r,
            w_speed_r,
            moisture_r,  # type: ignore
            dh,
            transition_probability,
        )

        do_propagate = p_prob > random()
        if not do_propagate:
            continue

        fuel_from = fuels.get_fuel(veg_from)  # type: ignore
        fuel_to = fuels.get_fuel(veg_to)  # type: ignore

        transition_time, ros, fireline_intensity = calculate_fire_behavior(
            fuel_from,
            fuel_to,
            dh,
            dist_to,
            angle_to,
            moisture_r,  # type: ignore
            w_dir_r,
            w_speed_r,
        )
        fire_spread_updates.append(
            (transition_time, row_to, col_to, ros, fireline_intensity)
        )
    return fire_spread_updates


@jit(cache=True, parallel=False, nopython=True, fastmath=True)
def next_updates_fn(
    rows: npt.NDArray[np.integer],
    cols: npt.NDArray[np.integer],
    realizations: npt.NDArray[np.integer],
    time: int,
    veg: npt.NDArray[np.integer],
    dem: npt.NDArray[np.floating],
    fire: npt.NDArray[np.int8],
    moisture: npt.NDArray[np.floating],
    wind_dir: npt.NDArray[np.floating],
    wind_speed: npt.NDArray[np.floating],
    fuels: FuelSystem,
) -> UpdateBatchTuple:
    next_rows = []
    next_cols = []
    next_realizations = []
    next_times = []
    next_ros = []
    next_fireline_intensities = []

    for index in range(len(rows)):
        row: int = rows[index]
        col: int = cols[index]
        realization: int = realizations[index]

        fire_spread_update = apply_single_update(
            row,
            col,
            veg,
            dem,
            fire[:, :, realization],
            moisture,
            wind_dir,
            wind_speed,  # type: ignore
            fuels,
        )

        for fire_spread in fire_spread_update:
            (transition_time, row_to, col_to, ros, fireline_intensity) = (
                fire_spread
            )
            next_times.append(time + transition_time)
            next_rows.append(row_to)
            next_cols.append(col_to)
            next_realizations.append(realization)
            next_ros.append(ros)
            next_fireline_intensities.append(fireline_intensity)

    return (
        np.array(next_times),
        np.array(next_rows),
        np.array(next_cols),
        np.array(next_realizations),
        np.array(next_ros),
        np.array(next_fireline_intensities),
    )
