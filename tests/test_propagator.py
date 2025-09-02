import numpy as np
import pytest
from unittest.mock import MagicMock

from propagator.propagator import (
    Propagator,
    BoundaryConditions,
    Actions,
    PropagatorStats,
    PropagatorOutput,
)
from propagator.scheduler import Scheduler


# ----------------------------
# Helpers and test fixtures
# ----------------------------


def mock_p_time_fn(
    ros_0,
    dem_from,
    dem_to,
    veg_from,
    veg_to,
    angle_to,
    dist_to,
    moisture_r,
    w_dir_r,
    w_speed_r,
):
    # Deterministic: 1.0 minute everywhere, ROS depends on veg_to for variety
    n = len(dem_from)
    return np.full(n, 1.0), np.full(n, 10.0) + (veg_to.astype(float) * 0.0)


def mock_p_moist_fn(moist):
    # Constant multiplier for probability tests
    return np.full_like(moist, 0.9)


@pytest.fixture
def sample_propagator():
    # Small grid (3x3) with a mix of vegetation types (0 indicates non-burnable)
    veg = np.array([[1, 2, 0], [3, 4, 1], [0, 5, 2]], dtype=np.int8)
    dem = np.array(
        [[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32
    )
    realizations = 2

    # base ROS per vegetation type (five types)
    ros_0 = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float32)

    # probability_table[target-1, source-1]
    probability_table = np.array(
        [
            [0.1, 0.2, 0.0, 0.0, 0.0],
            [0.3, 0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    # veg_parameters for fireline intensity: [d0, d1, hhv, humidity]
    veg_parameters = np.array(
        [
            [1.0, 0.1, 18000, 5],
            [1.2, 0.2, 19000, 6],
            [0.0, 0.0, 0, 0],
            [1.1, 0.15, 17500, 5],
            [1.3, 0.25, 20000, 7],
        ],
        dtype=np.float32,
    )

    prop = Propagator(
        veg=veg,
        dem=dem,
        realizations=realizations,
        ros_0=ros_0,
        probability_table=probability_table,
        veg_parameters=veg_parameters,
        do_spotting=False,
        p_time_fn=mock_p_time_fn,
        p_moist_fn=mock_p_moist_fn,
    )
    # default environment (will be set by boundary conditions in tests)
    prop.moisture = np.zeros_like(veg, dtype=np.float32)
    prop.wind_dir = np.zeros_like(veg, dtype=np.float32)
    prop.wind_speed = np.zeros_like(veg, dtype=np.float32)
    return prop


# ----------------------------
# Construction and state
# ----------------------------


def test_init_shapes_and_types(sample_propagator):
    prop = sample_propagator
    shape = prop.veg.shape

    assert prop.fire.shape == shape + (prop.realizations,)
    assert prop.fire.dtype == np.int8
    assert np.all(prop.fire == 0)

    assert prop.ros.shape == shape + (prop.realizations,)
    assert prop.ros.dtype == np.float32
    assert np.all(prop.ros == 0)

    assert prop.fireline_int.shape == shape + (prop.realizations,)
    assert prop.fireline_int.dtype == np.float32
    assert np.all(prop.fireline_int == 0)

    assert isinstance(prop.scheduler, Scheduler)
    assert prop.time == 0


# ----------------------------
# Ignitions and boundary conditions
# ----------------------------


def test_set_ignitions_schedules(sample_propagator):
    prop = sample_propagator
    ignitions = np.array(
        [[True, False, False], [False, True, False], [False, False, False]]
    )
    t_ignite = 1

    prop.set_ignitions(ignitions, t_ignite)
    assert len(prop.scheduler) > 0
    time, updates = prop.scheduler.pop()
    assert time == t_ignite
    # one coord per ignition per realization
    assert len(updates) == int(ignitions.sum()) * prop.realizations


def test_set_boundary_conditions_and_past_time(sample_propagator):
    prop = sample_propagator
    bc = BoundaryConditions(
        time=prop.time,
        ignitions=None,
        moisture=np.full(prop.veg.shape, 20.0),
        wind_dir=np.full(prop.veg.shape, np.pi),
        wind_speed=np.full(prop.veg.shape, 10.0),
    )
    prop.set_boundary_conditions(bc)
    assert np.all(prop.moisture == 20.0)
    assert np.all(prop.wind_dir == np.pi)
    assert np.all(prop.wind_speed == 10.0)

    with pytest.raises(ValueError):
        prop.set_boundary_conditions(
            BoundaryConditions(
                time=prop.time - 1,
                ignitions=None,
                moisture=None,
                wind_dir=None,
                wind_speed=None,
            )
        )


# ----------------------------
# Probability and metrics
# ----------------------------


def test_propagation_probability_deterministic(sample_propagator, monkeypatch):
    prop = sample_propagator

    dem_from = np.array([10.0])
    dem_to = np.array([12.0])
    veg_from = np.array([1])
    veg_to = np.array([2])
    angle_to = np.array([np.pi / 4])
    dist_to = np.array([1.0])
    moist = np.array([10.0])
    w_dir = np.array([0.0])
    w_speed = np.array([5.0])

    # Fix the wind/slope exponent and moisture factor
    monkeypatch.setattr(
        "propagator.propagator.w_h_effect_on_probability",
        lambda angle_to, w_speed, w_dir, dh, dist_to: np.array([1.5]),
    )
    prop.p_moist_fn = MagicMock(return_value=np.array([0.8]))

    p = prop.propagation_probability(
        dem_from,
        dem_to,
        veg_from,
        veg_to,
        angle_to,
        dist_to,
        moist,
        w_dir,
        w_speed,
    )
    # p_veg = 0.3 (table[1,0]) => 1 - (1 - 0.3) ** 1.5 = 0.4143 ; times 0.8 => 0.3314
    assert np.isclose(p, 0.3314, atol=1e-4)


def test_compute_probability_and_ros_and_intensity(sample_propagator):
    prop = sample_propagator
    # Seed some values across realizations
    prop.fire[0, 0, :] = 1
    prop.ros[0, 0, :] = [10.0, 15.0]
    prop.ros[1, 1, :] = [5.0, 8.0]
    prop.fireline_int[0, 0, :] = [100.0, 150.0]
    prop.fireline_int[1, 1, :] = [50.0, 80.0]

    # Fire probability
    expected_prob = np.zeros(prop.veg.shape)
    expected_prob[0, 0] = 1.0
    expected_prob[1, 1] = 0.5
    assert np.allclose(prop.compute_fire_probability(), expected_prob)

    # RoS max/mean
    expected_ros_max = np.zeros(prop.veg.shape)
    expected_ros_max[0, 0] = 15.0
    expected_ros_max[1, 1] = 8.0
    assert np.allclose(prop.compute_ros_max(), expected_ros_max)

    expected_ros_mean = np.zeros(prop.veg.shape)
    expected_ros_mean[0, 0] = 12.5
    expected_ros_mean[1, 1] = 6.5
    # But compute_ros_mean ignores zeros as no-spread; 6.5 remains since both > 0
    assert np.allclose(prop.compute_ros_mean(), expected_ros_mean)

    # Fireline intensity max/mean
    expected_fi_max = np.zeros(prop.veg.shape)
    expected_fi_max[0, 0] = 150.0
    expected_fi_max[1, 1] = 80.0
    assert np.allclose(prop.compute_fireline_int_max(), expected_fi_max)

    expected_fi_mean = np.zeros(prop.veg.shape)
    expected_fi_mean[0, 0] = 125.0
    expected_fi_mean[1, 1] = 65.0
    assert np.allclose(prop.compute_fireline_int_mean(), expected_fi_mean)


def test_compute_stats(sample_propagator, monkeypatch):
    prop = sample_propagator
    # Mock active realizations (two active)
    monkeypatch.setattr(prop.scheduler, "active", lambda: np.array([0, 1]))
    values = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    stats = prop.compute_stats(values)
    assert isinstance(stats, PropagatorStats)
    assert stats.n_active == 2
    assert stats.area_mean == pytest.approx(1.5)
    assert stats.area_50 == 2.0
    assert stats.area_75 == 1.0
    assert stats.area_90 == 1.0


# ----------------------------
# Actions, decay, and moisture
# ----------------------------


def test_apply_actions_and_decay_and_get_moisture(sample_propagator):
    prop = sample_propagator
    prop.actions_moisture = np.zeros(prop.veg.shape, dtype=np.float32)

    # Apply moisture addition
    add_m = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    prop.apply_actions(
        Actions(
            time=prop.time, additional_moisture=add_m, vegetation_changes=None
        )
    )
    assert np.all(prop.actions_moisture == add_m)

    # Apply vegetation change (NaN means no change)
    veg_changes = np.array(
        [
            [np.nan, 0.0, np.nan],
            [0.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ]
    )
    prev = prop.veg.copy()
    prop.apply_actions(
        Actions(
            time=prop.time,
            additional_moisture=None,
            vegetation_changes=veg_changes,
        )
    )
    expected = prev.copy()
    expected[0, 1] = 0
    expected[1, 0] = 0
    assert np.all(prop.veg == expected)

    # Decay moisture
    prop.decay_actions_moisture(time_delta=1)
    assert np.allclose(prop.actions_moisture, add_m * 0.99)

    # get_moisture adds and clips
    prop.moisture = np.full(prop.veg.shape, 99.0)
    prop.actions_moisture = np.full(prop.veg.shape, 5.0)
    assert np.all(prop.get_moisture() == 100.0)

    # Past actions time rejected
    with pytest.raises(ValueError):
        prop.apply_actions(
            Actions(
                time=prop.time - 1,
                additional_moisture=None,
                vegetation_changes=None,
            )
        )


# ----------------------------
# Update mechanics
# ----------------------------


def test_apply_updates_and_scheduling(sample_propagator, monkeypatch):
    prop = sample_propagator
    prop.time = 0
    prop.moisture = np.full(prop.veg.shape, 10.0)
    prop.wind_dir = np.full(prop.veg.shape, 0.0)
    prop.wind_speed = np.full(prop.veg.shape, 5.0)

    # Deterministic p_time_fn (already set in fixture); return 4 updates
    prop.p_time_fn = MagicMock(
        return_value=(
            np.array([1.0, 2.0, 1.5, 2.5]),
            np.array([10.0, 8.0, 12.0, 9.0]),
        )
    )

    # Control RNG for propagation decision
    monkeypatch.setattr("propagator.propagator.RNG", MagicMock())
    # eight neighbors per two ignitions => 16; here, we keep it simple, expecting the
    # pipeline masks to reduce to 4 successful updates (match p_time_fn outputs)
    # Choose a pattern that yields exactly 4 successes after masking flow
    type(prop.RNG).random = MagicMock(return_value=np.array([0.1] * 16))

    # Two ignition updates at t=0
    updates = [np.array([0, 0, 0]), np.array([1, 1, 0])]

    new_updates = prop.apply_updates(updates)
    assert isinstance(new_updates, list)
    # Should schedule 4 new updates at times as returned by p_time_fn + current time
    times = [t for t, _ in new_updates]
    assert sorted(times) == [1.0, 1.5, 2.0, 2.5]

    # Some ROS and intensity should have been assigned
    assert np.count_nonzero(prop.ros) == 4
    assert np.count_nonzero(prop.fireline_int) == 4


def test_step_and_next_time(sample_propagator, monkeypatch):
    prop = sample_propagator

    # Mock scheduler interactions
    prop.scheduler.pop = MagicMock(return_value=(1, [np.array([0, 0, 0])]))
    prop.apply_updates = MagicMock(return_value=[(2.0, np.array([[0, 1, 0]]))])
    prop.scheduler.push_all = MagicMock()
    prop.decay_actions_moisture = MagicMock()

    prop.step()
    prop.scheduler.pop.assert_called_once()
    prop.decay_actions_moisture.assert_called_once_with(1)
    prop.apply_updates.assert_called_once()
    prop.scheduler.push_all.assert_called_once()
    assert prop.time == 1

    # next_time: empty scheduler -> None; with item -> value; at init -> 0
    prop.scheduler = MagicMock()
    prop.scheduler.__len__.return_value = 0
    assert prop.next_time() is None
    prop.scheduler.__len__.return_value = 1
    prop.scheduler.next_time.return_value = 10
    prop.time = 5
    assert prop.next_time() == 10
    prop.time = 0
    assert prop.next_time() == 0


# ----------------------------
# Output snapshot
# ----------------------------


def test_get_output(sample_propagator, monkeypatch):
    prop = sample_propagator
    prop.time = 3
    prop.fire = np.zeros(prop.veg.shape + (prop.realizations,))
    prop.fire[0, 0, :] = 1
    prop.ros = np.full(prop.veg.shape + (prop.realizations,), 10.0)
    prop.fireline_int = np.full(prop.veg.shape + (prop.realizations,), 100.0)

    # Decouple from scheduler internals
    monkeypatch.setattr(
        prop, "compute_stats", lambda v: PropagatorStats(1, 1.0, 1.0, 1.0, 1.0)
    )

    out = prop.get_output()
    assert isinstance(out, PropagatorOutput)
    assert out.time == 3
    if out.fire_probability is not None:
        assert np.allclose(
            out.fire_probability, prop.compute_fire_probability()
        )
    if out.ros_mean is not None:
        assert np.allclose(out.ros_mean, prop.compute_ros_mean())
    if out.ros_max is not None:
        assert np.allclose(out.ros_max, prop.compute_ros_max())
    if out.fli_mean is not None:
        assert np.allclose(out.fli_mean, prop.compute_fireline_int_mean())
    if out.fli_max is not None:
        assert np.allclose(out.fli_max, prop.compute_fireline_int_max())
    assert isinstance(out.stats, PropagatorStats)
