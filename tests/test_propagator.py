import numpy as np
import pytest
from unittest.mock import MagicMock

from propagator.propagator import (
    Propagator,
    PropagatorBoundaryConditions,
    PropagatorActions,
    PropagatorStats,
    PropagatoOutput,
    PropagatorError,
)
from propagator.scheduler import Scheduler

# Mock functions for p_time_fn and p_moist_fn
def mock_p_time_fn(ros_0, dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moisture_r, w_dir_r, w_speed_r):
    # Simple mock: always return a propagation time of 1.0 and a ROS of 10.0
    return np.array([1.0] * len(dem_from)), np.array([10.0] * len(dem_from))

def mock_p_moist_fn(moist):
    # Simple mock: returns a constant moisture correction
    return np.full_like(moist, 0.9)

@pytest.fixture
def sample_propagator():
    # Define some sample input data for the Propagator
    veg = np.array([[1, 2, 0], [3, 4, 1], [0, 5, 2]], dtype=np.int8)
    dem = np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], dtype=np.float32)
    realizations = 2
    ros_0 = np.array([5.0, 6.0, 7.0, 8.0, 9.0], dtype=np.float16) # for 5 vegetation types
    probability_table = np.array(
        [[0.1, 0.2, 0.0, 0.0, 0.0],
         [0.3, 0.4, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32
    )
    veg_parameters = np.array(
        [[1.0, 0.1, 18000, 5],
         [1.2, 0.2, 19000, 6],
         [0.0, 0.0, 0, 0], # Placeholder for veg type 3 (index 2)
         [1.1, 0.15, 17500, 5],
         [1.3, 0.25, 20000, 7]], dtype=np.float32
    )
    do_spotting = False

    propagator = Propagator(
        veg=veg,
        dem=dem,
        realizations=realizations,
        ros_0=ros_0,
        probability_table=probability_table,
        veg_parameters=veg_parameters,
        do_spotting=do_spotting,
        p_time_fn=mock_p_time_fn,
        p_moist_fn=mock_p_moist_fn,
    )
    return propagator

def test_propagator_init(sample_propagator):
    prop = sample_propagator
    shape = prop.veg.shape
    realizations = prop.realizations

    assert prop.fire.shape == shape + (realizations,)
    assert prop.fire.dtype == np.int8
    assert np.all(prop.fire == 0)

    assert prop.ros.shape == shape + (realizations,)
    assert prop.ros.dtype == np.float16
    assert np.all(prop.ros == 0)

    assert prop.fireline_int.shape == shape + (realizations,)
    assert prop.fireline_int.dtype == np.float16
    assert np.all(prop.fireline_int == 0)

    assert prop.actions_moisture.shape == shape
    assert prop.actions_moisture.dtype == np.float16
    assert np.all(prop.actions_moisture == 0)

    assert isinstance(prop.scheduler, Scheduler)
    assert prop.time == 0

def test_set_ignitions(sample_propagator):
    prop = sample_propagator
    ignitions = np.array([[0, 0], [1, 1]], dtype=bool)
    time = 1

    prop.set_ignitions(ignitions, time)

    # Check if ignitions are pushed to the scheduler
    assert len(prop.scheduler) > 0
    next_time, updates = prop.scheduler.pop()
    assert next_time == time
    assert len(updates) == len(np.argwhere(ignitions)) * prop.realizations

    # Check if fire array is updated (set to 0 for initial ignition, will be set to 1 on first apply_updates)
    # The current implementation sets fire[p[0], p[1], t] = 0. This is a bit counter-intuitive for an ignition.
    # It seems to indicate "not yet burning"
    # assert prop.fire[0, 0, 0] == 0 # This will pass but might be misleading based on name

def test_compute_fire_probability(sample_propagator):
    prop = sample_propagator
    # Manually set some fire values for testing
    prop.fire[0, 0, 0] = 1
    prop.fire[0, 0, 1] = 1
    prop.fire[1, 1, 0] = 1
    # Expected: (1+1)/2 = 1 for (0,0), (1+0)/2 = 0.5 for (1,1), 0 elsewhere
    expected_prob = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    
    # We need to reshape expected_prob to match the original shape of the grid for comparison
    # given that prop.veg has shape (3,3)
    shape_0 = prop.veg.shape[0]
    shape_1 = prop.veg.shape[1]
    
    # Pad expected_prob with zeros if its shape is smaller than prop.veg.shape
    padded_expected_prob = np.zeros((shape_0, shape_1))
    padded_expected_prob[:expected_prob.shape[0], :expected_prob.shape[1]] = expected_prob
    
    assert np.allclose(prop.compute_fire_probability(), padded_expected_prob[:prop.fire.shape[0], :prop.fire.shape[1]])


def test_compute_ros_max(sample_propagator):
    prop = sample_propagator
    prop.ros[0, 0, 0] = 10.0
    prop.ros[0, 0, 1] = 15.0
    prop.ros[1, 1, 0] = 5.0
    prop.ros[1, 1, 1] = 8.0
    expected_ros_max = np.array([[15.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 0.0]])
    
    shape_0 = prop.veg.shape[0]
    shape_1 = prop.veg.shape[1]
    
    padded_expected_ros_max = np.zeros((shape_0, shape_1))
    padded_expected_ros_max[:expected_ros_max.shape[0], :expected_ros_max.shape[1]] = expected_ros_max
    
    assert np.allclose(prop.compute_ros_max(), padded_expected_ros_max[:prop.ros.shape[0], :prop.ros.shape[1]])


def test_compute_ros_mean(sample_propagator):
    prop = sample_propagator
    prop.ros[0, 0, 0] = 10.0
    prop.ros[0, 0, 1] = 15.0
    prop.ros[1, 1, 0] = 5.0
    prop.ros[1, 1, 1] = 0.0 # This should be ignored in mean calculation
    expected_ros_mean = np.array([[12.5, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 0.0]])
    
    shape_0 = prop.veg.shape[0]
    shape_1 = prop.veg.shape[1]
    
    padded_expected_ros_mean = np.zeros((shape_0, shape_1))
    padded_expected_ros_mean[:expected_ros_mean.shape[0], :expected_ros_mean.shape[1]] = expected_ros_mean
    
    assert np.allclose(prop.compute_ros_mean(), padded_expected_ros_mean[:prop.ros.shape[0], :prop.ros.shape[1]])


def test_compute_fireline_int_max(sample_propagator):
    prop = sample_propagator
    prop.fireline_int[0, 0, 0] = 100.0
    prop.fireline_int[0, 0, 1] = 150.0
    prop.fireline_int[1, 1, 0] = 50.0
    prop.fireline_int[1, 1, 1] = 80.0
    expected_fl_int_max = np.array([[150.0, 0.0, 0.0], [0.0, 80.0, 0.0], [0.0, 0.0, 0.0]])
    
    shape_0 = prop.veg.shape[0]
    shape_1 = prop.veg.shape[1]
    
    padded_expected_fl_int_max = np.zeros((shape_0, shape_1))
    padded_expected_fl_int_max[:expected_fl_int_max.shape[0], :expected_fl_int_max.shape[1]] = expected_fl_int_max
    
    assert np.allclose(prop.compute_fireline_int_max(), padded_expected_fl_int_max[:prop.fireline_int.shape[0], :prop.fireline_int.shape[1]])


def test_compute_fireline_int_mean(sample_propagator):
    prop = sample_propagator
    prop.fireline_int[0, 0, 0] = 100.0
    prop.fireline_int[0, 0, 1] = 150.0
    prop.fireline_int[1, 1, 0] = 50.0
    prop.fireline_int[1, 1, 1] = 0.0 # This should be ignored
    expected_fl_int_mean = np.array([[125.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 0.0]])
    
    shape_0 = prop.veg.shape[0]
    shape_1 = prop.veg.shape[1]
    
    padded_expected_fl_int_mean = np.zeros((shape_0, shape_1))
    padded_expected_fl_int_mean[:expected_fl_int_mean.shape[0], :expected_fl_int_mean.shape[1]] = expected_fl_int_mean
    
    assert np.allclose(prop.compute_fireline_int_mean(), padded_expected_fl_int_mean[:prop.fireline_int.shape[0], :prop.fireline_int.shape[1]])


def test_compute_stats(sample_propagator):
    prop = sample_propagator
    # Mock active scheduler for n_active
    prop.scheduler.active = MagicMock(return_value=np.array([[0,0,0], [1,1,0]]))

    # For compute_stats, 'values' is typically fire_probability
    values = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
    stats = prop.compute_stats(values)

    assert isinstance(stats, PropagatorStats)
    assert stats.n_active == 2 # From mock
    assert stats.area_mean == 1.5
    assert stats.area_50 == 2.0
    assert stats.area_75 == 1.0
    assert stats.area_90 == 1.0

def test_propagation_probability(sample_propagator):
    prop = sample_propagator
    # Use simple arrays for testing
    dem_from = np.array([10])
    dem_to = np.array([12])
    veg_from = np.array([1])
    veg_to = np.array([2])
    angle_to = np.array([np.pi / 4])
    dist_to = np.array([1.0])
    moist = np.array([10.0])
    w_dir = np.array([0.0])
    w_speed = np.array([5.0])

    # Mock w_h_effect_on_probability and p_moist_fn for deterministic results
    prop.p_moist_fn = MagicMock(return_value=np.array([0.8]))
    # For simplicity, let's assume w_h_effect_on_probability returns 1.5
    with (
        pytest.MonkeyPatch().context() as mp,
    ):
        mp.setattr(
            "propagator.propagator.w_h_effect_on_probability",
            lambda angle_to, w_speed, w_dir, dh, dist_to: np.array([1.5]),
        )
        probability = prop.propagation_probability(
            dem_from, dem_to, veg_from, veg_to, angle_to, dist_to, moist, w_dir, w_speed
        )

        # p_veg from probability_table[veg_to-1, veg_from-1] = probability_table[1,0] = 0.3
        # probability = 1 - (1 - p_veg) ** alpha_wh = 1 - (1 - 0.3)**1.5 = 1 - 0.7**1.5 = 1 - 0.5856 = 0.4143
        # final probability = probability * p_moist = 0.4143 * 0.8 = 0.3314
        assert np.isclose(probability, 0.3314, atol=1e-04)

def test_set_boundary_conditions(sample_propagator):
    prop = sample_propagator
    initial_time = prop.time
    
    # Test setting moisture and wind
    bc = PropagatorBoundaryConditions(
        time=initial_time,
        ignitions=None,
        moisture=np.full(prop.veg.shape, 20.0),
        wind_dir=np.full(prop.veg.shape, np.pi),
        wind_speed=np.full(prop.veg.shape, 10.0),
    )
    prop.set_boundary_conditions(bc)
    assert np.all(prop.moisture == 20.0)
    assert np.all(prop.wind_dir == np.pi)
    assert np.all(prop.wind_speed == 10.0)

    # Test setting ignitions
    ignitions = np.array([[0, 0]], dtype=bool)
    bc_ignitions = PropagatorBoundaryConditions(
        time=initial_time + 1,
        ignitions=ignitions,
        moisture=None,
        wind_dir=None,
        wind_speed=None,
    )
    prop.set_boundary_conditions(bc_ignitions)
    assert len(prop.scheduler) > 0
    # The actual fire array update happens in apply_updates after popping from scheduler

    # Test error for past time
    with pytest.raises(ValueError, match="Boundary conditions cannot be applied in the past."):
        past_bc = PropagatorBoundaryConditions(
            time=initial_time - 1,
            ignitions=None,
            moisture=None,
            wind_dir=None,
            wind_speed=None,
        )
        prop.set_boundary_conditions(past_bc)

def test_apply_actions(sample_propagator):
    prop = sample_propagator
    initial_time = prop.time
    prop.actions_moisture = np.zeros(prop.veg.shape, dtype=np.float16)
    
    # Test additional moisture
    actions_moisture = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    actions = PropagatorActions(
        time=initial_time,
        additional_moisture=actions_moisture,
        vegetation_changes=None,
    )
    prop.apply_actions(actions)
    assert np.all(prop.actions_moisture == actions_moisture)

    # Test vegetation changes
    veg_changes = np.array([[np.nan, 0.0, np.nan], [0.0, np.nan, np.nan], [np.nan, np.nan, np.nan]])
    initial_veg = prop.veg.copy()
    actions_veg = PropagatorActions(
        time=initial_time,
        additional_moisture=None,
        vegetation_changes=veg_changes,
    )
    prop.apply_actions(actions_veg)
    expected_veg = initial_veg.copy()
    expected_veg[0,1] = 0
    expected_veg[1,0] = 0
    assert np.all(prop.veg[~np.isnan(veg_changes)] == veg_changes[~np.isnan(veg_changes)])
    assert np.all(prop.veg[np.isnan(veg_changes)] == initial_veg[np.isnan(veg_changes)])


    # Test error for past time
    with pytest.raises(ValueError, match="Actions cannot be applied in the past."):
        past_actions = PropagatorActions(
            time=initial_time - 1,
            additional_moisture=None,
            vegetation_changes=None,
        )
        prop.apply_actions(past_actions)

def test_decay_actions_moisture(sample_propagator):
    prop = sample_propagator
    prop.actions_moisture = np.array([[10.0, 5.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 0.0]])
    
    # Test decay with default factor
    prop.decay_actions_moisture(time_delta=1)
    expected_moisture_default = np.array([[9.9, 4.95, 0.0], [0.0, 19.8, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(prop.actions_moisture, expected_moisture_default)

    # Test decay with custom factor
    prop.actions_moisture = np.array([[10.0, 5.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 0.0]])
    prop.decay_actions_moisture(time_delta=1, decay_factor=0.1)
    expected_moisture_custom = np.array([[9.0, 4.5, 0.0], [0.0, 18.0, 0.0], [0.0, 0.0, 0.0]])
    assert np.allclose(prop.actions_moisture, expected_moisture_custom)

    # Test when actions_moisture is None
    prop.actions_moisture = None
    prop.decay_actions_moisture(time_delta=1)
    assert prop.actions_moisture is None

def test_get_moisture(sample_propagator):
    prop = sample_propagator
    prop.moisture = np.full(prop.veg.shape, 5.0)
    
    # Case 1: No additional actions_moisture
    prop.actions_moisture = None
    assert np.all(prop.get_moisture() == 5.0)

    # Case 2: With additional actions_moisture
    prop.actions_moisture = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    expected_moisture = np.array([[6.0, 5.0, 5.0], [5.0, 7.0, 5.0], [5.0, 5.0, 5.0]])
    assert np.all(prop.get_moisture() == expected_moisture)

    # Case 3: Clipping to 100
    prop.moisture = np.full(prop.veg.shape, 99.0)
    prop.actions_moisture = np.full(prop.veg.shape, 5.0)
    expected_moisture_clipped = np.full(prop.veg.shape, 100.0)
    assert np.all(prop.get_moisture() == expected_moisture_clipped)

def test_step(sample_propagator):
    prop = sample_propagator
    
    # Mock scheduler and apply_updates
    mock_updates = [(1, np.array([[0, 0, 0], [1, 1, 0]]))]
    prop.scheduler.pop = MagicMock(return_value=(1, [[0, 0, 0], [1, 1, 0]]))
    prop.apply_updates = MagicMock(return_value=mock_updates)
    prop.scheduler.push_all = MagicMock()
    prop.decay_actions_moisture = MagicMock()

    prop.step()

    prop.scheduler.pop.assert_called_once()
    prop.decay_actions_moisture.assert_called_once_with(1)
    prop.apply_updates.assert_called_once() # Args are tricky to assert here without deep understanding of internal data
    prop.scheduler.push_all.assert_called_once_with(mock_updates)
    assert prop.time == 1

def test_get_output(sample_propagator):
    prop = sample_propagator
    prop.time = 5
    prop.fire = np.zeros(prop.veg.shape + (prop.realizations,))
    prop.fire[0,0,0] = 1
    prop.ros = np.full(prop.veg.shape + (prop.realizations,), 10.0)
    prop.fireline_int = np.full(prop.veg.shape + (prop.realizations,), 100.0)

    # Mock compute_stats as it depends on internal state of scheduler
    prop.compute_stats = MagicMock(return_value=PropagatorStats(n_active=1, area_mean=1.0, area_50=1.0, area_75=1.0, area_90=1.0))
    
    output = prop.get_output()

    assert isinstance(output, PropagatoOutput)
    assert output.time == 5
    assert np.allclose(output.fire_probability, prop.compute_fire_probability())
    assert np.allclose(output.ros_mean, prop.compute_ros_mean())
    assert np.allclose(output.ros_max, prop.compute_ros_max())
    assert np.allclose(output.fireline_int_mean, prop.compute_fireline_int_mean())
    assert np.allclose(output.fireline_int_max, prop.compute_fireline_int_max())
    assert isinstance(output.stats, PropagatorStats)

def test_next_time(sample_propagator):
    prop = sample_propagator
    # Test when scheduler is empty
    prop.scheduler = MagicMock()
    prop.scheduler.__len__.return_value = 0
    assert prop.next_time() is None

    # Test when scheduler has items
    prop.scheduler.__len__.return_value = 1
    prop.scheduler.next_time.return_value = 10
    assert prop.next_time() == 10
    
    # Test initial state where time is 0
    prop.time = 0
    prop.scheduler.__len__.return_value = 1
    prop.scheduler.next_time.return_value = 10
    assert prop.next_time() == 0
