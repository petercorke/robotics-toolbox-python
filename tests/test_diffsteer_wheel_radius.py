"""Wheel angular inputs must produce the motion of wheels with radius R."""

import bdsim
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from roboticstoolbox.blocks.mobile import DiffSteer


@pytest.mark.parametrize("radius", [0.1, 0.37, 1.0])
@pytest.mark.parametrize("heading", [0.0, 0.7])
@pytest.mark.parametrize("rates", [(2.0, 2.0), (-2.0, -2.0), (-2.0, 2.0), (1.0, 4.0)])
def test_wheel_radius_sets_linear_and_angular_motion(radius, heading, rates):
    block = DiffSteer(R=radius, w=0.5)
    state = np.array([1.0, -2.0, heading])
    inputs = np.array(rates)
    original = inputs.copy()
    centre_speed = radius * sum(rates) / 2
    turn_rate = radius * (rates[1] - rates[0]) / 0.5

    actual = block.test_deriv(*inputs, x=state)

    assert_allclose(
        actual,
        [centre_speed * np.cos(heading), centre_speed * np.sin(heading), turn_rate],
        atol=1e-14,
    )
    assert_array_equal(inputs, original)
    assert_array_equal(block.test_output(*inputs, x=state)[0], state)


@pytest.mark.parametrize("radius", [0.1, 1.0])
@pytest.mark.parametrize("rates", [(2.0, 2.0), (-2.0, -2.0), (-2.0, 2.0), (1.0, 4.0)])
def test_real_block_diagram_follows_wheel_travel(radius, rates):
    sim = bdsim.BDSim(
        graphics=False, animation=False, progress=False, banner=False, quiet=True
    )
    bd = sim.blockdiagram()
    left, right = (bd.CONSTANT(rate) for rate in rates)
    initial = np.array([1.0, -2.0, 0.7])
    block = bd.DIFFSTEER(R=radius, w=0.5, x0=initial)
    bd.connect(left, block[0])
    bd.connect(right, block[1])
    bd.compile()

    result = sim.run(bd, T=1.25, dt=0.025, watch=[block[0]])

    # Each wheel travels R * omega * t. The centre follows a straight line
    # for equal travel, or an arc about the instantaneous centre otherwise.
    left_travel, right_travel = radius * np.outer(rates, result.t)
    centre_travel = (left_travel + right_travel) / 2
    heading_change = (right_travel - left_travel) / 0.5
    heading = initial[2] + heading_change
    expected = np.tile(initial, (len(result.t), 1))
    expected[:, 2] = heading
    if rates[0] == rates[1]:
        expected[:, 0] += centre_travel * np.cos(initial[2])
        expected[:, 1] += centre_travel * np.sin(initial[2])
    else:
        turn_radius = 0.25 * sum(rates) / (rates[1] - rates[0])
        expected[:, 0] += turn_radius * (np.sin(heading) - np.sin(initial[2]))
        expected[:, 1] -= turn_radius * (np.cos(heading) - np.cos(initial[2]))
    assert_allclose(result.y, expected, atol=2e-8, rtol=1e-9)


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_speed_limits_apply_to_linear_wheel_speeds(direction):
    block = DiffSteer(R=0.2, w=0.5, speed_max=0.3)
    # The left wheel moves at 0.2; only the right wheel reaches the 0.3 limit.
    actual = block.test_deriv(direction, 4 * direction, x=[0, 0, 0])
    assert_allclose(actual, direction * np.array([0.25, 0.0, 0.2]), atol=1e-14)


def test_acceleration_limits_receive_linear_wheel_speeds():
    block = DiffSteer(R=0.1, w=0.5, accel_max=0.3)
    # The model's 0.1-second limiter permits 0.03 linear speed per call.
    # The left target (0.02) is already below that change; the right (0.04)
    # takes two calls. These are physical wheel speeds, not angular rates.
    assert_allclose(block.test_deriv(0.2, 0.4, x=[0, 0, 0]), [0.025, 0, 0.02])
    assert_allclose(block.test_deriv(0.2, 0.4, x=[0, 0, 0]), [0.03, 0, 0.04])


def test_radius_remains_fixed_but_width_and_speed_limit_are_settable():
    block = DiffSteer(R=0.2, w=0.5)
    with pytest.raises(ValueError):
        block.set_param("R", 0.4)
    block.set_param("w", 1.0)
    block.set_param("speed_max", 0.3)
    assert_allclose(block.test_deriv(1.0, 4.0, x=[0, 0, 0]), [0.25, 0, 0.1])
