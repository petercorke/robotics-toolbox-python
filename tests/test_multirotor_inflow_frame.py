"""Rotor aerodynamics use relative velocity expressed in the airframe."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath.base import rotx, roty, rotz

from roboticstoolbox.blocks.quad_model import quadrotor
from roboticstoolbox.blocks.uav import MultiRotor


@pytest.mark.parametrize("world_yaw", [0.0, 1.1, -1.7])
@pytest.mark.parametrize("tilted", [False, True])
@pytest.mark.parametrize("translating", [False, True])
@pytest.mark.parametrize("rotating", [False, True])
def test_global_yaw_rotation_preserves_body_dynamics(
    world_yaw: float, tilted: bool, translating: bool, rotating: bool
) -> None:
    block = MultiRotor(quadrotor, groundcheck=False)
    state = np.zeros(12)
    state[:3] = [1.2, -0.7, -4]
    if tilted:
        state[3:6] = [0.2, -0.15, 0.25]
    if translating:
        state[6:9] = [2.1, -0.7, 0.5]
    if rotating:
        state[9:] = [0.11, -0.16, 0.08]
    rotors = 800 * np.array([1.03, -0.97, 1.01, -0.99])
    original = state.copy()
    derivative = block.deriv(0, [rotors], state)
    flapping = np.array([block.a1s, block.b1s])

    # Rotate the entire experiment about world gravity. Body axes, angular
    # velocity and rotor commands are unchanged; world vectors must rotate.
    rotation = rotz(world_yaw)
    transformed = state.copy()
    transformed[:3] = rotation @ state[:3]
    transformed[3] += world_yaw
    transformed[6:9] = rotation @ state[6:9]
    actual = block.deriv(0, [rotors], transformed)
    expected = derivative.copy()
    expected[:3] = rotation @ derivative[:3]
    expected[6:9] = rotation @ derivative[6:9]

    nt.assert_allclose(actual, expected, atol=1e-12)
    nt.assert_allclose([block.a1s, block.b1s], flapping, atol=1e-12)
    nt.assert_array_equal(state, original)


@pytest.mark.parametrize("yaw", [0.0, np.pi / 2, -0.9])
@pytest.mark.parametrize("tilted", [False, True])
def test_forward_airflow_matches_independent_force_balance(
    yaw: float, tilted: bool
) -> None:
    block = MultiRotor(quadrotor, groundcheck=False)
    state = np.zeros(12)
    state[2] = -3
    state[3] = yaw
    if tilted:
        state[4:6] = [-0.2, 0.3]
    rotation = rotz(state[3]) @ roty(state[4]) @ rotx(state[5])
    forward_speed = 2.0
    state[6:9] = rotation @ np.array([forward_speed, 0, 0])
    rotor_speed = np.sqrt(quadrotor["M"] * quadrotor["g"] / (4 * quadrotor["b"]))
    rotors = rotor_speed * np.array([1, -1, 1, -1])

    # No body rotation or normal inflow: all rotors have the same longitudinal
    # flap angle. Sum their tilted thrust, then add world gravity.
    advance = forward_speed / (rotor_speed * quadrotor["r"])
    flap = (
        (8 * quadrotor["theta0"] / 3 + 2 * quadrotor["theta1"])
        * advance
        / (1 - advance**2 / 2)
    )
    thrust = (
        -quadrotor["M"] * quadrotor["g"] * np.array([np.sin(flap), 0, np.cos(flap)])
    )
    expected_acceleration = [0, 0, quadrotor["g"]] + rotation @ thrust / quadrotor["M"]
    # The hubs are all at height h, so their horizontal thrust has this net
    # pitching moment; symmetric in-plane lever arms and drag torques cancel.
    expected_angular = np.array([0, quadrotor["h"] * thrust[0], 0]) / np.diag(
        quadrotor["J"]
    )

    actual = block.deriv(0, [rotors], state)
    nt.assert_allclose(actual[6:9], expected_acceleration, atol=1e-12)
    nt.assert_allclose(actual[9:], expected_angular, atol=1e-12)
    nt.assert_allclose(block.a1s, flap, atol=1e-12)
    nt.assert_allclose(block.b1s, 0, atol=1e-12)
