"""The dynamics sum every rotor in an even-rotor airframe model."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath.base import rotx, roty, rotz

from roboticstoolbox.blocks.quad_model import quadrotor
from roboticstoolbox.blocks.uav import MultiRotor


@pytest.mark.parametrize("nrotors", [2, 4, 6, 8])
@pytest.mark.parametrize("thrust_ratio", [0.8, 1.0, 1.2])
def test_symmetric_rotors_balance_force_and_moments(
    nrotors: int, thrust_ratio: float
) -> None:
    model = dict(quadrotor, nrotors=nrotors)
    block = MultiRotor(model, groundcheck=False)
    speed = np.sqrt(thrust_ratio * model["M"] * model["g"] / (nrotors * model["b"]))
    rotors = speed * (-1.0) ** np.arange(nrotors)
    state = np.zeros(12)
    state[2] = -3

    expected = np.zeros(12)
    expected[8] = model["g"] * (1 - thrust_ratio)
    nt.assert_allclose(block.deriv(0, [rotors], state), expected, atol=1e-12)
    nt.assert_allclose(block.a1s, np.zeros(nrotors), atol=1e-12)
    nt.assert_allclose(block.b1s, np.zeros(nrotors), atol=1e-12)


@pytest.mark.parametrize("nrotors", [2, 4, 6, 8])
@pytest.mark.parametrize("last_rotor", [False, True])
def test_one_rotor_increment_matches_its_force_and_lever_arm(
    nrotors: int, last_rotor: bool
) -> None:
    model = dict(quadrotor, nrotors=nrotors)
    block = MultiRotor(model, groundcheck=False)
    squared_speeds = np.full(nrotors, model["M"] * model["g"] / (nrotors * model["b"]))
    selected = nrotors - 1 if last_rotor else 0
    increment = 13000.0
    squared_speeds[selected] += increment
    rotors = np.sqrt(squared_speeds) * (-1.0) ** np.arange(nrotors)
    state = np.zeros(12)
    state[:6] = [1, -2, -3, 0.5, -0.2, 0.3]
    rotation = rotz(state[3]) @ roty(state[4]) @ rotx(state[5])

    angle = 2 * np.pi * selected / nrotors
    additional_force = model["b"] * increment
    force = np.array([0, 0, -model["M"] * model["g"] - additional_force])
    moment = np.array(
        [
            -model["d"] * np.sin(angle) * additional_force,
            model["d"] * np.cos(angle) * additional_force,
            (-1.0) ** selected * model["k"] * increment,
        ]
    )
    expected = np.zeros(12)
    expected[6:9] = [0, 0, model["g"]] + rotation @ force / model["M"]
    expected[9:] = moment / np.diag(model["J"])

    nt.assert_allclose(block.deriv(0, [rotors], state), expected, atol=1e-12)
