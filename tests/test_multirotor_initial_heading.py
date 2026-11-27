"""The four-value initial pose specifies yaw without tilting the airframe."""

import bdsim
import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox.blocks.quad_model import quadrotor
from roboticstoolbox.blocks.uav import MultiRotor


@pytest.mark.parametrize("yaw", [0.0, 0.6, -1.1, np.pi / 2])
@pytest.mark.parametrize("use_parameter", [False, True])
def test_short_pose_preserves_heading(yaw: float, use_parameter: bool) -> None:
    initial = [1.2, -0.7, -3.0, yaw]
    if use_parameter:
        block = MultiRotor(quadrotor)
        block.set_param("x0", initial)
    else:
        block = MultiRotor(quadrotor, x0=initial)

    state = block.getstate0()
    nt.assert_allclose(state, np.r_[initial, np.zeros(8)])
    nt.assert_allclose(block.output(0, [], state)[0]["x"], [*initial, 0, 0])


@pytest.mark.parametrize("yaw", [0.0, np.pi / 3, -0.8])
def test_heading_only_initial_pose_can_hover(yaw: float) -> None:
    block = MultiRotor(quadrotor, x0=[1, 2, -3, yaw], groundcheck=False)
    speed = np.sqrt(quadrotor["M"] * quadrotor["g"] / (4 * quadrotor["b"]))
    rotors = speed * np.array([1, -1, 1, -1])

    # Equal, alternating rotors balance gravity and all moments. A yaw change
    # keeps the thrust vertical, so all twelve state derivatives remain zero.
    nt.assert_allclose(block.deriv(0, [rotors], block.getstate0()), 0, atol=1e-12)


@pytest.mark.parametrize("use_parameter", [False, True])
def test_heading_only_hover_in_a_real_diagram(use_parameter: bool) -> None:
    sim = bdsim.BDSim(graphics=False, animation=False, quiet=True)
    bd = sim.blockdiagram()
    initial = [1.0, 2.0, -3.0, np.pi / 3]
    block = bd.MULTIROTOR(
        quadrotor, x0=None if use_parameter else initial, groundcheck=False
    )
    if use_parameter:
        block.set_param("x0", initial)
    speed = np.sqrt(quadrotor["M"] * quadrotor["g"] / (4 * quadrotor["b"]))
    rotors = bd.CONSTANT(speed * np.array([1, -1, 1, -1]))
    pose = bd.FUNCTION(lambda state: state["x"], nin=1, nout=1)
    bd.connect(rotors, block)
    bd.connect(block, pose)
    bd.compile()
    result = sim.run(bd, T=0.2, dt=0.02, watch=[pose[0]])

    expected = np.tile([*initial, 0, 0], (len(result.t), 1))
    nt.assert_allclose(result.y, expected, atol=1e-10)


@pytest.mark.parametrize("size", [3, 6, 12])
def test_other_initial_state_forms_keep_their_order(size: int) -> None:
    initial = np.arange(size, dtype=float) / 10
    block = MultiRotor(quadrotor, x0=initial)
    nt.assert_array_equal(block.getstate0(), np.r_[initial, np.zeros(12 - size)])
