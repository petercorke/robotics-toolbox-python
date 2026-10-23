"""Bad arguments to blocks are errors, not asserts, which python -O removes"""

import numpy as np
import pytest

import roboticstoolbox as rtb

pytest.importorskip("bdsim")

from roboticstoolbox.blocks.arm import CirclePath, Traj  # noqa: E402


def test_circlepath_centre_must_be_a_3_vector():
    with pytest.raises(ValueError, match="centre"):
        CirclePath(radius=1, centre=[1, 2], pose=None)


def test_traj_y0_and_yf_must_be_the_same_length():
    with pytest.raises(ValueError, match="same length"):
        Traj(y0=[0, 0], yf=[1, 2, 3], T=1)


def test_traj_is_not_defined_before_zero_or_after_the_end():
    block = Traj(y0=0, yf=1, T=1, time=True)

    class State:
        tf = 1.0

    block.start(State())
    with pytest.raises(ValueError, match="x<0"):
        block.output(-0.1, None, None)
    with pytest.raises(ValueError, match="x>"):
        block.output(1.1, None, None)


def test_rne_of_a_standard_dh_robot_is_a_value_error():
    # the guard was an assert, so python -O let it return a wrong torque
    from roboticstoolbox.robot.Robot import Robot

    puma = rtb.models.DH.Puma560()
    z = np.zeros(6)
    with pytest.raises(ValueError, match="standard DH"):
        Robot.rne(puma, z, z, z)
