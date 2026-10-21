"""The joint limits and the display of a robot with symbolic parameters."""

import numpy as np
import numpy.testing as nt

import roboticstoolbox as rtb


def test_symbolic_robot_qlim():
    robot = rtb.models.DH.Puma560(symbolic=True)
    numeric = rtb.models.DH.Puma560()

    nt.assert_allclose(robot.qlim, numeric.qlim)


def test_symbolic_robot_can_be_printed():
    robot = rtb.models.DH.Puma560(symbolic=True)

    assert "DHRobot" in str(robot)


def test_unset_revolute_limits_are_the_default():
    robot = rtb.DHRobot([rtb.RevoluteDH(a=1.0)], name="one")

    nt.assert_allclose(robot.qlim, [[-np.pi], [np.pi]])
