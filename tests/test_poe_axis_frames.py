"""PoE frame construction supports axes parallel to the reference x axis."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3, Twist3

from roboticstoolbox import PoEPrismatic, PoERevolute, PoERobot, Robot
from roboticstoolbox.tools.types import NDArray


@pytest.mark.parametrize(
    "axis",
    [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1e-10, -1e-10],
    ],
)
@pytest.mark.parametrize("joint", ["revolute", "prismatic"])
def test_poe_axis_frame(axis: list[float], joint: str) -> None:
    direction: NDArray = np.asarray(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    q = [0.45]
    home = SE3.Trans(0.3, -0.2, 0.5) * SE3.RPY(0.2, -0.3, 0.4)
    if joint == "revolute":
        # A point on the x axis still gives zero principal point for an x axis.
        point = [0.2, 0, 0]
        link = PoERevolute(direction, point)
        twist = Twist3.UnitRevolute(direction, point)
    else:
        link = PoEPrismatic(direction)
        twist = Twist3.UnitPrismatic(direction)

    robot = PoERobot([link], home)
    converted = Robot(robot.ets())
    expected = twist.exp(q[0]) * home

    nt.assert_allclose(robot.fkine(q).A, expected.A, atol=1e-9)
    nt.assert_allclose(converted.fkine(q).A, expected.A, atol=1e-9)
    nt.assert_allclose(converted.jacob0(q), robot.jacob0(q), atol=1e-9)
    nt.assert_allclose(converted.jacobe(q), robot.jacobe(q), atol=1e-9)
