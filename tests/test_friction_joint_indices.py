"""Friction vectors address joints, independently of fixed links and traversal order."""

import numpy as np
import pytest

from roboticstoolbox import ET, ET2, DHRobot, Link, Link2, RevoluteDH, Robot, Robot2


@pytest.mark.parametrize(
    "et, link_type, robot_type, rotation",
    [(ET, Link, Robot, "Rz"), (ET2, Link2, Robot2, "R")],
)
@pytest.mark.parametrize("indices", [[0, 1], [1, 0]])
@pytest.mark.parametrize("fixed_links", [False, True])
@pytest.mark.parametrize(
    "velocities, expected",
    [([2, -3], [-2.2, 20.1]), ([-2, 3], [1.8, -21.3]), ([0, 0], [0, 0])],
)
def test_friction_uses_joint_links(
    et, link_type, robot_type, rotation, indices, fixed_links, velocities, expected
) -> None:
    links = []
    parent = None
    rotate = getattr(et, rotation)
    for i, (viscous, gear, coulomb) in enumerate(
        [(0.2, 2, [0.3, -0.1]), (0.7, -3, [0.8, -0.4])]
    ):
        if fixed_links:
            parent = link_type(et.tx(0.1), name=f"fixed{i}", parent=parent, B=9, G=10)
            links.append(parent)
        parent = link_type(
            rotate(jindex=indices[i]),
            name=f"joint{i}",
            parent=parent,
            B=viscous,
            G=gear,
            Tc=coulomb,
        )
        links.append(parent)
    robot = robot_type(links)
    qd = np.empty(2)
    expected_tau = np.empty(2)
    qd[indices] = velocities
    expected_tau[indices] = expected

    np.testing.assert_allclose(robot.friction(qd), expected_tau)


def test_dh_friction_preserves_serial_joint_order() -> None:
    robot = DHRobot(
        [
            RevoluteDH(B=0.2, G=2, Tc=[0.3, -0.1]),
            RevoluteDH(B=0.7, G=-3, Tc=[0.8, -0.4]),
        ]
    )
    np.testing.assert_allclose(robot.friction([2, -3]), [-2.2, 20.1])
