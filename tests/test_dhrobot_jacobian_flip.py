"""DHRobot Jacobians must account for a flipped joint

The reference is a finite difference of forward kinematics, which shares no
code with the Jacobian, and the ETS Jacobian of the same robot.
"""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import (
    DHRobot,
    PrismaticDH,
    PrismaticMDH,
    RevoluteDH,
    RevoluteMDH,
)


def _robots() -> dict[str, DHRobot]:
    return {
        "std revolute flip": DHRobot(
            [RevoluteDH(a=0.4, alpha=0.5), RevoluteDH(d=0.2, a=0.3, flip=True)]
        ),
        "std prismatic flip": DHRobot(
            [
                RevoluteDH(a=0.4),
                PrismaticDH(a=0.1, alpha=-0.7, flip=True, qlim=[-1, 2]),
            ]
        ),
        "mdh prismatic flip offset": DHRobot(
            [
                RevoluteMDH(a=0.4),
                PrismaticMDH(a=0.1, alpha=-0.7, offset=0.3, flip=True, qlim=[-1, 2]),
            ]
        ),
        "mdh revolute flip offset": DHRobot(
            [
                RevoluteMDH(a=0.4, alpha=0.5, offset=0.4, flip=True),
                RevoluteMDH(d=0.2, a=0.3, offset=-0.3),
                PrismaticMDH(alpha=0.2, flip=True, qlim=[-1, 2]),
            ]
        ),
        "no flip control": DHRobot(
            [RevoluteDH(a=0.4, alpha=0.5), PrismaticDH(a=0.1, qlim=[-1, 2])]
        ),
    }


def _finite_difference_jacob0(robot: DHRobot, q: np.ndarray) -> np.ndarray:
    # velocity of the end-effector in the world frame, from forward kinematics
    h = 1e-6
    J = np.zeros((6, robot.n))
    for i in range(robot.n):
        dq = np.zeros(robot.n)
        dq[i] = h
        Tp, Tm = robot.fkine(q + dq).A, robot.fkine(q - dq).A
        dR = Tp[:3, :3] @ Tm[:3, :3].T
        J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * h)
        J[3:, i] = (
            np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            / 2
            / (2 * h)
        )
    return J


@pytest.mark.parametrize("name", list(_robots()))
def test_jacob0_matches_finite_difference_of_fkine(name: str) -> None:
    robot = _robots()[name]
    rng = np.random.default_rng(1)
    for _ in range(3):
        q = rng.uniform(-1, 1, robot.n)
        nt.assert_allclose(
            robot.jacob0(q), _finite_difference_jacob0(robot, q), atol=1e-6
        )


@pytest.mark.parametrize("name", list(_robots()))
def test_jacob0_and_jacobe_match_the_ets_jacobian(name: str) -> None:
    robot = _robots()[name]
    ets = robot.ets()
    rng = np.random.default_rng(2)
    for _ in range(3):
        q = rng.uniform(-1, 1, robot.n)
        nt.assert_allclose(robot.jacob0(q), ets.jacob0(q), atol=1e-9)
        nt.assert_allclose(robot.jacobe(q), ets.jacobe(q), atol=1e-9)


def test_flip_negates_exactly_the_flipped_columns() -> None:
    q = np.array([0.3, -0.4])
    plain = DHRobot([RevoluteDH(a=0.4), RevoluteDH(d=0.2, a=0.3)])
    flipped = DHRobot([RevoluteDH(a=0.4), RevoluteDH(d=0.2, a=0.3, flip=True)])
    # the flipped joint at q is the plain joint at -q
    nt.assert_allclose(
        flipped.jacobe(q)[:, 1], -plain.jacobe(np.array([0.3, 0.4]))[:, 1], atol=1e-12
    )
    nt.assert_allclose(
        flipped.jacobe(q)[:, 0], plain.jacobe(np.array([0.3, 0.4]))[:, 0]
    )
