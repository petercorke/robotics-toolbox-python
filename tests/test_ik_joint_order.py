"""
Inverse kinematics returns joint values in the order that forward kinematics reads them.

``ETS.eval`` reads a joint vector by joint number, so when the joint numbers of a full
chain are a permutation of ``0..n-1`` the solution must be in joint-number order, and
running it through ``eval`` must reproduce the requested pose.  Every solver has to
agree: the Python ``ikine_XX`` solvers and the compiled ``ik_XX`` methods.
"""

import numpy as np
import pytest

from roboticstoolbox import ET, ETS
from tests import requires_compiled, skip_no_qp

LIMITS = [-3.0, 3.0]
TOLERANCE = 5e-3  # the solvers' default tolerance is on a weighted, squared error


def arm(numbers: list[int]) -> ETS:
    """A six-joint arm; ``numbers[k]`` is the joint number carried by the k-th joint."""
    return ETS(
        [
            ET.Rz(jindex=numbers[0], qlim=LIMITS),
            ET.tz(0.3),
            ET.Ry(jindex=numbers[1], qlim=LIMITS),
            ET.tx(0.4),
            ET.Ry(jindex=numbers[2], qlim=LIMITS),
            ET.tx(0.35),
            ET.Rz(jindex=numbers[3], qlim=LIMITS),
            ET.Ry(jindex=numbers[4], qlim=LIMITS),
            ET.tx(0.1),
            ET.Rz(jindex=numbers[5], qlim=LIMITS),
            ET.tx(0.05),
        ]
    )


#: the value of each joint along the chain, so the same pose in every numbering
Q_ALONG_CHAIN = np.array([0.4, 0.7, -0.9, 0.5, 0.8, -0.3])

#: joint numbers along the chain
NUMBERINGS = {
    "in link order": [0, 1, 2, 3, 4, 5],
    "first two swapped": [1, 0, 2, 3, 4, 5],
    "reversed": [5, 4, 3, 2, 1, 0],
    "rotated": [2, 3, 4, 5, 0, 1],
}

SOLVERS = [
    "ikine_LM",
    "ikine_NR",
    "ikine_GN",
    pytest.param("ikine_QP", marks=skip_no_qp),
    pytest.param("ik_LM", marks=requires_compiled),
    pytest.param("ik_NR", marks=requires_compiled),
    pytest.param("ik_GN", marks=requires_compiled),
]


def q_in_joint_number_order(numbers: list[int]) -> np.ndarray:
    """The pose ``Q_ALONG_CHAIN`` written as a vector indexed by joint number."""
    q = np.empty(6)
    q[numbers] = Q_ALONG_CHAIN
    return q


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("numbers", NUMBERINGS.values(), ids=NUMBERINGS.keys())
def test_solution_reproduces_the_pose(solver: str, numbers: list[int]) -> None:
    ets = arm(numbers)
    target = ets.eval(q_in_joint_number_order(numbers))

    # one of the solvers may fail from an unlucky start, so ask for most of several
    reproduced = 0
    for seed in range(5):
        kwargs = {"seed": seed} if solver.startswith("ikine") else {}
        solution = getattr(ets, solver)(target, **kwargs)
        reproduced += np.abs(ets.eval(solution.q) - target).max() < TOLERANCE
    assert reproduced >= 4, f"{reproduced} of 5 solutions reproduce the pose"


def test_the_numberings_are_the_same_arm() -> None:
    """Guard on the test itself: each numbering describes one and the same pose."""
    reference = arm(NUMBERINGS["in link order"]).eval(
        q_in_joint_number_order(NUMBERINGS["in link order"])
    )
    for numbers in NUMBERINGS.values():
        pose = arm(numbers).eval(q_in_joint_number_order(numbers))
        np.testing.assert_allclose(pose, reference, atol=1e-12)


@pytest.mark.parametrize("solver", ["ikine_LM", pytest.param("ik_LM", marks=requires_compiled)])
def test_a_sparse_subchain_keeps_compact_coordinates(solver: str) -> None:
    """Joint numbers 3..8 are not 0..n-1, so a length-n vector is compact, in chain order."""
    ets = arm([3, 4, 5, 6, 7, 8])
    target = ets.eval(Q_ALONG_CHAIN)
    reproduced = 0
    for seed in range(5):
        kwargs = {"seed": seed} if solver.startswith("ikine") else {}
        solution = getattr(ets, solver)(target, **kwargs)
        assert solution.q.shape == (6,)
        reproduced += np.abs(ets.eval(solution.q) - target).max() < TOLERANCE
    assert reproduced >= 4
