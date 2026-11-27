"""Random configurations obey the limits of the addressed joint."""

import numpy as np
import numpy.testing as nt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox import ET, ET2, ETS, ETS2, Robot, Robot2


@pytest.mark.parametrize("et, ets_type, rotation", [(ET, ETS, "Rz"), (ET2, ETS2, "R")])
@pytest.mark.parametrize("indices", [[0, 1], [1, 0], [2, 5]])
@pytest.mark.parametrize("count", [1, 8])
def test_ets_random_q_respects_joint_indices(
    et, ets_type, rotation, indices, count
) -> None:
    rotate = getattr(et, rotation)
    sequence = ets_type(
        [
            rotate(jindex=indices[0], qlim=[1, 2]),
            et.tx(1),
            rotate(jindex=indices[1], qlim=[3, 4]),
        ]
    )
    q = sequence.random_q(count)
    assert q.shape == ((2,) if count == 1 else (count, 2))
    # Interpret exactly the vector returned to FK, including compact subchains.
    resolved = sequence._resolve_q(q, allow_trajectory=count != 1)
    for joint in sequence.joints():
        assert np.all(resolved[..., joint.jindex] >= joint.qlim[0])
        assert np.all(resolved[..., joint.jindex] <= joint.qlim[1])


@pytest.mark.parametrize(
    "et, robot_type, rotation", [(ET, Robot, "Rz"), (ET2, Robot2, "R")]
)
def test_robot_random_q_respects_joint_indices(et, robot_type, rotation) -> None:
    rotate = getattr(et, rotation)
    robot = robot_type(
        rotate(jindex=1, qlim=[1, 2]) * et.tx(1) * rotate(jindex=0, qlim=[3, 4])
    )
    q = robot.random_q()
    assert q.shape == (2,)
    for link in robot.links:
        if link.isjoint:
            assert link.qlim[0] <= q[link.jindex] <= link.qlim[1]


@pytest.mark.parametrize("count", [1, 5])
def test_seeded_ets_random_q_is_unchanged_for_an_ordinary_robot(count) -> None:
    # Book scripts seed the generator.  For a robot whose joint indices are in link
    # order, the vectorised draw must give the numbers that one draw per joint, in
    # order, always gave.
    ets = rtb.models.Panda().ets()
    qlim = ets.qlim
    np.random.seed(42)
    q = ets.random_q(count)
    np.random.seed(42)
    expected = np.array(
        [
            [np.random.uniform(qlim[0, j], qlim[1, j]) for j in range(ets.n)]
            for _ in range(count)
        ]
    )
    nt.assert_allclose(q, expected[0] if count == 1 else expected, rtol=0, atol=1e-15)


def test_seeded_robot_random_q_is_unchanged_for_an_ordinary_robot() -> None:
    robot = rtb.models.Panda()
    qlim = robot.qlim
    np.random.seed(42)
    q = robot.random_q()
    np.random.seed(42)
    expected = np.array(
        [np.random.uniform(qlim[0, j], qlim[1, j]) for j in range(robot.n)]
    )
    nt.assert_allclose(q, expected, rtol=0, atol=1e-15)
