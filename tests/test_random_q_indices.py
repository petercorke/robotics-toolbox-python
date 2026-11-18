"""Random configurations obey the limits of the addressed joint."""

import numpy as np
import pytest

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
