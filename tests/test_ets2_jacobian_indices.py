"""Planar Jacobian queries preserve explicit joint indexing."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import ET2, ETS2


@pytest.mark.parametrize("indices", [(0, 3), (3, 0), (2, 5), (0, 1)])
@pytest.mark.parametrize("frame", ["base", "end"])
def test_planar_jacobian_preserves_joint_indices(
    indices: tuple[int, int], frame: str
) -> None:
    ets = ETS2(
        [
            ET2.R(jindex=indices[0]),
            ET2.tx(0.7),
            ET2.R(jindex=indices[1]),
            ET2.tx(0.4),
        ]
    )
    q = np.full(max(indices) + 1, 0.9)
    q[list(indices)] = [0.3, -0.6]
    pose_before = ets.fkine(q).A.copy()
    joint_indices = ets.jindices.copy()
    expected = np.array(
        [
            [-0.7 * np.sin(0.3) - 0.4 * np.sin(-0.3), -0.4 * np.sin(-0.3)],
            [0.7 * np.cos(0.3) + 0.4 * np.cos(-0.3), 0.4 * np.cos(-0.3)],
            [1.0, 1.0],
        ]
    )
    if frame == "end":
        expected[:2] = ets.fkine(q).R.T @ expected[:2]

    actual = ets.jacob0(q) if frame == "base" else ets.jacobe(q)

    nt.assert_array_equal(ets.jindices, joint_indices)
    nt.assert_array_equal([joint.jindex for joint in ets.joints()], indices)
    nt.assert_allclose(ets.fkine(q).A, pose_before, atol=1e-14)
    nt.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("frame", ["base", "end"])
def test_planar_jacobian_compact_sparse_coordinates(frame: str) -> None:
    ets = ETS2([ET2.R(jindex=0), ET2.tx(0.7), ET2.R(jindex=3), ET2.tx(0.4)])
    q_compact = [0.3, -0.6]
    q_global = [0.3, 0.8, -0.9, -0.6]

    jacobian = ets.jacob0 if frame == "base" else ets.jacobe
    nt.assert_allclose(jacobian(q_compact), jacobian(q_global), atol=1e-14)
    nt.assert_array_equal(ets.jindices, [0, 3])
