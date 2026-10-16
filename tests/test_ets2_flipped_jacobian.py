"""Planar Jacobians differentiate the signed joint coordinates used by FK."""

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath.base import tr2xyt

from roboticstoolbox import ET2, ETS2
from roboticstoolbox.tools.types import NDArray


def _numerical_jacobian(ets: ETS2, q: NDArray, frame: str) -> NDArray:
    step = 1e-6
    result = np.empty((3, ets.n))
    for joint in range(ets.n):
        delta = np.zeros(ets.n)
        delta[joint] = step
        before = tr2xyt(ets.fkine(q - delta).A)
        after = tr2xyt(ets.fkine(q + delta).A)
        result[:, joint] = (after - before) / (2 * step)

    if frame == "end":
        result[:2] = ets.fkine(q).R.T @ result[:2]
    return result


@pytest.mark.parametrize("kind", ["R", "tx", "ty"])
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("frame", ["base", "end"])
def test_planar_joint_jacobian(kind: str, flip: bool, frame: str) -> None:
    ets = ETS2([ET2.R(0.4), getattr(ET2, kind)(flip=flip), ET2.tx(0.7), ET2.ty(0.2)])
    q = np.array([0.3])
    actual = ets.jacob0(q) if frame == "base" else ets.jacobe(q)

    nt.assert_allclose(actual, _numerical_jacobian(ets, q, frame), atol=1e-9)


@pytest.mark.parametrize("frame", ["base", "end"])
def test_mixed_planar_flipped_joint_jacobian(frame: str) -> None:
    ets = ETS2(
        [
            ET2.tx(0.2),
            ET2.R(flip=True),
            ET2.tx(0.5),
            ET2.ty(flip=True),
            ET2.R(0.6),
            ET2.tx(),
            ET2.ty(0.3),
        ]
    )
    q = np.array([0.35, 0.25, 0.4])
    actual = ets.jacob0(q) if frame == "base" else ets.jacobe(q)

    nt.assert_allclose(actual, _numerical_jacobian(ets, q, frame), atol=1e-9)
