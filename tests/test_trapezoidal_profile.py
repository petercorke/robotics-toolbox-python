"""The scalar profile functions: shape of the result, and the trapezoid as an object"""

from types import SimpleNamespace

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import quintic_func, trapezoidal_func
from roboticstoolbox.blocks.arm import Traj


@pytest.mark.parametrize("builder", [quintic_func, trapezoidal_func])
def test_scalar_time_gives_scalars(builder) -> None:
    for value in builder(0, 1, 2)(0.7):
        nt.assert_equal(np.ndim(value), 0)


@pytest.mark.parametrize("builder", [quintic_func, trapezoidal_func])
@pytest.mark.parametrize("times", [[0.5, 1.0], [[0.0, 0.5], [1.0, 2.0]]])
def test_array_time_keeps_its_shape(builder, times) -> None:
    for value in builder(0, 1, 2)(times):
        nt.assert_equal(np.shape(value), np.shape(times))


def test_trapezoidal_func_is_a_callable_object_with_attributes() -> None:
    from roboticstoolbox.tools.trajectory import TrapezoidalProfile

    f = trapezoidal_func(0, 1, 2)
    nt.assert_equal(isinstance(f, TrapezoidalProfile), True)
    nt.assert_allclose(f.V, 0.75)  # 1.5 times the average velocity
    nt.assert_allclose(f.tb, 2 / 3)  # blend duration
    nt.assert_allclose(f(0.0)[0], 0)
    nt.assert_allclose(f(2.0)[0], 1)
    nt.assert_allclose(f([0.0, 1.0, 2.0])[0], [0, 0.5, 1])


def test_trapezoidal_func_validates_velocity() -> None:
    with pytest.raises(ValueError, match="V too small"):
        trapezoidal_func(0, 1, 2, V=0.1)
    with pytest.raises(ValueError, match="V too big"):
        trapezoidal_func(0, 1, 2, V=10)


@pytest.mark.parametrize("traj", ["trapezoidal", "quintic"])
def test_traj_block_outputs_one_row_per_axis(traj: str) -> None:
    block = Traj(y0=[0, 1], yf=[1, 3], T=2, time=True, traj=traj)
    block.start(SimpleNamespace(T=2))
    y, yd, ydd = block.output(1.0, [], [])
    nt.assert_allclose(y, [0.5, 2.0], atol=1e-12)
    nt.assert_equal(np.shape(yd), (2,))
    nt.assert_equal(np.shape(ydd), (2,))


def test_traj_block_rejects_an_unknown_trajectory_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown traj"):
        Traj(y0=0, yf=1, T=1, time=True, traj="cubic")
