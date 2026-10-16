"""Behaviour that the typing pass of tools/trajectory.py exposed"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import jtraj, mtraj, quintic, trapezoidal_func
from roboticstoolbox.tools.trajectory import Trajectory


def test_qplot_runs() -> None:
    # xplot was never imported into the module: NameError
    tg = mtraj(quintic, [0, 1], [1, 2], 20)
    tg.qplot()
    plt.close("all")


def test_jtraj_accepts_a_list_of_times() -> None:
    # the documentation says array_like, but only an ndarray had .flatten()
    times = [0.0, 0.5, 1.0, 1.5, 2.0]
    from_list = jtraj([0, 1], [1, 2], times)
    from_array = jtraj([0, 1], [1, 2], np.array(times))
    nt.assert_allclose(from_list.q, from_array.q)
    nt.assert_allclose(from_list.qd, from_array.qd)


def test_trajectory_without_velocity_raises() -> None:
    tg = Trajectory("test", np.arange(3.0), np.zeros(3))
    nt.assert_array_equal(tg.q, np.zeros(3))
    with pytest.raises(ValueError, match="no velocity"):
        tg.qd
    with pytest.raises(ValueError, match="no acceleration"):
        tg.qdd


def test_trapezoidal_func_attributes() -> None:
    f = trapezoidal_func(0, 1, 2)
    nt.assert_allclose(f.V, 0.75)  # 1.5 times the average velocity
    nt.assert_allclose(f.tb, 2 / 3)  # blend duration
    # a scalar and a vector of times give a vector and a matching row
    nt.assert_allclose(f(1.0)[0], [0.5])
    nt.assert_allclose(f([0.0, 1.0, 2.0])[0], [0, 0.5, 1])
