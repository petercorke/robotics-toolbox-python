"""
A time step of zero used to hang: ``plt.pause(0)`` waits until the event loop is stopped.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

import roboticstoolbox as rtb  # noqa: E402
from roboticstoolbox import ET2  # noqa: E402
from roboticstoolbox.backends.PyPlot import PyPlot, PyPlot2  # noqa: E402

pytestmark = pytest.mark.timeout(20)


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def env3d():
    env = PyPlot()
    env.launch()
    env.add(rtb.models.DH.Puma560(), readonly=True)
    return env


@pytest.fixture
def env2d():
    env = PyPlot2()
    env.launch()
    env.add(rtb.Robot2(ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1)), readonly=True)
    return env


@pytest.mark.parametrize("dt", [0, 0.0, -0.05])
def test_3d_step_needs_a_positive_dt(env3d, dt):
    with pytest.raises(ValueError, match="greater than zero"):
        env3d.step(dt)


@pytest.mark.parametrize("dt", [0, 0.0, -0.05])
def test_2d_step_needs_a_positive_dt(env2d, dt):
    with pytest.raises(ValueError, match="greater than zero"):
        env2d.step(dt)


def inline_env(cls, robot):
    env = cls()
    env.launch(render_mode="notebook-inline")
    env.add(robot, readonly=True)
    return env


@pytest.mark.parametrize("dt", [0, 0.0])
def test_in_a_notebook_there_is_no_pause_and_zero_is_fine(dt):
    # the notebook animation draws a frame for each step and does not pause
    inline_env(PyPlot, rtb.models.DH.Puma560()).step(dt)
    inline_env(
        PyPlot2, rtb.Robot2(ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1))
    ).step(dt)


def test_a_small_dt_is_fine(env3d, env2d):
    env3d.step(0.001)
    env2d.step(0.001)


def test_the_default_dt_is_fine(env3d):
    env3d.step()


def test_plot_with_a_zero_dt_raises_and_does_not_hang():
    puma = rtb.models.DH.Puma560()
    with pytest.raises(ValueError, match="greater than zero"):
        puma.plot(puma.qr, backend="pyplot", dt=0, block=False)
