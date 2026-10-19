"""How a plot is rendered is chosen from the environment, so that the same call
works whatever ``%matplotlib`` says.  In a notebook an ``ipympl`` figure cannot update
while a cell is running, so an animation is rendered inline instead."""

import sys
from unittest import mock

import matplotlib
import pytest

import roboticstoolbox as rtb
from roboticstoolbox.backends.PyPlot.PyPlot import PyPlot, _resolve_render_mode

# the package exports classes of the same names as these modules, so ask for the modules
pyplot_module = sys.modules["roboticstoolbox.backends.PyPlot.PyPlot"]
pyplot2_module = sys.modules["roboticstoolbox.backends.PyPlot.PyPlot2"]

IPYMPL = "module://ipympl.backend_nbagg"
INLINE = "module://matplotlib_inline.backend_inline"


def _environment(monkeypatch, notebook, backend):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: notebook)
    monkeypatch.setattr(matplotlib, "get_backend", lambda: backend)


@pytest.mark.parametrize("animation", [False, True])
def test_outside_a_notebook_it_is_a_window(monkeypatch, animation):
    _environment(monkeypatch, False, "agg")

    assert _resolve_render_mode(None, animation) == "window"


@pytest.mark.parametrize("animation", [False, True])
def test_inline_backend_is_inline(monkeypatch, animation):
    _environment(monkeypatch, True, INLINE)

    assert _resolve_render_mode(None, animation) == "notebook-inline"


def test_ipympl_is_a_widget_for_a_static_plot(monkeypatch):
    _environment(monkeypatch, True, IPYMPL)

    assert _resolve_render_mode(None) == "notebook-widget"
    assert _resolve_render_mode(None, animation=False) == "notebook-widget"


def test_ipympl_animation_is_inline(monkeypatch):
    _environment(monkeypatch, True, IPYMPL)

    assert _resolve_render_mode(None, animation=True) == "notebook-inline"


@pytest.mark.parametrize("mode", ["window", "notebook-widget", "notebook-inline"])
def test_an_explicit_mode_is_used_as_it_is(monkeypatch, mode):
    _environment(monkeypatch, True, IPYMPL)

    assert _resolve_render_mode(mode, animation=True) == mode


def test_an_invalid_mode_is_rejected():
    with pytest.raises(ValueError):
        _resolve_render_mode("popup")


def test_the_2d_backend_uses_the_same_choice():
    assert pyplot2_module._resolve_render_mode is _resolve_render_mode
    assert pyplot2_module._isnotebook is pyplot_module._isnotebook


def _launch_kwargs(robot, q):
    with mock.patch.object(PyPlot, "launch", autospec=True) as launch:
        try:
            robot.plot(q, backend="pyplot", dt=0.0)
        except Exception:
            pass  # the figure was not launched, only what was asked for matters
    return launch.call_args.kwargs


def test_plot_of_a_trajectory_asks_for_an_animation():
    robot = rtb.models.DH.Puma560()
    q = rtb.jtraj(robot.qz, robot.qn, 5).q

    assert _launch_kwargs(robot, q)["animation"] is True


def test_plot_of_one_pose_is_not_an_animation():
    robot = rtb.models.DH.Puma560()

    assert "animation" not in _launch_kwargs(robot, robot.qn)


def test_plot_leaves_an_explicit_render_mode_alone():
    robot = rtb.models.DH.Puma560()
    q = rtb.jtraj(robot.qz, robot.qn, 5).q

    with mock.patch.object(PyPlot, "launch", autospec=True) as launch:
        try:
            robot.plot(q, backend="pyplot", dt=0.0, render_mode="notebook-widget")
        except Exception:
            pass
    assert launch.call_args.kwargs["render_mode"] == "notebook-widget"
