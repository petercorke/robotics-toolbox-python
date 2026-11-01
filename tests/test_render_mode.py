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


# --- display in a notebook ---------------------------------------------------
# These run on the Agg backend with the display function replaced, so they check
# what is asked of IPython and when, not how a real frontend (VS Code, Colab,
# JupyterLab) draws it.


@pytest.fixture
def displayed(monkeypatch):
    import matplotlib.pyplot as plt

    shown = []
    monkeypatch.setattr(pyplot_module, "_ipy_display", shown.append)
    plt.close("all")
    yield shown
    plt.close("all")


def test_a_widget_figure_is_displayed_once_when_it_is_drawn(displayed):
    import matplotlib.pyplot as plt

    robot = rtb.models.DH.Puma560()
    env = PyPlot()
    env.launch(render_mode="notebook-widget")

    # created quietly: nothing shown while the figure is empty
    assert displayed == []

    env.add(robot, q=robot.qn, readonly=True)
    assert displayed == [env.fig.canvas]

    # a second robot is drawn into the figure already on show
    env.add(robot, q=robot.qr, readonly=True)
    assert displayed == [env.fig.canvas]
    plt.close(env.fig)


def test_a_figure_supplied_by_the_caller_is_not_displayed_by_launch(displayed):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    env = PyPlot()
    env.launch(fig=fig, render_mode="notebook-widget")
    env.add(rtb.models.DH.Puma560(), q=[0, 0, 0, 0, 0, 0], readonly=True)

    assert displayed == []
    plt.close(fig)


def test_colab_replaces_inline_frames_like_jupyterlite(monkeypatch, displayed):
    import types

    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    env = PyPlot()
    env.launch(render_mode="notebook-inline")

    assert env._inline_is_jl is True


def test_other_environments_update_one_inline_display(displayed):
    env = PyPlot()
    env.launch(render_mode="notebook-inline")

    assert env._inline_is_jl is False


def test_colab_is_a_notebook(monkeypatch):
    # Colab's IPython shell is not a ZMQInteractiveShell, so it is recognised by
    # its module; otherwise it gets a window and the same figure is displayed
    # several times
    import types

    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))

    assert pyplot_module._isnotebook() is True


def test_colab_with_ipympl_is_a_widget_for_a_static_plot(monkeypatch):
    import types

    monkeypatch.setitem(sys.modules, "google.colab", types.ModuleType("google.colab"))
    monkeypatch.setattr(matplotlib, "get_backend", lambda: IPYMPL)

    assert _resolve_render_mode(None) == "notebook-widget"
    assert _resolve_render_mode(None, animation=True) == "notebook-inline"


# --- one notebook check for the whole package ---------------------------------


def test_mobile_uses_the_same_notebook_check():
    from roboticstoolbox.tools.plot import isnotebook

    # the package also exports a class called Vehicle, so ask for the module
    vehicle_module = sys.modules["roboticstoolbox.mobile.Vehicle"]

    assert pyplot_module._isnotebook is isnotebook
    assert vehicle_module._isnotebook is isnotebook


def _shell(monkeypatch, name):
    import builtins

    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    shell = type(name, (), {})()
    monkeypatch.setattr(builtins, "get_ipython", lambda: shell, raising=False)


def test_a_jupyter_kernel_is_a_notebook(monkeypatch):
    from roboticstoolbox.tools.plot import isnotebook

    _shell(monkeypatch, "ZMQInteractiveShell")

    assert isnotebook() is True


def test_ipython_in_a_terminal_is_not_a_notebook_even_if_ipykernel_is_loaded(
    monkeypatch,
):
    # the check the mobile module used to have counted a loaded ipykernel
    import types

    from roboticstoolbox.tools.plot import isnotebook

    _shell(monkeypatch, "TerminalInteractiveShell")
    monkeypatch.setitem(sys.modules, "ipykernel", types.ModuleType("ipykernel"))

    assert isnotebook() is False


# --- interactive mode ----------------------------------------------------------
# With the ipympl backend a figure made in a later cell is only displayed if Matplotlib's
# interactive mode is on.  Launching an inline animation used to switch it off and leave
# it off.  These check the state, not the display, which needs a real notebook frontend.


@pytest.fixture
def interactive_state():
    import matplotlib.pyplot as plt

    was = plt.isinteractive()
    plt.close("all")
    yield
    plt.close("all")
    plt.ion() if was else plt.ioff()


@pytest.mark.parametrize("was_interactive", [True, False])
def test_an_inline_launch_leaves_interactive_mode_as_it_was(
    interactive_state, was_interactive
):
    import matplotlib.pyplot as plt

    plt.ion() if was_interactive else plt.ioff()

    env = PyPlot()
    env.launch(render_mode="notebook-inline")

    assert plt.isinteractive() is was_interactive


def test_an_inline_animation_leaves_interactive_mode_on(interactive_state, monkeypatch):
    import matplotlib.pyplot as plt

    # the inline frames go to IPython's display, which is not wanted here
    monkeypatch.setattr(
        pyplot_module, "_ipy_display", lambda frame, display_id=None: None
    )
    monkeypatch.setattr(pyplot_module, "_ipy_clear_output", lambda wait=False: None)
    plt.ion()
    robot = rtb.models.DH.Puma560()
    env = PyPlot()
    env.launch(render_mode="notebook-inline")
    env.add(robot, q=robot.qn, readonly=True)
    for q in rtb.jtraj(robot.qz, robot.qn, 4).q:
        robot.q = q
        env.step(0.0)

    assert plt.isinteractive() is True
    # the inline figure is not registered with pyplot, so it cannot be displayed twice
    assert plt.get_fignums() == []


def test_a_widget_launch_still_turns_interactive_mode_on(interactive_state, displayed):
    import matplotlib.pyplot as plt

    plt.ioff()
    env = PyPlot()
    env.launch(render_mode="notebook-widget")

    assert plt.isinteractive() is True
