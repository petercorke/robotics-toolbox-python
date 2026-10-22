"""What ``robot.plot`` asks of the Swift backend.

Swift is a web page served by the Python process.  In a notebook it is shown in a frame
of the cell instead of a browser window, and an animation is paced in real time:
unpaced, Swift draws at most 60 frames per second and skips the steps that come sooner,
so a whole trajectory runs in a few milliseconds and only its first frame is drawn.
This was checked in JupyterLab and Chromium, where the frame of the cell shows the
robot moving, with the timer advancing, and no browser window opens.
"""

import importlib
import types
from unittest import mock

import numpy.testing as nt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox.backends.PyPlot.PyPlot import PyPlot

# the package also exports a class called PyPlot, so ask for the module by name
pyplot_module = importlib.import_module("roboticstoolbox.backends.PyPlot.PyPlot")


class Swift:
    """A stand-in for Swift 2.1, which has launch(notebook=...)."""

    def __init__(self):
        self.launched_with = None
        self.steps = 0

    def launch(self, notebook=False, **kwargs):
        self.launched_with = dict(kwargs, **({"notebook": True} if notebook else {}))

    def add(self, robot, **kwargs):
        # what the real Swift returns: a handle that owns the joint state it shows
        self.handle = types.SimpleNamespace(q=None)
        return self.handle

    def step(self, dt=0.05):
        self.steps += 1


class OldSwift(Swift):
    """A stand-in for Swift 2.0, where the notebook frame was browser="notebook"."""

    def launch(self, **kwargs):
        self.launched_with = kwargs


@pytest.fixture(params=[Swift, OldSwift], ids=["swift-2.1", "swift-2.0"])
def robot(request, monkeypatch):
    robot = rtb.models.DH.Puma560()
    backend = request.param()
    # the class is named Swift, as plot() recognises the backend by it
    type(backend).__name__ = "Swift"
    monkeypatch.setattr(
        type(robot), "_get_graphical_backend", lambda self, backend_name=None: backend
    )
    robot.swift = backend
    return robot


def _trajectory(robot):
    return rtb.jtraj(robot.qz, robot.qn, 5).q


def _in_the_frame(robot):
    """Was the notebook frame asked for, in the way the Swift in use understands?"""
    launched = robot.swift.launched_with
    if isinstance(robot.swift, OldSwift):
        return launched.get("browser") == "notebook"
    return launched.get("notebook") is True and "browser" not in launched


def _in_a_window(robot):
    launched = robot.swift.launched_with
    return "notebook" not in launched and launched.get("browser") != "notebook"


def test_notebook_shows_swift_in_the_cell_and_paces_an_animation(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: True)
    trajectory = _trajectory(robot)

    robot.plot(trajectory)

    assert _in_the_frame(robot)
    assert robot.swift.launched_with["realtime"] is True
    assert robot.swift.steps == 5
    # the handle was driven, not robot.q (deprecated by Swift), and robot.q ends at the
    # last pose as it does for the other backends
    nt.assert_allclose(robot.swift.handle.q, trajectory[-1])
    nt.assert_allclose(robot.q, trajectory[-1])


def test_one_pose_is_not_paced(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: True)

    robot.plot(robot.qn)

    assert _in_the_frame(robot)
    assert robot.swift.launched_with["realtime"] is False


def test_outside_a_notebook_swift_has_a_browser_window(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: False)

    robot.plot(_trajectory(robot))

    assert _in_a_window(robot)
    assert robot.swift.launched_with["realtime"] is True


def test_notebook_false_gives_a_window_even_in_a_notebook(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: True)

    robot.plot(robot.qn, notebook=False)

    assert _in_a_window(robot)


def test_notebook_true_gives_the_frame_even_in_a_script(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: False)

    robot.plot(robot.qn, notebook=True)

    assert _in_the_frame(robot)


def test_naming_a_browser_gives_a_window_even_in_a_notebook(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: True)

    robot.plot(robot.qn, browser="firefox")

    assert robot.swift.launched_with["browser"] == "firefox"
    assert "notebook" not in robot.swift.launched_with


def test_a_browser_and_the_frame_cannot_both_be_asked_for(robot):
    with pytest.raises(ValueError, match="notebook=True"):
        robot.plot(robot.qn, notebook=True, browser="firefox")


def test_browser_notebook_is_deprecated_but_still_works(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: False)

    with pytest.warns(FutureWarning, match="notebook=True"):
        robot.plot(robot.qn, browser="notebook")

    assert _in_the_frame(robot)


def test_realtime_can_be_turned_off(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: False)

    robot.plot(_trajectory(robot), realtime=False)

    assert robot.swift.launched_with["realtime"] is False


def test_the_other_backends_are_not_given_swift_options(monkeypatch):
    robot = rtb.models.DH.Puma560()
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: False)

    with mock.patch.object(PyPlot, "launch", autospec=True) as launch:
        try:
            robot.plot(_trajectory(robot), backend="pyplot", dt=0.0)
        except Exception:
            pass  # the figure was not launched, only what was asked for matters

    for name in ("browser", "notebook", "realtime"):
        assert name not in launch.call_args.kwargs


# --- the real Swift, with no browser (headless) -------------------------------------

swift = pytest.importorskip("swift")


def test_a_trajectory_takes_as_long_as_it_says(monkeypatch):
    """Unpaced, Swift skips every step that comes less than 1/60 s after the last one
    that it drew, so the whole trajectory ran in milliseconds and was never seen.
    Headless Swift still paces its steps, so the time taken shows that plot() did."""
    import time

    monkeypatch.setenv("SWIFT_HEADLESS", "1")
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: False)
    robot = rtb.models.URDF.Panda()
    dt = 0.05
    trajectory = rtb.jtraj(robot.qr, robot.qz, 10).q

    start = time.monotonic()
    env = robot.plot(trajectory, backend="swift", dt=dt)
    elapsed = time.monotonic() - start
    env.close()

    assert elapsed >= 0.8 * 10 * dt, f"10 steps of {dt} s took only {elapsed:.3f} s"


def test_one_pose_is_not_slowed_down(monkeypatch):
    import time

    monkeypatch.setenv("SWIFT_HEADLESS", "1")
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: False)
    robot = rtb.models.URDF.Panda()

    start = time.monotonic()
    env = robot.plot(robot.qr, backend="swift")
    elapsed = time.monotonic() - start
    env.close()

    assert elapsed < 3
