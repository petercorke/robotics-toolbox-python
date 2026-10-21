"""What ``robot.plot`` asks of the Swift backend.

Swift is a web page served by the Python process.  In a notebook it is shown in a frame
of the cell instead of a browser window, and an animation is paced in real time:
unpaced, Swift draws at most 60 frames per second and skips the steps that come sooner,
so a whole trajectory runs in a few milliseconds and only its first frame is drawn.
This was checked in JupyterLab and Chromium, where the frame of the cell shows the
robot moving, with the timer advancing, and no browser window opens.
"""

import importlib
from unittest import mock

import pytest

import roboticstoolbox as rtb
from roboticstoolbox.backends.PyPlot.PyPlot import PyPlot

# the package also exports a class called PyPlot, so ask for the module by name
pyplot_module = importlib.import_module("roboticstoolbox.backends.PyPlot.PyPlot")


class Swift:
    """A stand-in for the Swift backend that records how it was launched."""

    def __init__(self):
        self.launched_with = None
        self.steps = 0

    def launch(self, **kwargs):
        self.launched_with = kwargs

    def add(self, robot, **kwargs):
        return None

    def step(self, dt=0.05):
        self.steps += 1


@pytest.fixture
def robot(monkeypatch):
    robot = rtb.models.DH.Puma560()
    backend = Swift()
    monkeypatch.setattr(
        type(robot), "_get_graphical_backend", lambda self, backend_name=None: backend
    )
    robot.swift = backend
    return robot


def _trajectory(robot):
    return rtb.jtraj(robot.qz, robot.qn, 5).q


def test_notebook_shows_swift_in_the_cell_and_paces_an_animation(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: True)

    robot.plot(_trajectory(robot))

    assert robot.swift.launched_with["browser"] == "notebook"
    assert robot.swift.launched_with["realtime"] is True
    assert robot.swift.steps == 5


def test_one_pose_is_not_paced(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: True)

    robot.plot(robot.qn)

    assert robot.swift.launched_with["browser"] == "notebook"
    assert robot.swift.launched_with["realtime"] is False


def test_outside_a_notebook_swift_has_a_browser_window(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: False)

    robot.plot(_trajectory(robot))

    assert "browser" not in robot.swift.launched_with
    assert robot.swift.launched_with["realtime"] is True


def test_explicit_options_are_used_as_given(robot, monkeypatch):
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: True)

    robot.plot(_trajectory(robot), browser="firefox", realtime=False)

    assert robot.swift.launched_with["browser"] == "firefox"
    assert robot.swift.launched_with["realtime"] is False


def test_the_other_backends_are_not_given_swift_options(monkeypatch):
    robot = rtb.models.DH.Puma560()
    monkeypatch.setattr(pyplot_module, "_isnotebook", lambda: False)

    with mock.patch.object(PyPlot, "launch", autospec=True) as launch:
        try:
            robot.plot(_trajectory(robot), backend="pyplot", dt=0.0)
        except Exception:
            pass  # the figure was not launched, only what was asked for matters

    assert "browser" not in launch.call_args.kwargs
    assert "realtime" not in launch.call_args.kwargs
