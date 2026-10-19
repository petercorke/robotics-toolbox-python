"""Animating a robot in a notebook, with a real IPython kernel.

A trajectory plotted in a notebook is an animation: the frames are shown while the
cell that asked for it is running.  An ``ipympl`` figure cannot do that.  It updates
only when the kernel handles a message sent by the browser, which it cannot do in a
running cell, so only the first and the last frame ever appeared (blank figure, long
pause, final frame).  The Toolbox therefore pushes each frame of an animation as a
message of its own, whatever ``%matplotlib`` says, and these tests check that those
messages leave the kernel while the cell runs.

The browser is not part of the test.  Before this was fixed the frames were
checked in JupyterLab and Chromium, which showed one image during the animation, and
13 of 13 different ones with the frames pushed as they are here.
"""

import hashlib

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("ipympl")
jupyter_client = pytest.importorskip("jupyter_client")

NFRAMES = 6
WIDGET = "application/vnd.jupyter.widget-view+json"
SVG = "image/svg+xml"

SETUP = """
%matplotlib {backend}
import roboticstoolbox as rtb
robot = rtb.models.DH.Puma560()
trajectory = rtb.jtraj(robot.qz, robot.qn, {n}).q
"""

pytestmark = pytest.mark.timeout(180)


def _run(kc, code):
    """Run a cell and return the messages it produced, in order."""
    msg_id = kc.execute(code)
    messages = []
    while True:
        msg = kc.get_iopub_msg(timeout=120)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        if msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle":
            return messages
        messages.append(msg)


@pytest.fixture
def kernel():
    km, kc = jupyter_client.manager.start_new_kernel(startup_timeout=120)
    try:
        yield kc
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


def _frames(messages):
    """The SVG frames pushed into the cell output, whether new or an update."""
    return [
        m["content"]["data"][SVG]
        for m in messages
        if m["msg_type"] in ("display_data", "update_display_data")
        and SVG in m["content"]["data"]
    ]


def _errors(messages):
    return [m["content"] for m in messages if m["msg_type"] == "error"]


def _plot(kc, backend, call):
    assert not _errors(_run(kc, SETUP.format(backend=backend, n=NFRAMES)))
    messages = _run(kc, call)
    assert not _errors(messages), _errors(messages)
    return messages


@pytest.mark.parametrize("backend", ["ipympl", "inline"])
def test_a_trajectory_is_animated_in_the_running_cell(kernel, backend):
    messages = _plot(kernel, backend, "robot.plot(trajectory, dt=0.0)")

    frames = _frames(messages)
    # a frame for every step, and every one different (the pose and the time change)
    assert len(frames) >= NFRAMES
    assert len({hashlib.md5(f.encode()).hexdigest() for f in frames}) >= NFRAMES


def test_one_pose_is_an_interactive_widget_under_ipympl(kernel):
    messages = _plot(kernel, "ipympl", "robot.plot(robot.qn)")

    shown = [m for m in messages if m["msg_type"] == "display_data"]
    assert any(WIDGET in m["content"]["data"] for m in shown)
    assert not _frames(messages)


def test_an_explicit_render_mode_is_used_for_a_trajectory(kernel):
    messages = _plot(
        kernel,
        "ipympl",
        "robot.plot(trajectory, dt=0.0, render_mode='notebook-widget')",
    )

    assert not _frames(messages)
