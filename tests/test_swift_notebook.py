"""Swift in a notebook, with a real IPython kernel and the real Swift.

In a notebook ``robot.plot`` has to show Swift in a frame of the cell, and not open a
browser window of its own.  Swift then waits for the page in that frame to connect,
which no browser does here, so the test looks for the frame in what the kernel sends,
and then interrupts the cell.

What the frame shows, a robot that moves while the timer advances, was checked in
JupyterLab and Chromium.  ``test_swift_plot_options.py`` checks that a trajectory is
paced.
"""

import shutil

import pytest

pytest.importorskip("ipykernel")
pytest.importorskip("swift")
jupyter_client = pytest.importorskip("jupyter_client")

pytestmark = pytest.mark.timeout(180)

SETUP = """
import webbrowser
opened = []   # a browser window opened by Swift would be recorded here
webbrowser.open_new_tab = lambda *args, **kwargs: opened.append(args) or True
import roboticstoolbox as rtb
panda = rtb.models.URDF.Panda()
"""


def _messages(kc, msg_id, until):
    """The messages of a cell, in order, until ``until(message)`` or the cell ends."""
    messages = []
    while True:
        msg = kc.get_iopub_msg(timeout=120)
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        messages.append(msg)
        if until(msg) or (
            msg["msg_type"] == "status" and msg["content"]["execution_state"] == "idle"
        ):
            return messages


def _is_frame(msg):
    data = msg["content"].get("data", {}) if msg["msg_type"] == "display_data" else {}
    return "<iframe" in data.get("text/html", "")


@pytest.fixture
def kernel(monkeypatch):
    # a browser window that Swift opens by mistake must not open on a developer's screen
    if shutil.which("true"):
        monkeypatch.setenv("BROWSER", "true")
    monkeypatch.delenv("SWIFT_HEADLESS", raising=False)
    km, kc = jupyter_client.manager.start_new_kernel(startup_timeout=120)
    try:
        _messages(kc, kc.execute(SETUP), lambda m: False)
        yield km, kc
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)


@pytest.mark.parametrize("call", ["panda.plot(panda.qr)", "panda.plot(trajectory)"])
def test_swift_is_shown_in_a_frame_of_the_cell(kernel, call):
    km, kc = kernel
    kc.execute("trajectory = rtb.jtraj(panda.qr, panda.qz, 20).q")

    msg_id = kc.execute(call)
    messages = _messages(kc, msg_id, _is_frame)
    frames = [m for m in messages if _is_frame(m)]
    assert frames, "no frame for Swift was shown in the cell"
    assert 'src="http://localhost:' in frames[0]["content"]["data"]["text/html"]

    # stop the cell, which is waiting for a page that will never connect
    km.interrupt_kernel()
    _messages(kc, msg_id, lambda m: False)

    reply = _messages(kc, kc.execute("print(len(opened))"), lambda m: False)
    streams = "".join(m["content"]["text"] for m in reply if m["msg_type"] == "stream")
    assert streams.strip() == "0", "Swift opened a browser window"
