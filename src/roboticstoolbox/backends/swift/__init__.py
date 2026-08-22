import numpy as np

from swift.SwiftRoute import SwiftServer, SwiftSocket, start_servers
from swift.SwiftElement import (
    SwiftElement,
    Slider,
    Select,
    Checkbox,
    Radio,
    Button,
    Label,
)
from swift.Swift import Swift as _SwiftBase


class Swift(_SwiftBase):
    """Swift backend with RTB capability flags."""

    supports_teach: bool = True
    supports_ellipse: bool = False

    def _add_teach_panel(self, robot, q, handle, block):
        """
        Add a joint-slider teach panel plus a live end-effector pose
        readout, and wire the sliders to drive ``handle`` (the
        AssemblyHandle ``BaseRobot.teach()`` got back from ``self.add()``)
        via a single per-step callback -- see jhavl/swift#85 for why
        driving the robot model's own .q/.qd directly is deprecated.

        :param robot: the robot being taught, already added to this scene
        :param q: initial joint configuration to seed the panel/display
            with -- may differ from ``robot.q``'s current value
        :param handle: the AssemblyHandle ``self.add(robot, readonly=True)``
            returned for this robot instance
        :param block: unlike PyPlot (whose own env.hold() enters
            matplotlib's GUI mainloop, which processes slider events on
            its own), Swift's hold() only sleeps and polls for a
            disconnect -- it never calls step(), so nothing would ever
            notice a dragged slider without an active step() loop
            running somewhere. When block, this method runs that loop
            itself (self.run(), blocking until disconnect/^C) rather
            than relying on teach()'s own later `if block: env.hold()`.
            When not block, a single self.step() seeds the initial
            display and the caller is responsible for stepping (matches
            teach(block=False)'s existing contract for other backends).
        :returns: True if block, signalling to teach()'s shared code
            that this method already fully handled blocking itself --
            teach()'s own `if block: env.hold()` must be skipped in that
            case, not just as an optimisation: calling env.hold() a
            second time here can hang outright (its disconnect-poll
            never expires in headless mode, and even non-headless
            there's a race around close() and socket.USERS). None
            (falsy) when not block, matching PyPlot/PyPlot2's own
            _add_teach_panel, which never blocks internally at all.
        """
        qlim = robot.qlim

        # One label per value (x/y/z/r/p/y), matching PyPlot's own six
        # separate fig.text() calls -- compact=True keeps this from
        # taking up excessive sidebar space (Label's default styling is
        # sized for an occasional standalone heading, not several
        # stacked close together -- see swift's Label(compact=) docstring).
        pose_labels = [Label("", compact=True) for _ in range(6)]
        for label in pose_labels:
            self.add(label)

        def update_pose_labels(qv):
            T = robot.fkine(qv)
            t = np.round(T.t, 3)
            r = np.round(T.rpy(unit="deg"), 3)
            pose_labels[0].label = f"x: {t[0]}"
            pose_labels[1].label = f"y: {t[1]}"
            pose_labels[2].label = f"z: {t[2]}"
            pose_labels[3].label = f"r: {r[0]}&#176;"
            pose_labels[4].label = f"p: {r[1]}&#176;"
            pose_labels[5].label = f"y: {r[2]}&#176;"

        def teach_update(t, values):
            # Sliders display revolute joints in degrees, prismatic in
            # native units (metres) -- toradians() converts the whole
            # vector back in one call, only touching revolute entries.
            q_display = np.array([values[f"q{j}"] for j in range(robot.n)])
            q_new = robot.toradians(q_display)
            update_pose_labels(q_new)
            return q_new

        # Safe with readonly=True: Swift.step()'s callback branch runs
        # unconditionally whenever a callback is set -- readonly only
        # gates the *other* (non-callback) per-step update path.
        handle.callback = teach_update

        for j in range(robot.n):
            lo, hi = qlim[0, j], qlim[1, j]
            if robot.isrevolute(j):
                lo_disp, hi_disp, val_disp = np.degrees(lo), np.degrees(hi), np.degrees(q[j])
                step = 1.0
                unit = "&#176;"
            else:
                lo_disp, hi_disp, val_disp = lo, hi, q[j]
                step = (hi - lo) / 100
                unit = "m"

            self.add(
                Slider(
                    lambda x: None,
                    # min/max/value stay full precision -- precision=
                    # below only rounds the *displayed* text, unlike a
                    # naive round() here which would bake rounding error
                    # into the actual driven value once the callback
                    # reads it back from env.values.
                    min=float(lo_disp),
                    max=float(hi_disp),
                    step=step,
                    value=float(val_disp),
                    label=f"{robot.name} joint {j}",
                    unit=unit,
                    precision=2,
                ),
                name=f"q{j}",
            )

        update_pose_labels(q)
        self.step()

        if block:
            # Unbounded (duration=None): keep responding to slider drags
            # for as long as the browser stays connected, same as any
            # other Swift script's own `while True: env.step(dt)` loop --
            # just with run()'s disconnect-awareness (and ^C handling)
            # folded in instead of looping forever after the tab is gone.
            # Returns normally on a disconnect (graceful or mid-step);
            # only ^C skips the rest of this method (raises SystemExit).
            self.run()

            # PyPlot's teach() mutates robot.q throughout its own session,
            # so a caller naturally finds the final taught pose in
            # robot.q once teach() returns -- see BaseRobot.teach()'s
            # docstring. Swift's AssemblyHandle deliberately doesn't
            # mirror handle.q into robot.q during the session (that's
            # the whole point of the refactor -- see jhavl/swift#85), but
            # write it back once, here, at the point the session ends,
            # so callers see the same thing regardless of backend. A
            # deliberate one-time exception for this specific
            # single-owner interactive session, not a general precedent
            # -- the same "stateless over stateful" tension PyPlot's own
            # teach() already has (see desiderata.md), just accepted
            # here rather than solved.
            robot.q = handle.q.copy()

            return True


__all__ = [
    "Swift",
    "Slider",
    "SwiftElement",
    "Label",
    "Select",
    "Button",
    "Checkbox",
    "Radio",
    "SwiftServer",
    "SwiftSocket",
    "start_servers",
]
