#!/usr/bin/env python
"""
Hand-rolled Swift teach panel: one named slider per joint, driving the
robot via a single per-step callback on the handle env.add_robot()
returns -- the same mechanism roboticstoolbox's own robot.teach(q,
backend="swift") now uses internally (see roboticstoolbox.backends.swift.
Swift._add_teach_panel). Useful as a template for a custom panel beyond
what teach() offers (e.g. extra UI elements alongside the sliders);
for the common case, robot.teach(q, backend="swift") does this in one
call.

Named sliders push their value into env.values; the handle's callback
reads from there and returns the new q each step -- there's no explicit
per-slider setter function, and no direct robot.q/handle.q mutation in
the loop, env.step() drives everything.
"""
import numpy as np
import roboticstoolbox as rtb
from swift import Swift, Slider, Label

# Launch the simulator Swift
env = Swift()
env.launch(ground_opacity=0.3)

# Make a robot and add it to Swift
# robot = rtb.models.UR5()
robot = rtb.models.Panda()

handle = env.add_robot(robot)
handle.q = robot.qr

# compact=True keeps six stacked Labels from taking up excessive
# sidebar space -- Label's default styling is sized for an occasional
# standalone heading, not several stacked close together.
pose_labels = [Label("", compact=True) for _ in range(6)]
for label in pose_labels:
    env.add(label)


def update_pose_labels(q):
    T = robot.fkine(q)
    t = np.round(T.t, 3)
    r = np.round(T.rpy(unit="deg"), 3)
    pose_labels[0].desc = f"x: {t[0]}"
    pose_labels[1].desc = f"y: {t[1]}"
    pose_labels[2].desc = f"z: {t[2]}"
    pose_labels[3].desc = f"r: {r[0]}&#176;"
    pose_labels[4].desc = f"p: {r[1]}&#176;"
    pose_labels[5].desc = f"y: {r[2]}&#176;"


def teach_update(t, values):
    # Sliders display revolute joints in degrees, prismatic in native
    # units (metres) -- toradians() converts the whole vector back in
    # one call, only touching revolute entries.
    q_display = np.array([values[f"q{j}"] for j in range(robot.n)])
    q_new = robot.toradians(q_display)
    update_pose_labels(q_new)
    return q_new


handle.callback = teach_update

# Loop through each joint and add a slider to Swift to control it
qlim = robot.qlim
for j in range(robot.n):
    lo, hi = qlim[0, j], qlim[1, j]
    if robot.isrevolute(j):
        lo_disp, hi_disp, val_disp = np.degrees(lo), np.degrees(hi), np.degrees(handle.q[j])
        step = 1.0
        unit = "&#176;"
    else:
        lo_disp, hi_disp, val_disp = lo, hi, handle.q[j]
        step = (hi - lo) / 100
        unit = "m"

    env.add(
        Slider(
            lambda x: None,
            # min/max/value stay full precision -- precision= below only
            # rounds the *displayed* text, so the slider's actual driven
            # value doesn't lose precision to display rounding.
            min=float(lo_disp),
            max=float(hi_disp),
            step=step,
            value=float(val_disp),
            desc=f"{robot.name} joint {j}",
            unit=unit,
            precision=2,
        ),
        name=f"q{j}",
    )

update_pose_labels(handle.q)

env.run()
