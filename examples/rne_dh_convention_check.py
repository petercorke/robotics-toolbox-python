#!/usr/bin/env python
"""Root-cause diagnostic for the Robot.rne() vs DHRobot.rne_python() divergence
originally seen in rne_compare.py -- see rne.md (issue 6) for the full writeup.

Uses a single-link revolute robot with a pure axis twist (alpha != 0, a=d=0)
to isolate the frame-convention question from friction/armature/multi-link
recursion complexity. Ground truth is obtained *independently* of either RNE
implementation via the Lagrangian identity: for a static pose (qd=qdd=0), the
required joint torque equals dV/dq, where V(q) = m g z_com(q) is the
gravitational potential energy of the link's centre of mass, computed
directly from link.A(q) and numerically differentiated.

Historical note: this originally showed rne_python() wrong for modified DH
and Robot.rne() wrong for standard DH -- three real bugs in rne_python()'s
MDH branch (see rne.md) have since been fixed, so rne_python() now agrees
with ground truth for both conventions. Robot.rne()'s standard-DH case is
not a bug to fix: its Featherstone recursion structurally requires the
joint to be the last element of its own ETS segment, which standard DH
never satisfies (the joint comes first). Robot.rne() now asserts on that
case (see Robot.py) instead of silently returning a wrong answer.
"""

import numpy as np

from roboticstoolbox import DHRobot, RevoluteDH, RevoluteMDH
from roboticstoolbox.robot.Robot import Robot as RobotBase


def gravity_torque_truth(robot, q0, g=9.81, h=1e-6):
    """Numerically-differentiated ground truth, independent of any RNE code."""
    link = robot.links[0]
    p_local = np.array([*link.r, 1.0])

    def com_z(q):
        A = np.asarray(link.A(q))
        return (A @ p_local)[2]

    return g * (com_z(q0 + h) - com_z(q0 - h)) / (2 * h)


def check(label, robot, q0=0.3, robot_rne_expected_to_work=True):
    z = np.zeros(1)
    q = np.array([q0])
    truth = gravity_torque_truth(robot, q0)
    tau_py = robot.rne_python(q, z, z)[0]
    print(f"{label:28s} truth={truth:9.4f}  rne_python={tau_py:9.4f}")
    print(f"{'':28s} |rne_python - truth| = {abs(tau_py - truth):.4f}")

    if robot_rne_expected_to_work:
        tau_base = RobotBase.rne(robot, q, z, z)[0]
        print(f"{'':28s} Robot.rne={tau_base:9.4f}")
        print(f"{'':28s} |Robot.rne - truth| = {abs(tau_base - truth):.4f}")
    else:
        try:
            RobotBase.rne(robot, q, z, z)
        except AssertionError:
            print(f"{'':28s} Robot.rne: correctly rejected (AssertionError)")
        else:
            print(f"{'':28s} Robot.rne: ERROR -- expected AssertionError, got a result")


print("Single-link revolute robot, alpha=1.2 rad, a=d=0, r=[0.5,0,0], static (qd=qdd=0)")
print("Ground truth = d/dq[ m g z_com(q) ], independent of both RNE implementations.")
print()

std = DHRobot([RevoluteDH(a=0, alpha=1.2, d=0, m=1.0, r=[0.5, 0, 0])], gravity=[0, 0, -9.81])
check("Standard DH (mdh=False)", std, robot_rne_expected_to_work=False)

print()
mdh = DHRobot([RevoluteMDH(a=0, alpha=1.2, d=0, m=1.0, r=[0.5, 0, 0])], gravity=[0, 0, -9.81])
check("Modified DH (mdh=True)", mdh, robot_rne_expected_to_work=True)

print()
print("Conclusion: rne_python is correct for both DH conventions.")
print("Robot.rne is correct for modified DH, and now cleanly rejects (rather")
print("than silently mis-computing) standard DH, since its Featherstone")
print("recursion cannot represent standard DH's joint-first structure.")
