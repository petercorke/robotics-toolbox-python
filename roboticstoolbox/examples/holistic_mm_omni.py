#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rtb
import spatialgeometry as sg
import spatialmath as sm
import qpsolvers as qp
import numpy as np
import math


def step_robot(r, Tep):

    wTe = r.fkine(r.q, fast=True)

    eTep = np.linalg.inv(wTe) @ Tep

    # Spatial error
    et = np.sum(np.abs(eTep[:3, -1]))

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(r.n + 6)

    # Joint velocity component of Q
    Q[: r.n, : r.n] *= Y
    Q[:3, :3] *= 1.0 / et

    # Slack component of Q
    Q[r.n :, r.n :] = (1.0 / et) * np.eye(6)

    v, _ = rtb.p_servo(wTe, Tep, 1.5)

    v[3:] *= 1.3

    # The equality contraints
    Aeq = np.c_[r.jacobe(r.q, fast=True), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((r.n + 6, r.n + 6))
    bin = np.zeros(r.n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.1

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[: r.n, : r.n], bin[: r.n] = r.joint_velocity_damper(ps, pi, r.n)

    # Linear component of objective function: the manipulability Jacobian
    c = np.concatenate(
        (np.zeros(3), -r.jacobm(start=r.links[5]).reshape((r.n - 3,)), np.zeros(6))
    )

    # Get base to face end-effector
    kε = 0.5
    bTe = r.fkine(r.q, include_base=False, fast=True)
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[r.qdlim[: r.n], 10 * np.ones(6)]
    ub = np.r_[r.qdlim[: r.n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
    qd = qd[: r.n]

    if et > 0.5:
        qd *= 0.7 / et
    else:
        qd *= 1.4

    if et < 0.02:
        return True, qd
    else:
        return False, qd


env = swift.Swift()
env.launch(realtime=True)

ax_goal = sg.Axes(0.1)
env.add(ax_goal)

frankie = rtb.models.FrankieOmni()
frankie.q = frankie.qr
env.add(frankie)

arrived = False
dt = 0.025

# Behind
env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])
wTep = frankie.fkine(frankie.q) * sm.SE3.Rz(np.pi)
wTep.A[:3, :3] = np.diag([-1, 1, -1])
wTep.A[0, -1] -= 4.0
wTep.A[2, -1] -= 0.25
ax_goal.base = wTep
env.step()


while not arrived:

    arrived, frankie.qd = step_robot(frankie, wTep.A)
    env.step(dt)

    # Reset bases
    base_new = frankie.fkine(frankie._q, end=frankie.links[3], fast=True)
    frankie._base.A[:] = base_new
    frankie.q[:3] = 0

env.hold()
