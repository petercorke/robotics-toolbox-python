#!/usr/bin/env python
"""
@author Kerry He and Rhys Newbury
"""

import swift
import spatialgeometry as sg
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp
import math


def transform_between_vectors(a, b):
    # Finds the shortest rotation between two vectors using angle-axis,
    # then outputs it as a 4x4 transformation matrix
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    angle = np.arccos(np.dot(a, b))
    axis = np.cross(a, b)

    return sm.SE3.AngleAxis(angle, axis)


# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a Fetch and Camera robot object
fetch = rtb.models.Fetch()
fetch_camera = rtb.models.FetchCamera()

# Set joint angles to zero configuration
fetch.q = fetch.qz
fetch_camera.q = fetch_camera.qz

# Make target object obstacles with velocities
target = sg.Sphere(radius=0.05, base=sm.SE3(-2.0, 0.0, 0.5))

# Make line of sight object to visualize where the camera is looking
sight_base = sm.SE3.Ry(np.pi / 2) * sm.SE3(0.0, 0.0, 2.5)
centroid_sight = sg.Cylinder(
    radius=0.001,
    length=5.0,
    base=fetch_camera.fkine(fetch_camera.q).A @ sight_base.A,
)

# Add the Fetch and other shapes to the simulator
env.add(fetch)
env.add(fetch_camera)
env.add(centroid_sight)
env.add(target)

# Set the desired end-effector pose to the location of target
Tep = fetch.fkine(fetch.q)
Tep.A[:3, :3] = sm.SE3.Rz(np.pi).R
Tep.A[:3, 3] = target.T[:3, -1]

env.step()

n_base = 2
n_arm = 8
n_camera = 2
n = n_base + n_arm + n_camera


def step():

    # Find end-effector pose in world frame
    wTe = fetch.fkine(fetch.q).A
    # Find camera pose in world frame
    wTc = fetch_camera.fkine(fetch_camera.q).A

    # Find transform between end-effector and goal
    eTep = np.linalg.inv(wTe) @ Tep.A
    # Find transform between camera and goal
    cTep = np.linalg.inv(wTc) @ Tep.A

    # Spatial error between end-effector and target
    et = np.sum(np.abs(eTep[:3, -1]))

    # Weighting function used for objective function
    def w_lambda(et, alpha, gamma):
        return alpha * np.power(et, gamma)

    # Quadratic component of objective function
    Q = np.eye(n + 10)

    Q[: n_base + n_arm, : n_base + n_arm] *= 0.01  # Robotic manipulator
    Q[:n_base, :n_base] *= w_lambda(et, 1.0, -1.0)  # Mobile base
    Q[n_base + n_arm : n, n_base + n_arm : n] *= 0.01  # Camera
    Q[n : n + 3, n : n + 3] *= w_lambda(et, 1000.0, -2.0)  # Slack arm linear
    Q[n + 3 : n + 6, n + 3 : n + 6] *= w_lambda(et, 0.01, -5.0)  # Slack arm angular
    Q[n + 6 : -1, n + 6 : -1] *= 100  # Slack camera
    Q[-1, -1] *= w_lambda(et, 1000.0, 3.0)  # Slack self-occlusion

    # Calculate target velocities for end-effector to reach target
    v_manip, _ = rtb.p_servo(wTe, Tep, 1.5)
    v_manip[3:] *= 1.3

    # Calculate target angular velocity for camera to rotate towards target
    head_rotation = transform_between_vectors(np.array([1, 0, 0]), cTep[:3, 3])
    v_camera, _ = rtb.p_servo(sm.SE3(), head_rotation, 25)

    # The equality contraints to achieve velocity targets
    Aeq = np.c_[fetch.jacobe(fetch.q), np.zeros((6, 2)), np.eye(6), np.zeros((6, 4))]
    beq = v_manip.reshape((6,))

    jacobe_cam = fetch_camera.jacobe(fetch_camera.q)[3:, :]
    Aeq_cam = np.c_[
        jacobe_cam[:, :3],
        np.zeros((3, 7)),
        jacobe_cam[:, 3:],
        np.zeros((3, 6)),
        np.eye(3),
        np.zeros((3, 1)),
    ]
    Aeq = np.r_[Aeq, Aeq_cam]
    beq = np.r_[beq, v_camera[3:].reshape((3,))]

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 10, n + 10))
    bin = np.zeros(n + 10)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.1

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[: fetch.n, : fetch.n], bin[: fetch.n] = fetch.joint_velocity_damper(
        ps, pi, fetch.n
    )

    Ain_torso, bin_torso = fetch_camera.joint_velocity_damper(0.0, 0.05, fetch_camera.n)
    Ain[2, 2] = Ain_torso[2, 2]
    bin[2] = bin_torso[2]

    Ain_cam, bin_cam = fetch_camera.joint_velocity_damper(ps, pi, fetch_camera.n)
    Ain[n_base + n_arm : n, n_base + n_arm : n] = Ain_cam[3:, 3:]
    bin[n_base + n_arm : n] = bin_cam[3:]

    # Create line of sight object between camera and object
    c_Ain, c_bin = fetch.vision_collision_damper(
        target,
        camera=fetch_camera,
        camera_n=2,
        q=fetch.q[: fetch.n],
        di=0.3,
        ds=0.2,
        xi=1.0,
        end=fetch.link_dict["gripper_link"],
        start=fetch.link_dict["shoulder_pan_link"],
    )

    if c_Ain is not None and c_bin is not None:
        c_Ain = np.c_[
            c_Ain, np.zeros((c_Ain.shape[0], 9)), -np.ones((c_Ain.shape[0], 1))
        ]

        Ain = np.r_[Ain, c_Ain]
        bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.concatenate(
        (
            np.zeros(n_base),
            # -fetch.jacobm(start=fetch.links[3]).reshape((n_arm,)),
            np.zeros(n_arm),
            np.zeros(n_camera),
            np.zeros(10),
        )
    )

    # Get base to face end-effector
    kε = 0.5
    bTe = fetch.fkine(fetch.q, include_base=False).A
    θε = math.atan2(bTe[1, -1], bTe[0, -1])
    ε = kε * θε
    c[0] = -ε

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[
        fetch.qdlim[: fetch.n],
        fetch_camera.qdlim[3 : fetch_camera.n],
        100 * np.ones(9),
        0,
    ]
    ub = np.r_[
        fetch.qdlim[: fetch.n],
        fetch_camera.qdlim[3 : fetch_camera.n],
        100 * np.ones(9),
        100,
    ]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)
    qd_cam = np.concatenate((qd[:3], qd[fetch.n : fetch.n + 2]))
    qd = qd[: fetch.n]

    if et > 0.5:
        qd *= 0.7 / et
        qd_cam *= 0.7 / et
    else:
        qd *= 1.4
        qd_cam *= 1.4

    arrived = et < 0.02

    fetch.qd = qd
    fetch_camera.qd = qd_cam
    centroid_sight.T = fetch_camera.fkine(fetch_camera.q).A @ sight_base.A

    return arrived


arrived = False
while not arrived:
    arrived = step()
    env.step(0.01)
