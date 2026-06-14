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
from pathlib import Path

# ============================================================================
# TUNABLE CONTROL PARAMETERS
# ============================================================================

# Robot model selection
# Set to either "RbKairosPlus" or "RbRoboutPlus"
ROBOT_MODEL = "RbKairosPlus"

REPO_ROOT = Path(__file__).resolve().parents[2]
XACRO_BASE_PATH = REPO_ROOT / "rtb-data" / "rtbdata" / "xacro"
MODEL_XACRO_PATHS = {
    "RbKairosPlus": XACRO_BASE_PATH / "robotnik_rbkairos_plus_urdf",
    "RbRoboutPlus": XACRO_BASE_PATH / "robotnik_rbrobout_plus_urdf",
}

# Arm Convergence (Primary Tuning)
SERVO_GAIN = 4.0                    # Servo gain for arm (only) p_servo; higher = faster response
REGULARIZATION_Y = 0.005            # Joint velocity regularization; lower = more aggressive
SLACK_PENALTY = 1.0                 # Slack variable penalty multiplier (2.0/et)
ARRIVAL_THRESHOLD = 0.001           # Spatial error threshold for convergence (meters)

# Joint Limits & Safety
JOINT_LIMIT_MARGIN = 0.01            # Minimum margin from joint limits (radians)
JOINT_DAMPER_INFLUENCE = 0.9        # Influence angle for velocity damper (radians)

# Arm Posture (Self-Collision Avoidance)
POSTURE_REFERENCE = np.array([0.0, -1.2, 1.8, -1.6, -1.57, 0.0])  # Reference posture for corridor
POSTURE_BIAS = 0.0                  # Strength of posture bias (0 = off, 1 = strong)

# Base Motion Control
if ROBOT_MODEL == "RbKairosPlus":
    BASE_KEEP_OUT_RADIUS = 0.65         # RB-KAIROS # Keep-out radius for mobile base (meters)
elif ROBOT_MODEL == "RbRoboutPlus":
    BASE_KEEP_OUT_RADIUS = 1.45         # RB-ROBOUT # Keep-out radius for mobile base (meters)
BASE_VELOCITY_GAIN = 0.4            # Translation velocity gain (kv)
BASE_ROTATION_GAIN = 0.6            # Rotation (yaw) velocity gain (kw)
BASE_MAX_VELOCITY = 0.5             # Maximum base linear velocity (m/s)
BASE_MAX_ROTATION = 1.0             # Maximum base rotational velocity (rad/s)

# Mecanum Wheel Geometry
WHEEL_RADIUS = 0.127                # Mecanum wheel radius (meters)
WHEEL_TRACK_X = 0.21528             # Wheelbase length (lx)
WHEEL_TRACK_Y = 0.2590              # Wheelbase width (ly)
WHEEL_MAX_SPEED = 0.5               # Maximum wheel speed command (rad/s)

# Error Scaling
ERROR_SCALE_FAR_THRESHOLD = 2.8     # Distance threshold for velocity scaling
ERROR_SCALE_FAR = 1.4               # Velocity scale when error > threshold
ERROR_SCALE_CLOSE = 1.4             # Velocity scale when error <= threshold
SLACK_VARIABLE_BOUND = 10.0         # Upper bound on slack variables

# End-Effector Configuration
ARM_EE_LINK = "arm_ee_link"         # Arm end-effector link name
ARM_ACTIVATION_RADIUS = 0.8         # Arm starts moving when base is within this distance of target (meters)
ARM_ACTIVATION_BAND = 0.50          # Smooth activation band width (meters)

# Starting Configuration
ARM_STARTING_POSTURE = np.array([0.0, -2.4125, 2.4125, -1.57, -1.57, 0.0])  # Initial arm joint angles (radians)
WHEELS_USE_REST_CONFIG = True       # If True, wheels start from rest config (qr); if False, use zeros

# Arm Joint Limits for Commanded Motion
ARM_MAX_VELOCITY = np.array([0.6, 0.6, 0.6, 0.8, 0.8, 1.0])      # Max arm joint speed (rad/s)
ARM_MAX_ACCELERATION = np.array([1.2, 1.2, 1.2, 1.6, 1.6, 2.0])  # Max arm joint accel (rad/s^2)

# Internal state used by arm acceleration limiter.
ARM_QD_PREV = None

# ============================================================================


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def step_robot(r: rtb.ERobot, Tep, dt):
    global ARM_QD_PREV

    # Use the arm end-effector explicitly
    wTe = r.fkine(r.q, end=ARM_EE_LINK)

    eTep = np.linalg.inv(wTe) @ Tep

    # Spatial error
    et = np.sum(np.abs(eTep[:3, -1]))

    arm_joint_indices = sorted(
        [
            link.jindex
            for link in r.links
            if link.isjoint and link.name.startswith("arm_") and link.jindex is not None
        ]
    )

    # Get wheel indices early to compute dist_goal and determine arm activation
    wheel_names = [
        "front_right_wheel_link",
        "front_left_wheel_link",
        "back_left_wheel_link",
        "back_right_wheel_link",
    ]
    wheel_indices = []
    for name in wheel_names:
        link = next((l for l in r.links if l.name == name), None)
        if link is not None and link.jindex is not None:
            wheel_indices.append(int(link.jindex))

    # Compute dist_goal early to gate servo BEFORE QP
    dist_goal = 0.0
    if len(wheel_indices) == 4:
        wTb = r.fkine(r.q, end="base_link").A
        p_err_w = Tep[:2, 3] - wTb[:2, 3]
        dist_goal = np.linalg.norm(p_err_w)

    # Use remaining drive distance (beyond base keep-out radius) for arm activation.
    activation_distance = dist_goal
    if len(wheel_indices) == 4:
        activation_distance = max(dist_goal - BASE_KEEP_OUT_RADIUS, 0.0)

    # Smooth activation in [0, 1] to avoid abrupt arm enable/disable switching.
    if ARM_ACTIVATION_BAND > 1e-9:
        arm_activation = np.clip(
            (ARM_ACTIVATION_RADIUS - activation_distance) / ARM_ACTIVATION_BAND,
            0.0,
            1.0,
        )
    else:
        arm_activation = 1.0 if activation_distance <= ARM_ACTIVATION_RADIUS else 0.0
    arm_in_position = arm_activation > 1e-3

    # Compute servo command and smoothly gate arm task contribution
    v, _ = rtb.p_servo(wTe, Tep, SERVO_GAIN)
    v[3:] *= 1.3
    v *= arm_activation

    # print(f"dist_goal: {dist_goal:.4f}, arm_in_position: {arm_in_position}, v_norm: {np.linalg.norm(v):.6f}")

    # Quadratic component of objective function
    Q = np.eye(r.n + 6)

    # Joint velocity component of Q
    Q[: r.n, : r.n] *= REGULARIZATION_Y
    Q[:3, :3] *= 1.0 / et

    # Slack component of Q
    slack_penalty = SLACK_PENALTY / et
    Q[r.n :, r.n :] = slack_penalty * np.eye(6)

    # The equality constraints. Map an arm-chain Jacobian into the full joint space.
    J_arm = r.jacobe(r.q, end=ARM_EE_LINK)
    if J_arm.shape[1] == r.n:
        J_full = J_arm
    else:
        J_full = np.zeros((6, r.n))
        if len(arm_joint_indices) == J_arm.shape[1]:
            J_full[:, arm_joint_indices] = J_arm
        else:
            # Fallback for unexpected models: keep dimensions consistent.
            J_full[:, : J_arm.shape[1]] = J_arm

    Aeq = np.c_[arm_activation * J_full, np.eye(6)]
    beq = (arm_activation * v).reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((r.n + 6, r.n + 6))
    bin = np.zeros(r.n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = JOINT_LIMIT_MARGIN

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = JOINT_DAMPER_INFLUENCE

    # Form the joint limit velocity damper
    Ain[: r.n, : r.n], bin[: r.n] = r.joint_velocity_damper(ps, pi, r.n)

    # Linear component of objective function: the manipulability Jacobian.
    # Some models return jacobm for all joints (virtual-base ETS), others only for
    # the arm chain (URDF). Fill the objective robustly without hardcoded reshape.
    c = np.zeros(r.n + 6)
    jm = np.asarray(r.jacobm(end=ARM_EE_LINK)).reshape(-1)
    if jm.size == r.n:
        c[:r.n] = -jm
    elif len(arm_joint_indices) == jm.size:
        c[arm_joint_indices] = -jm
    else:
        c[: jm.size] = -jm

    # Fade manipulability objective for arm joints with activation.
    if len(arm_joint_indices) > 0:
        c[arm_joint_indices] *= arm_activation

    # Add base-heading objective only for the virtual-base model where the first
    # joint controls base yaw.
    if jm.size == r.n and r.n >= 3:
        kε = 0.5
        bTe = r.fkine(r.q, include_base=False, end=ARM_EE_LINK).A
        θε = math.atan2(bTe[1, -1], bTe[0, -1])
        ε = kε * θε
        c[0] = -ε

    # The lower and upper bounds on the joint velocity and slack variable.
    # URDF model qdlim length can be smaller than r.n, so pad safely.
    qdlim = np.asarray(r.qdlim).reshape(-1)
    if qdlim.size < r.n:
        pad_val = np.max(qdlim) if qdlim.size > 0 else 1.0
        qdlim = np.pad(qdlim, (0, r.n - qdlim.size), constant_values=pad_val)
    else:
        qdlim = qdlim[: r.n]

    lb = -np.r_[qdlim, SLACK_VARIABLE_BOUND * np.ones(6)]
    ub = np.r_[qdlim, SLACK_VARIABLE_BOUND * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub, solver='quadprog')
    if qd is None:
        qd = np.zeros(r.n + 6)
    qd = qd[: r.n]

    if len(wheel_indices) == 4:
        # Base pose and planar error to the end-effector goal.
        wTb = r.fkine(r.q, end="base_link").A
        p_err_w = Tep[:2, 3] - wTb[:2, 3]
        Rwb = wTb[:2, :2]

        # Keep-out radius: the mobile base supports the reach, but must not
        # drive all the way to the arm end-effector target.
        d_stop = BASE_KEEP_OUT_RADIUS
        if dist_goal > 1e-6:
            dir_w = p_err_w / dist_goal
        else:
            dir_w = np.zeros(2)

        # Only command translation for the distance outside the keep-out radius.
        dist_drive = max(dist_goal - d_stop, 0.0)
        v_base_w = BASE_MAX_VELOCITY * dist_drive * dir_w
        v_base_w = np.clip(v_base_w, -BASE_MAX_VELOCITY, BASE_MAX_VELOCITY)
        p_err_b = Rwb.T @ v_base_w

        # Body-frame velocity command for omnidirectional base.
        kv = BASE_VELOCITY_GAIN
        kw = BASE_ROTATION_GAIN
        vx = np.clip(kv * p_err_b[0], -BASE_MAX_VELOCITY, BASE_MAX_VELOCITY)
        vy = np.clip(kv * p_err_b[1], -BASE_MAX_VELOCITY, BASE_MAX_VELOCITY)

        # Enforce hard stop for base translation inside keep-out zone.
        if dist_goal <= d_stop:
            vx = 0.0
            vy = 0.0

        theta_target = math.atan2(p_err_w[1], p_err_w[0])
        theta_base = math.atan2(wTb[1, 0], wTb[0, 0])
        wz = np.clip(kw * wrap_to_pi(theta_target - theta_base), -BASE_MAX_ROTATION, BASE_MAX_ROTATION)

        # Mecanum inverse kinematics (Robotnik wheel geometry from xacro).
        wheel_r = WHEEL_RADIUS
        lx = WHEEL_TRACK_X
        ly = WHEEL_TRACK_Y
        a = lx + ly
        M = (1.0 / wheel_r) * np.array(
            [
                [1.0, -1.0, -a],  # front right
                [1.0, 1.0, a],     # front left
                [1.0, -1.0, a],    # back left
                [1.0, 1.0, -a],    # back right
            ]
        )

        qd_wheels = M @ np.array([vx, vy, wz])
        qd[wheel_indices] = np.clip(qd_wheels, -WHEEL_MAX_SPEED, WHEEL_MAX_SPEED)

        # Integrate commanded base twist to move the floating base transform.
        T = np.array(r._T, dtype=float)
        theta = math.atan2(T[1, 0], T[0, 0])
        cth = math.cos(theta)
        sth = math.sin(theta)
        vx_w = cth * vx - sth * vy
        vy_w = sth * vx + cth * vy

        x_new = T[0, 3] + vx_w * dt
        y_new = T[1, 3] + vy_w * dt
        theta_new = wrap_to_pi(theta + wz * dt)

        T_new = np.eye(4)
        T_new[0, 0] = math.cos(theta_new)
        T_new[0, 1] = -math.sin(theta_new)
        T_new[1, 0] = math.sin(theta_new)
        T_new[1, 1] = math.cos(theta_new)
        T_new[2, 2] = 1.0
        T_new[0, 3] = x_new
        T_new[1, 3] = y_new
        T_new[2, 3] = T[2, 3]
        r._T = T_new

    # Keep arm in a collision-safe posture corridor to avoid intersecting the base (only when active).
    if arm_in_position and len(arm_joint_indices) == 6:
        q_ref_arm = POSTURE_REFERENCE
        k_posture = POSTURE_BIAS
        qd[arm_joint_indices] += (arm_activation * k_posture) * (q_ref_arm - r.q[arm_joint_indices])

    if et > ERROR_SCALE_FAR_THRESHOLD:
        qd *= ERROR_SCALE_FAR / et
    else:
        qd *= ERROR_SCALE_CLOSE

    # Enforce joint speed limits after adding posture/base terms.
    qd = np.clip(qd, -qdlim, qdlim)

    # Apply explicit arm velocity and acceleration limits.
    if len(arm_joint_indices) > 0:
        qd_arm = qd[arm_joint_indices]

        vlim = np.asarray(ARM_MAX_VELOCITY, dtype=float).reshape(-1)
        if vlim.size == 1:
            vlim = np.full(len(arm_joint_indices), float(vlim[0]))
        elif vlim.size != len(arm_joint_indices):
            vlim = np.full(len(arm_joint_indices), float(np.max(vlim)))
        qd_arm = np.clip(qd_arm, -vlim, vlim)

        alim = np.asarray(ARM_MAX_ACCELERATION, dtype=float).reshape(-1)
        if alim.size == 1:
            alim = np.full(len(arm_joint_indices), float(alim[0]))
        elif alim.size != len(arm_joint_indices):
            alim = np.full(len(arm_joint_indices), float(np.max(alim)))

        if ARM_QD_PREV is None or ARM_QD_PREV.shape[0] != len(arm_joint_indices):
            ARM_QD_PREV = np.zeros(len(arm_joint_indices))

        dq_max = alim * dt
        qd_arm = np.clip(qd_arm, ARM_QD_PREV - dq_max, ARM_QD_PREV + dq_max)

        qd[arm_joint_indices] = qd_arm
        ARM_QD_PREV = qd_arm.copy()

    # Debug: show arm joint velocities
    if len(arm_joint_indices) == 6:
        qd_arm = qd[arm_joint_indices]
        # print(f"  qd_arm: {qd_arm}")

    if et < ARRIVAL_THRESHOLD:
        return True, qd
    else:
        return False, qd


env = swift.Swift()
env.launch(realtime=True)

ax_goal = sg.Axes(0.1)
env.add(ax_goal)

ax_ee = sg.Axes(0.08)
env.add(ax_ee)

if ROBOT_MODEL == "RbKairosPlus":
    robot = rtb.models.RbKairosPlus(str(MODEL_XACRO_PATHS[ROBOT_MODEL]))
elif ROBOT_MODEL == "RbRoboutPlus":
    robot = rtb.models.RbRoboutPlus(str(MODEL_XACRO_PATHS[ROBOT_MODEL]))
else:
    raise ValueError(
        f"Unsupported ROBOT_MODEL '{ROBOT_MODEL}'. "
        "Use 'RbKairosPlus' or 'RbRoboutPlus'."
    )

if WHEELS_USE_REST_CONFIG:
    robot.q = robot.qr
else:
    robot.q = np.zeros(robot.n)

# For the URDF model, start from a safer arm posture to reduce self-intersection
# with the mobile platform during whole-body motion.
arm_joint_indices_init = sorted(
    [
        link.jindex
        for link in robot.links
        if link.isjoint and link.name.startswith("arm_") and link.jindex is not None
    ]
)
if len(arm_joint_indices_init) == 6:
    robot.q[arm_joint_indices_init] = ARM_STARTING_POSTURE

env.add(robot)

has_virtual_base = any(link.name == "base0" for link in robot.links) and any(
    link.name == "base1" for link in robot.links
)

arrived = False
dt = 0.025

# Behind
env.set_camera_pose([-2, 3, 0.7], [-2, 0.0, 0.5])
wTep = robot.fkine(robot.q, end=ARM_EE_LINK) * sm.SE3.Rz(np.pi)
wTep.A[:3, :3] = np.diag([-1, 1, -1]) @ sm.SE3.Ry(-np.pi / 2).A[:3, :3]
wTep.A[0, -1] -= 4.0
wTep.A[2, -1] -= 0.25
ax_goal.T = wTep
ax_ee.T = robot.fkine(robot.q, end=ARM_EE_LINK)
env.step()


while not arrived:

    arrived, robot.qd = step_robot(robot, wTep.A, dt)
    ax_ee.T = robot.fkine(robot.q, end=ARM_EE_LINK)
    env.step(dt)

    # Reset virtual base joints only for the ETS model.
    if has_virtual_base:
        base_new = robot.fkine(robot._q, end=robot.links[3]).A
        robot._T = base_new
        robot.q[:3] = 0

env.hold()
