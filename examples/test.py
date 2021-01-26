#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp
import time

# Launch the simulator Swift
env = rtb.backends.Swift()
env.launch()

# Create a Panda robot object
r = rtb.models.Frankie()
panda = rtb.models.Panda()

# Set joint angles to ready configuration
r.q = r.qr

b1 = rtb.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(1.95, 0, 0.20))

b2 = rtb.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(1.95, 0, 0.60))

b3 = rtb.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(1.95, 0, 1.00))

b4 = rtb.Box(
    scale=[0.60, 1.1, 0.02],
    base=sm.SE3(1.95, 0, 1.40))

b5 = rtb.Box(
    scale=[0.60, 0.02, 1.40],
    base=sm.SE3(1.95, 0.55, 0.7))

b6 = rtb.Box(
    scale=[0.60, 0.02, 1.40],
    base=sm.SE3(1.95, -0.55, 0.7))

# Make two obstacles with velocities
s0 = rtb.Sphere(
    radius=0.05,
    base=sm.SE3(1.52, 0.4, 0.4)
)
# s0.v = [0, -0.2, 0, 0, 0, 0]

s1 = rtb.Sphere(
    radius=0.05,
    base=sm.SE3(0.5, 0.45, 0.85)
)
s1.v = [0, -0.2, 0, 0, 0, 0]

# s2 = rtb.Box(
#     scale=[0.1, 3.0, 1.0],
#     base=sm.SE3(1.0, 0.0, 1.5) * sm.SE3.Ry(-np.pi/3)
# )

s2 = rtb.Sphere(
    radius=0.2,
    base=sm.SE3(1.0, -0.3, 0.4)
)

# s3 = rtb.Box(
#     scale=[2.0, 0.1, 2.0],
#     base=sm.SE3(0.0, 0.5, 0.0)
# )

# s4 = rtb.Box(
#     scale=[2.0, 0.1, 2.0],
#     base=sm.SE3(0.0, -0.5, 0.0)
# )

collisions = [s0, s1, s2]

# Make a target
target = rtb.Sphere(
    radius=0.02,
    base=sm.SE3(1.3, -0.2, 0.0)
)

# Add the puma to the simulator
env.add(r)
env.add(s0)
env.add(s1)
env.add(s2)
# env.add(s3)
# env.add(s4)
env.add(b1)
env.add(b2)
env.add(b3)
env.add(b4)
env.add(b5)
env.add(b6)
env.add(target)



time.sleep(1)

Tep = r.fkine(r.q) * sm.SE3.Tx(1.3) * sm.SE3.Ty(0.4) * sm.SE3.Tz(-0.2)
target.base = Tep

arrived = False

dt = 0.01

# env.start_recording('frankie_recording', 1 / dt)

while not arrived:

    # The pose of the Panda's end-effector
    Te = r.fkine(r.q)

    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))

    v, arrived = rtb.p_servo(r.fkine(r.q), Tep, gain=0.6, threshold=0.01)

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(r.n + 6)

    # Joint velocity component of Q
    Q[:r.n, :r.n] *= Y

    # Slack component of Q
    Q[r.n:, r.n:] = (1 / e) * np.eye(6)
    # Q[r.n:, r.n:] = 10000 * np.eye(6)

    # The equality contraints
    Aeq = np.c_[r.jacobe(r.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((r.n + 6, r.n + 6))
    bin = np.zeros(r.n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:r.n, :r.n], bin[:r.n] = r.joint_velocity_damper(ps, pi, r.n)

    # For each collision in the scene
    for collision in collisions:

        # Form the velocity damper inequality contraint for each collision
        # object on the robot to the collision in the scene
        c_Ain, c_bin = r.link_collision_damper(
            collision, r.q[:r.n], 0.3, 0.01, xi=1.0,
            startlink=r.link_dict['panda_base0'],
            endlink=r.link_dict['frankie_hand'])

        # If there are any parts of the robot within the influence distance
        # to the collision in the scene
        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

            # Stack the inequality constraints
            Ain = np.r_[Ain, c_Ain]
            bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    Jm = r.jacobm(r.q).reshape((r.n,))
    Jm[1] = 0
    Jm[2:] = panda.jacobm(r.q[2:]).reshape((7,))
    # print(Jm)
    # Jm = np.zeros(9)
    c = np.r_[-Jm, np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[r.qdlim[:r.n], 10 * np.ones(6)]
    ub = np.r_[r.qdlim[:r.n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)

    # Apply the joint velocities to the Panda
    r.qd[:r.n] = qd[:r.n]

    # Step the simulator by 50 ms
    env.step(dt)

# env.stop_recording()