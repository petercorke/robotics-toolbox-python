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
env = rtb.backend.Swift()
env.launch()

# Create a Panda robot object
panda = rtb.models.Panda()

# Set joint angles to ready configuration
# panda.q = [
    # -0.5653, -0.1941, -1.2602, -0.7896, -2.3227, -0.3919, -2.5173, 0.0, 0.0]
panda.q = panda.qr

# Number of joint in the panda which we are controlling
n = 7

# Make two obstacles with velocities
s0 = rtb.Sphere(
    radius=0.05,
    base=sm.SE3(0.45, 0.4, 0.3)
)
s0.v = [0.01, -0.2, 0, 0, 0, 0]

s1 = rtb.Sphere(
    radius=0.05,
    base=sm.SE3(0.1, 0.35, 0.65)
)
s1.v = [0, -0.2, 0, 0, 0, 0]

collisions = [s0, s1]

# Make a target
target = rtb.Sphere(
    radius=0.02,
    base=sm.SE3(0.6, -0.2, 0.0)
)

# Add the Panda and shapes to the simulator
env.add(panda)
env.add(s0)
env.add(s1)
env.add(target)

# Set the desired end-effector pose to the location of target
Tep = panda.fkine()
Tep.A[:3, 3] = target.base.t
Tep.A[2, 3] += 0.1

arrived = False

while not arrived:

    # The pose of the Panda's end-effector
    Te = panda.fkine()

    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi/180]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v, arrived = rtb.p_servo(Te, Tep, 0.5, 0.05)

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)

    # The equality contraints
    Aeq = np.c_[panda.jacobe(), np.eye(6)]
    beq = v.reshape((6,))

    # # The inequality constraints for joint limit avoidance
    # Ain = np.zeros((n + 6, n + 6))
    # bin = np.zeros(n + 6)

#     # The minimum angle (in radians) in which the joint is allowed to approach
#     # to its limit
#     ps = 0.05

#     # The influence angle (in radians) in which the velocity damper
#     # becomes active
#     pi = 0.9

#     # Form the joint limit velocity damper
#     Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

    Ain = None
    bin = None

    for collision in collisions:

        c_Ain, c_bin = panda.link_collision_damper(
            collision, panda.q[:n], 0.3, 0.05,
            panda.elinks['panda_joint1'], panda.elinks['panda_hand_joint'])

        if c_Ain is not None and c_bin is not None:
            c_Ain = np.c_[c_Ain, np.zeros((c_Ain.shape[0], 6))]

            if Ain is None:
                Ain = c_Ain
            else:
                Ain = np.r_[Ain, c_Ain]

            if bin is None:
                bin = np.array(c_bin)
            else:
                bin = np.r_[bin, c_bin]

    # Linear component of objective function: the manipulability Jacobian
    c = np.r_[-panda.jacobm().reshape((n,)), np.zeros(6)]

    # The lower and upper bounds on the joint velocity and slack variable
    lb = -np.r_[panda.qdlim[:n], 10 * np.ones(6)]
    ub = np.r_[panda.qdlim[:n], 10 * np.ones(6)]

    # Solve for the joint velocities dq
    qd = qp.solve_qp(Q, c, Ain, bin, Aeq, beq, lb=lb, ub=ub)

    # Apply the joint velocities to the Panda
    panda.qd[:n] = qd[:n]

    # Step the simulator by 50 ms
    env.step(50)
