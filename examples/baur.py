#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
import swift
import spatialmath as sm
import numpy as np

"""
This is an implementation of the controller from:
J. Baur, J. Pfaff, H. Ulbrich, and T. Villgrattner, “Design and development
of a redundant modular multipurpose agricultural manipulator,” in 2012
IEEE/ASME International Conference on Advanced Intelligent Mechatronics
(AIM), 2012, pp. 823–830.
"""

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# Create a Panda robot object
panda = rtb.models.Panda()

# Set joint angles to ready configuration
panda.q = panda.qr

# Add the Panda to the simulator
env.add(panda)

# Number of joint in the panda which we are controlling
n = 7

# Set the desired end-effector pose
Tep = panda.fkine(panda.q) * sm.SE3(0.3, 0.2, 0.3)

arrived = False

while not arrived:

    # The pose of the Panda's end-effector
    Te = panda.fkine(panda.q)

    # The manipulator Jacobian in the end-effecotr frame
    Je = panda.jacobe(panda.q)

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v, arrived = rtb.p_servo(Te, Tep, 1.0)

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # The manipulability Jacobian
    Jm = panda.jacobm(panda.q)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Add cost to going in the direction of joint limits, if they are within
    # the influence distance
    b = np.zeros((n, 1))

    for i in range(n):
        if panda.q[i] - panda.qlim[0, i] <= pi:
            b[i, 0] = -1 * \
                np.power(((panda.q[i] - panda.qlim[0, i]) - pi), 2) \
                / np.power((ps - pi), 2)
        if panda.qlim[1, i] - panda.q[i] <= pi:
            b[i, 0] = 1 * \
                np.power(((panda.qlim[1, i] - panda.q[i]) - pi), 2) \
                / np.power((ps - pi), 2)

    # Project the gradient of manipulability into the null-space of the
    # differential kinematics
    null = (
        np.eye(n) - np.linalg.pinv(Je) @ Je
    ) @ (Jm - b)

    # Solve for the joint velocities dq
    qd = np.linalg.pinv(Je) @ v + 1 / Y * null.flatten()

    # Apply the joint velocities to the Panda
    panda.qd[:n] = qd[:n]

    # Step the simulator by 50 ms
    env.step(0.05)
