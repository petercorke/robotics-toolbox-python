#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

"""
This is an implementation of the controller from:
Park, C. Wangkyun, and Y. Youngil, “Computation of gradientof manipulability
for kinematically redundant manipulators includingdual manipulators system,”
Transactions on Control, Automation and Systems Engineering, vol. 1, no. 1,
pp. 8–15, 1999.
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

    # Project the gradient of manipulability into the null-space of the
    # differential kinematics
    null = (
        np.eye(n) - np.linalg.pinv(Je) @ Je
    ) @ Jm

    # Solve for the joint velocities dq
    qd = np.linalg.pinv(Je) @ v + 1 / Y * null.flatten()

    # Apply the joint velocities to the Panda
    panda.qd[:n] = qd[:n]

    # Step the simulator by 50 ms
    env.step(0.05)
