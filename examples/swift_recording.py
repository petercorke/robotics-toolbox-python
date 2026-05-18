#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import swift
import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np
import qpsolvers as qp

"""
This is an implementation of the controller from:
J. Haviland and P. Corke, “A purely-reactive manipulability-maximising
motion controller,” arXiv preprint arXiv:2002.11901,2020.
"""

# Launch the simulator Swift
env = swift.Swift()

# Launch the sim in chrome as only chrome supports webm videos
env.launch('google-chrome')

# Create a Panda robot object
panda = rtb.models.Panda()

# Set joint angles to ready configuration
panda.q = panda.qr

# Add the Panda to the simulator
env.add(panda)

# The timestep for the simulation, 25ms
dt = 0.025

# Start recording with a framerate of 1/dt and
# call the video panda_swift_recording
# Export video as a webm file (this only works in Chrome)
env.start_recording('panda_swift_recording', 1 / dt)

# To export as a gif replace the above line with
# env.start_recording('panda_swift_recording', 1 / dt, format='gif')

# To export as a tar folder of jpg captures replace with
# env.start_recording('panda_swift_recording', 1 / dt, format='jpg')

# To export as a tar folder of png captures replace with
# env.start_recording('panda_swift_recording', 1 / dt, format='png')

# Number of joint in the panda which we are controlling
n = 7

# Set the desired end-effector pose
Tep = panda.fkine(panda.q) * sm.SE3(0.3, 0.2, 0.3)

arrived = False

while not arrived:

    # The pose of the Panda's end-effector
    Te = panda.fkine(panda.q)

    # Transform from the end-effector to desired pose
    eTep = Te.inv() * Tep

    # Spatial error
    e = np.sum(np.abs(np.r_[eTep.t, eTep.rpy() * np.pi / 180]))

    # Calulate the required end-effector spatial velocity for the robot
    # to approach the goal. Gain is set to 1.0
    v, arrived = rtb.p_servo(Te, Tep, 1.0)

    # Gain term (lambda) for control minimisation
    Y = 0.01

    # Quadratic component of objective function
    Q = np.eye(n + 6)

    # Joint velocity component of Q
    Q[:n, :n] *= Y

    # Slack component of Q
    Q[n:, n:] = (1 / e) * np.eye(6)

    # The equality contraints
    Aeq = np.c_[panda.jacobe(panda.q), np.eye(6)]
    beq = v.reshape((6,))

    # The inequality constraints for joint limit avoidance
    Ain = np.zeros((n + 6, n + 6))
    bin = np.zeros(n + 6)

    # The minimum angle (in radians) in which the joint is allowed to approach
    # to its limit
    ps = 0.05

    # The influence angle (in radians) in which the velocity damper
    # becomes active
    pi = 0.9

    # Form the joint limit velocity damper
    Ain[:n, :n], bin[:n] = panda.joint_velocity_damper(ps, pi, n)

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
    env.step(dt)

# Stop recording and save the video (to the downloads folder)
env.stop_recording()
