#!/usr/bin/env python

import numpy as np
from spatialmath import SE3


def p_servo(wTe, wTep, gain=2, threshold=0.1):
    '''
    Position-based servoing.

    Returns the end-effector velocity which will cause the robot to approach
    the desired pose.

    :param wTe: The current pose of the end-effecor in the base frame.
    :type wTe: SE3
    :param wTep: The desired pose of the end-effecor in the base frame.
    :type wTep: SE3
    :param gain: The gain for the controller
    :type gain: float
    :param threshold: The threshold or tolerance of the final error between
        the robot's pose and desired pose
    :type threshold: float

    :returns v: The velocity of the end-effecotr which will casue the robot
        to approach wTep
    :rtype v: ndarray(6)
    :returns arrived: True if the robot is within the threshold of the final
        pose
    :rtype arrived: bool

    '''

    if not isinstance(wTe, SE3):
        wTe = SE3(wTe)

    if not isinstance(wTep, SE3):
        wTep = SE3(wTep)

    # Pose difference
    eTep = wTe.inv() * wTep

    # Translational velocity error
    ev = eTep.t

    # Angular velocity error
    ew = eTep.rpy('rad')

    # Form error vector
    e = np.r_[ev, ew]

    # Desired end-effector velocity
    v = gain * e

    if np.sum(np.abs(e)) < threshold:
        arrived = True
    else:
        arrived = False

    return v, arrived
