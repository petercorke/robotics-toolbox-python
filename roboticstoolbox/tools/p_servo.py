#!/usr/bin/env python

import numpy as np
from spatialmath import SE3, base
import math


def _angle_axis(T, Td):
    e = np.empty(6)
    e[:3] = Td[:3, -1] - T[:3, -1]
    R = Td[:3, :3] @ T[:3, :3].T
    li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    if base.iszerovec(li):
        # diagonal matrix case
        if np.trace(R) > 0:
            # (1,1,1) case
            a = np.zeros((3,))
        else:
            a = np.pi / 2 * (np.diag(R) + 1)
    else:
        # non-diagonal matrix case
        ln = base.norm(li)
        a = math.atan2(ln, np.trace(R) - 1) * li / ln

    e[3:] = a

    return e


def p_servo(wTe, wTep, gain=1.0, threshold=0.1, method="rpy"):
    """
    Position-based servoing.

    Returns the end-effector velocity which will cause the robot to approach
    the desired pose.

    :param wTe: The current pose of the end-effecor in the base frame.
    :type wTe: SE3 or ndarray
    :param wTep: The desired pose of the end-effecor in the base frame.
    :type wTep: SE3 or ndarray
    :param gain: The gain for the controller. Can be vector corresponding to each
        axis, or scalar corresponding to all axes.
    :type gain: float, or array-like
    :param threshold: The threshold or tolerance of the final error between
        the robot's pose and desired pose
    :type threshold: float
    :param method: The method used to calculate the error. Default is 'rpy' -
        error in the end-effector frame. 'angle-axis' - error in the base frame
        using angle-axis method.
    :type method: string: 'rpy' or 'angle-axis'

    :returns v: The velocity of the end-effecotr which will casue the robot
        to approach wTep
    :rtype v: ndarray(6)
    :returns arrived: True if the robot is within the threshold of the final
        pose
    :rtype arrived: bool

    """

    if isinstance(wTe, SE3):
        wTe = wTe.A

    if isinstance(wTep, SE3):
        wTep = wTep.A

    if method == "rpy":
        # Pose difference
        eTep = np.linalg.inv(wTe) @ wTep
        e = np.empty(6)

        # Translational error
        e[:3] = eTep[:3, -1]

        # Angular error
        e[3:] = base.tr2rpy(eTep, unit="rad", order="zyx", check=False)
    else:
        e = _angle_axis(wTe, wTep)

    if base.isscalar(gain):
        k = gain * np.eye(6)
    else:
        k = np.diag(gain)

    v = k @ e
    arrived = True if np.sum(np.abs(e)) < threshold else False

    return v, arrived
