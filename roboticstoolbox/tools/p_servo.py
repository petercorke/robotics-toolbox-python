#!/usr/bin/env python

import numpy as np
from spatialmath import SE3, base
import math


def p_servo(wTe, wTep, gain=2, threshold=0.1):
    """
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

    """

    if not isinstance(wTe, SE3):
        wTe = SE3(wTe)

    if not isinstance(wTep, SE3):
        wTep = SE3(wTep)

    # Pose difference
    eTep = wTe.inv() * wTep

    # Translational error
    ev = eTep.t

    # Angular error
    ew = eTep.rpy("rad")

    # Form error vector
    e = np.r_[ev, ew]

    # Desired end-effector velocity
    v = gain * e

    if np.sum(np.abs(e)) < threshold:
        arrived = True
    else:
        arrived = False

    return v, arrived


# def _angle_axis(T, Td):
#     d = base.transl(Td) - base.transl(T)
#     R = base.t2r(Td) @ base.t2r(T).T
#     li = np.r_[R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]

#     if base.iszerovec(li):
#         # diagonal matrix case
#         if np.trace(R) > 0:
#             # (1,1,1) case
#             a = np.zeros((3,))
#         else:
#             a = np.pi / 2 * (np.diag(R) + 1)
#     else:
#         # non-diagonal matrix case
#         ln = base.norm(li)
#         a = math.atan2(ln, np.trace(R) - 1) * li / ln

#     return np.r_[d, a]
