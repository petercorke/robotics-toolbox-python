#!/usr/bin/env python

import numpy as np
import transforms3d as t3


def p_servo(wTe, wTep, gain = 3, threshold = 0.1):
    """
    Position-based servoing. Returns the end-effector velocity which will cause
    the robot to approach the desired pose.
    
    Parameters
    ----------
    wTe : numpy.ndarray((4, 4))
        The current pose of the end-effecor in the base frame. Homogeneous 
        transform (SE3)
    wTep : numpy.ndarray((4, 4))
        The desired pose of the end-effecor in the base frame. Homogeneous 
        transform (SE3)
    gain : float
        The gain for the controller
    threshold : float
        The threshold or tolerance of the final error between the robot's pose
        and desired pose

    Returns
    -------
    v : numpy.ndarray((7, 1))
        The velocity of the end-effecotr which will casue the robot to approach 
        wTep
    arrived : bool
        True if the robot is within the threshold of the final pose

    Examples
    --------
    >>> v, arrived = p_servo(wTe, wTep, 3)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """

    # Pose difference
    eTep = np.linalg.inv(wTe) @ wTep

    # Translational velocity error
    ev = eTep[0:3,-1]

    # Angular velocity error
    ew = t3.euler.mat2euler(eTep[0:3, 0:3])

    # Form error vector
    e = np.expand_dims(np.concatenate((ev, ew)), axis=1)

    # Desired end-effector velocity
    v = gain * e

    if np.sum(np.abs(e)) < threshold:
        arrived = True 
    else:
        arrived = False

    return v, arrived
