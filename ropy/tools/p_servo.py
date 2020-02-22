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


def planar_translation(a, b):
    """
    Returns xy distance between trans a and b
    
    Parameters
    ----------
    a : numpy.ndarray((4, 4))
        Homogeneous transform a (SE3)
    b : numpy.ndarray((4, 4))
        Homogeneous transform b (SE3)

    Returns
    -------
    dist : float
        The xy distance between trans a and b

    Examples
    --------
    >>> diff = planar_translation(tf_a, tf_b)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """

    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError('The input a, b must be a numpy ndarray')

    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError('The input a, b dimensions be 4,4')

    return (np.sum((a[0:2, -1] - b[0:2, -1])**2))**0.5


def transform(a, b):
    """
    Returns the trans for going from a to b
    
    Parameters
    ----------
    a : numpy.ndarray((4, 4))
        Homogeneous transform a (SE3)
    b : numpy.ndarray((4, 4))
        Homogeneous transform b (SE3)

    Returns
    -------
    trans : numpy.ndarray((4, 4))
        Homogeneous transform a (SE3)

    Examples
    --------
    >>> trans = transform(tf_a, tf_b)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """

    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError('The input a, b must be a numpy ndarray')

    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError('The input a, b dimensions be 4,4')

    return np.matmul(np.linalg.inv(a), b)


def mean_trans(ts):
    """
    Returns the mean trans from a list of trans (rotation mean is only mean yaw)
    
    Parameters
    ----------
    ts : numpy.ndarray((4, 4, n))
        A series of n Homogeneous transforms (SE3)

    Returns
    -------
    mean_rans : numpy.ndarray((4, 4))
        The mean homogeneous transform (SE3) of the yaw and translation

    Examples
    --------
    >>> mean = mean_trans(tf_s)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """

    if not isinstance(ts, np.ndarray):
        raise TypeError('The input ts must be a numpy ndarray')

    ys = [t3.euler.mat2euler(t[0:3, 0:3])[2] for t in ts]

    y_mean = np.arctan2(
        np.sum([np.sin(y) for y in ys]), 
        np.sum([np.cos(y) for y in ys]))

    mean = np.mean(ts, axis=0)
    mean[0:3, 0:3] = t3.euler.euler2mat(0, 0, y_mean)
    return mean


def relative_yaw_to_trans(a, b):
    """
    Returns the relative yaw to b, from a
    
    Parameters
    ----------
    a : numpy.ndarray((4, 4))
        Homogeneous transform a (SE3)
    b : numpy.ndarray((4, 4))
        Homogeneous transform b (SE3)

    Returns
    -------
    yaw : float
        The relative yaw to b, from a

    Examples
    --------
    >>> yaw = relative_yaw_to_trans(tf_a, tf_b)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """

    if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
        raise TypeError('The input a, b must be a numpy ndarray')

    if a.shape != (4, 4) or b.shape != (4, 4):
        raise ValueError('The input a, b dimensions be 4,4')

    return ang_diff(
        t3.euler.mat2euler(a[0:3, 0:3])[2], #This is the yaw angle of tf a
        np.arctan2(b[1, -1] - a[1, -1], b[0, -1] - a[0, -1]))


def rpy_to_trans(roll, pitch, yaw):
    """
    Returns the input rpy to a homogeneous SE3 transform
    
    Parameters
    ----------
    roll : float
        Roll angle in radians
    pitch : float
        pitch angle in radians
    yaw : float
        yaw angle in radians

    Returns
    -------
    trans : numpy.ndarray((4, 4))
        Homogeneous transform (SE3)

    Examples
    --------
    >>> trans = rpy_to_trans(roll, pitch, yaw)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """

    roll = float(roll)
    pitch = float(pitch)
    yaw = float(yaw)

    rot = t3.euler.euler2mat(roll, pitch, yaw)
    trans = np.zeros((4,4))
    trans[3,3] = 1
    trans[0:3, 0:3] = rot
    return trans


def xyzrpy_to_trans(x, y, z, roll, pitch, yaw):
    """
    Returns the input xyz, rpy to a homogeneous SE3 transform
    
    Parameters
    ----------
    x : float
        x in metres
    y : float
        y in metres
    z : float
        z in metres
    roll : float
        Roll angle in radians
    pitch : float
        pitch angle in radians
    yaw : float
        yaw angle in radians

    Returns
    -------
    trans : numpy.ndarray((4, 4))
        Homogeneous transform (SE3)

    Examples
    --------
    >>> trans = xyzrpy_to_trans(roll, pitch, yaw)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """

    trans = rpy_to_trans(roll, pitch, yaw)
    trans[0, 3] = x
    trans[1, 3] = y
    trans[2, 3] = z

    return trans

def transl(x, y, z):
    """
    Returns the input xyz to a homogeneous SE3 transform
    
    Parameters
    ----------
    x : float
        x in metres
    y : float
        y in metres
    z : float
        z in metres

    Returns
    -------
    trans : numpy.ndarray((4, 4))
        Homogeneous transform (SE3)

    Examples
    --------
    >>> trans = transl(x, y, z)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """

    trans = np.eye(4)
    trans[0, 3] = x
    trans[1, 3] = y
    trans[2, 3] = z

    return trans