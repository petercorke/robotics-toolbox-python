#!/usr/bin/env python

import numpy as np

def jacob0(robot, q):
    """
    The manipulator Jacobian matrix maps joint velocity to end-effector 
    spatial velocity, expressed in the world-coordinate frame.
    
    Parameters
    ----------
    robot : An object of SerialLink or subclass type
        The robot to calculate the fkine of
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot

    Returns
    -------
    J : float
        The manipulator Jacobian in 0 frame

    Examples
    --------
    >>> diff = ang_diff(1, 2)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """

    if not isinstance(q, np.ndarray):
        raise TypeError('q array must be a numpy ndarray.')
    if q.shape != (robot.n,):
        raise ValueError('q must be a 1 dim (n,) array')
    
    Je = robot.Je
    T = robot.T
    R = T[:3, :3]

    J0 = np.concatenate(
        (
            np.concatenate((R, np.zeros((3,3))), axis=1), 
            np.concatenate((np.zeros((3,3)), R), axis=1)
        )
    )

    J0 = np.matmul(J0, Je)
            
    return J0
