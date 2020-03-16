#!/usr/bin/env python

import numpy as np

def jacobe(robot, q):
    """
    The manipulator Jacobian matrix maps joint velocity to end-effector 
    spatial velocity V = JE*QD in the end-effector frame.
    
    Parameters
    ----------
    robot : An object of SerialLink or subclass type
        The robot to calculate the fkine of
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot

    Returns
    -------
    J : float
        The manipulator Jacobian in ee frame

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
    
    n = robot.n
    L = robot.links
    J = np.zeros((6, robot.n))

    U = robot.tool

    for j in range(n-1, -1, -1):
        if robot.mdh == 0:
            # standard DH convention
            U = np.matmul(L[j].A(q[j]), U)
        
        if L[j].is_revolute:
            # revolute axis
            d = np.array([[ -U[0,0] * U[1,3] + U[1,0] * U[0,3] ],
                          [ -U[0,1] * U[1,3] + U[1,1] * U[0,3] ],
                          [ -U[0,2] * U[1,3] + U[1,2] * U[0,3] ]])
            delta = np.expand_dims(U[2,:3], axis=1)  # nz oz az
        else:
            # prismatic axis
            d = np.expand_dims(U[2,:3], axis=1)      # nz oz az
            delta = np.zeros((3,))

        J[:,j] = np.squeeze(np.concatenate((d, delta)))

        if robot.mdh != 0:
            # modified DH convention
            U = np.matmul(L[j].A(q[j]), U)
        
    return J
