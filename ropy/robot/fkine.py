#! /usr/bin/env python

import numpy as np
# from robotics.robot.serial_link import SerialLink

def fkine(robot, q):
    '''
    Evaluate fkine for each point on a trajectory of joints q
    
    Attributes:
    --------
        robot : An object of SerialLink or subclass type
            The robot to calculate the fkine of
        q : float np.ndarray(1,n)
            The joint angles/configuration of the robot

    See Also
    --------
    ropy.robot.Link : A link superclass for all link types
    '''

    # if not isinstance(robot, SerialLink):
    #     raise TypeError('The robot must be of type SerialLink.')
    if not isinstance(q, np.ndarray):
        raise TypeError('q array must be a numpy ndarray.')
    if q.shape != (robot.n,):
        raise ValueError('q must be a 1 dim (n,) array')
    
    n = robot.n
    L = robot.links
    t = robot.base

    for i in range(n):
        t = np.matmul(t, L[i].A(q[i]))

    t = np.matmul(t, robot.tool)

    return t
