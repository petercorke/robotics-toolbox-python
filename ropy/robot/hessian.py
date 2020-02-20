#!/usr/bin/env python

import numpy as np

def _init_hessian(robot, q):
    

def hessian(robot, q):
    """
    The manipulator Hessian tensor is the second partial derivative of the
    robot's pose. This function represents the Hessian in the robot's base
    frame.
    
    Parameters
    ----------
    robot : An object of SerialLink or subclass type
        The robot to calculate the fkine of
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot

    Returns
    -------
    H : float
        The manipulator Hessian in base frame

    Examples
    --------
    >>> H = hessian(panda, qr)
    
    See Also
    --------
    ropy.tools.relative_yaw_to_trans : Returns the relative yaw to b, 
        from a
    """
    
    if not isinstance(q, np.ndarray):
        raise TypeError('q array must be a numpy ndarray.')
    if q.shape != (robot.n,):
        raise ValueError('q must be a 1 dim (n,) array')






