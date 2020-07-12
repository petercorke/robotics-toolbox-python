#!/usr/bin/env python

import numpy as np
from ropy.robot.Link import Link
from ropy.robot.fkine import fkine
from ropy.robot.jocobe import jacobe
from ropy.robot.jocob0 import jacob0
from spatialmath.base.argcheck import getvector, ismatrix
import spatialmath.base as sp


class SerialLink(object):
    """
    A superclass for arm type robots

    Note: Link subclass elements passed in must be all standard, or all 
          modified, DH parameters.
    
    :param name: Name of the robot
    :type name: string
    :param manufacturer: Manufacturer of the robot
    :type manufacturer: string
    :param base: Locaation of the base 
    :type base: float np.ndarray(4,4)
    :param tool: Location of the tool 
    :type tool: float np.ndarray(4,4)
    :param links: Series of links which define the robot 
    :type links: List[n]
    :param mdh: 0 if standard D&H, else 1 
    :type mdh: int
    :param n: Number of joints in the robot
    :type n: int
    :param T: The current pose of the robot 
    :type T: float np.ndarray(4,4)
    :param q: The current joint angles of the robot
    :type q: float np.ndarray(1,n)

    Examples
    --------
    >>> L[0] = Revolute('d', 0, 'a', a1, 'alpha', np.pi/2)

    >>> L[1] = Revolute('d', 0, 'a', a2, 'alpha', 0)

    >>> twolink = SerialLink(L, 'name', 'two link');

    See Also
    --------
    ropy.robot.ets : A superclass which represents the kinematics of a 
                     serial-link manipulator
    ropy.robot.Link : A link superclass for all link types
    ropy.robot.Revolute : A revolute link class
    """

    def __init__(self,
            L, 
            name = 'noname', 
            manufacturer = '', 
            base = np.eye(4, 4),
            tool = np.eye(4, 4)
            ):

        self._name = name
        self._manuf = manufacturer
        self._links = []
        self._base = base
        self._tool = tool
        self._T = np.eye(4)

        super(SerialLink, self).__init__()        

        # Verify link length
        if not isinstance(L, list):
            raise TypeError('The links L must be stored in a list.')
        else:
            if not isinstance(L[0], Link):
                raise TypeError('The links in L must be of Link type.')
            else:
                self._links = L
        
        # Number of joints in the robot
        self._n = len(self._links)

        # Current joint angles of the robot
        self._q = np.zeros((self._n,))

        # Check the DH convention
        self._mdh = self.links[0].mdh
        for i in range(self._n):
            if not self._links[i].mdh == self._mdh:
                raise ValueError('Robot has mixed D&H links conventions.')


    # Property methods

    @property
    def name(self):
        return self._name

    @property
    def manuf(self):
        return self._manuf

    @property
    def links(self):
        return self._links

    @property
    def base(self):
        return self._base

    @property
    def tool(self):
        return self._tool

    @property
    def n(self):
        return self._n

    @property
    def mdh(self):
        return self._mdh

    @property
    def q(self):
        return self._q  


    # Setter methods
    @base.setter
    def f(self, T):
        if not isinstance(T, np.ndarray):
            raise TypeError('Transformation matrix must be a numpy ndarray')
        elif T.shape != (4,4):
            raise ValueError('Transformation matrix must be a 4x4')
        self._base = T

    @q.setter
    def q(self, q_new):
        if not isinstance(q_new, np.ndarray):
            raise TypeError('q array must be a numpy ndarray')
        elif q_new.shape != (self._n,):
            raise ValueError('q must be a 1 dim (n,) array')
        
        self._q = q_new



    """
    The spatial velocity Jacobian which relates the velocity in end-effector 
    frame to velocity in the base frame.
    
    Parameters
    ----------
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot

    Returns
    -------
    J : float np.ndarray(6,n)
        The velocity Jacobian in 0 frame

    Examples
    --------
    >>> J = panda.jacob0v(np.array([1,1,1,1,1,1,1]))
    >>> J = panda.J0v
    
    See Also
    --------
    ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame 
    ropy.robot.m : Calculates the manipulability index of the robot
    ropy.robot.Jm : Calculates the manipiulability Jacobian
    ropy.robot.fkine : Calculates the forward kinematics of a robot
    """
    def jacob0v(self, q):
        r = self.fkine(q)[0:3,0:3]

        Jv = np.zeros((6,6))
        Jv[:3,:3] = r
        Jv[3:,3:] = r

        return Jv



    """
    The spatial velocity Jacobian which relates the velocity in base 
    frame to velocity in the end-effector frame.
    
    Parameters
    ----------
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot

    Returns
    -------
    J : float np.ndarray(6,n)
        The velocity Jacobian in ee frame

    Examples
    --------
    >>> J = panda.jacobev(np.array([1,1,1,1,1,1,1]))
    >>> J = panda.Jev
    
    See Also
    --------
    ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame 
    ropy.robot.m : Calculates the manipulability index of the robot
    ropy.robot.Jm : Calculates the manipiulability Jacobian
    ropy.robot.fkine : Calculates the forward kinematics of a robot
    """
    def jacobev(self, q):
        r = self.fkine(q)[0:3,0:3]
        r = np.linalg.inv(r)

        Jv = np.zeros((6,6))
        Jv[:3,:3] = r
        Jv[3:,3:] = r

        return Jv
