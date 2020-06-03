#!/usr/bin/env python

import numpy as np
from rtb.robot.Link import Link
# from ropy.robot.fkine import fkine
# from ropy.robot.jocobe import jacobe
# from ropy.robot.jocob0 import jacob0
# from ropy.robot.ets import ets

class SerialLink(object):
    """
    A superclass for arm type robots

    Note: Link subclass elements passed in must be all standard, or all 
          modified, DH parameters.
    
    Attributes:
    --------
        name : string
            Name of the robot
        manufacturer : string
            Manufacturer of the robot
        base : float np.ndarray(4,4)
            Locaation of the base
        tool : float np.ndarray(4,4)
            Location of the tool
        links : List[n]
            Series of links which define the robot
        mdh : int
            0 if standard D&H, else 1
        n : int
            Number of joints in the robot
        T : float np.ndarray(4,4)
            The current pose of the robot
        q : float np.ndarray(1,n)
            The current joint angles of the robot
        Je : float np.ndarray(6,n)
            The manipulator Jacobian matrix maps joint velocity to end-effector
            spatial velocity in the ee frame
        J0 : float np.ndarray(6,n)
            The manipulator Jacobian matrix maps joint velocity to end-effector
            spatial velocity in the 0 frame
        He : float np.ndarray(6,n,n)
            The manipulator Hessian matrix maps joint acceleration to end-effector
            spatial acceleration in the ee frame
        H0 : float np.ndarray(6,n,n)
            The manipulator Hessian matrix maps joint acceleration to end-effector
            spatial acceleration in the 0 frame

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
            base = np.eye(4,4),
            tool = np.eye(4,4)
            ):

        self._name = name
        self._manuf = manufacturer
        self._links = []
        self._base = base
        self._tool = tool
        self._T = np.eye(4)

        super(SerialLink, self).__init__()        

        if not isinstance(L, list):
            raise TypeError('The links L must be stored in a list.')
        else:
            if not isinstance(L[0], Link):
                raise TypeError('The links in L must be of Link type.')
            else:
                self._links = L

        self._n = len(self._links)
        self._q = np.zeros((self._n,))

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


