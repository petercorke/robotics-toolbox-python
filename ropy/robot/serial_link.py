#!/usr/bin/env python

import numpy as np
from ropy.robot.link import Link
from ropy.robot.fkine import fkine
from ropy.robot.jocobe import jacobe
from ropy.robot.jocob0 import jacob0
from ropy.robot.ets import ets

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

        # Initialise Properties
        self._ets = ets(self)


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

    @property
    def T(self):
        return self.fkine(self.q)

    @property
    def Je(self):
        return self.jacobe(self.q)

    @property
    def J0(self):
        return self._ets.jacob0(self.q)

    # @property
    # def He(self):
    #     self._He = self._ets.hessiane(self.q)
    #     return self._He

    @property
    def H0(self):
        return self.hessian0(self.q)

    @property
    def Jev(self):
        return self.jacobev(self.q)

    @property
    def J0v(self):
        return self.jacob0v(self.q)

    @property
    def ets(self):
        return self._ets.to_string()

    @property
    def Jm(self):
        return self.jacobm(self.q)

    @property
    def m(self):
        return self.manip(self.q)

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



    """
    The manipulator Jacobian matrix maps joint velocity to end-effector 
    spatial velocity, expressed in the world-coordinate frame. This 
    function calulcates this based on the ETS of the robot. This Jacobian
    is in the base frame.
    
    Parameters
    ----------
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot

    Returns
    -------
    J : float np.ndarray(6,n)
        The manipulator Jacobian in 0 frame

    Examples
    --------
    >>> J = panda.jacob0(np.array([1,1,1,1,1,1,1]))
    >>> J = panda.J0
    
    See Also
    --------
    ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame 
    ropy.robot.m : Calculates the manipulability index of the robot
    ropy.robot.Jm : Calculates the manipiulability Jacobian
    ropy.robot.fkine : Calculates the forward kinematics of a robot

    References
    --------
    - Kinematic Derivatives using the Elementary Transform Sequence,
      J. Haviland and P. Corke
    """
    def jacob0(self, q):
        return self._ets.jacob0(q)



    """
    The manipulator Jacobian matrix maps joint velocity to end-effector 
    spatial velocity, expressed in the world-coordinate frame. This 
    function calulcates this based on the ETS of the robot. This Jacobian
    is in the end-effector frame.
    
    Parameters
    ----------
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot

    Returns
    -------
    J : float np.ndarray(6,n)
        The manipulator Jacobian in ee frame

    Examples
    --------
    >>> J = panda.jacobe(np.array([1,1,1,1,1,1,1]))
    >>> J = panda.Je
    
    See Also
    --------
    ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame 
    ropy.robot.m : Calculates the manipulability index of the robot
    ropy.robot.Jm : Calculates the manipiulability Jacobian
    ropy.robot.fkine : Calculates the forward kinematics of a robot

    References
    --------
    - Kinematic Derivatives using the Elementary Transform Sequence,
      J. Haviland and P. Corke
    """
    def jacobe(self, q):
        J0 = self._ets.jacob0(q)
        Je = self.jacobev(q) @ J0
        return Je



    """
    The manipulator Hessian tensor maps joint acceleration to end-effector 
    spatial acceleration, expressed in the world-coordinate frame. This 
    function calulcates this based on the ETS of the robot.
    
    Parameters
    ----------
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot

    Returns
    -------
    H : float np.ndarray(1,n,n)
        The manipulator Hessian in 0 frame

    Examples
    --------
    >>> H = panda.hessian0(np.array([1,1,1,1,1,1,1]))
    >>> H = panda.H0
    
    See Also
    --------
    ropy.robot.jacob0 : Calculates the kinematic Jacobian in the world frame 
    ropy.robot.m : Calculates the manipulability index of the robot
    ropy.robot.Jm : Calculates the manipiulability Jacobian
    ropy.robot.fkine : Calculates the forward kinematics of a robot

    References
    --------
    - Kinematic Derivatives using the Elementary Transform Sequence,
      J. Haviland and P. Corke
    """
    def hessian0(self, q):
        return self._ets.hessian0(q)



    '''
    Evaluates the forward kinematics of a robot based on its ETS and 
    joint angles q.
    
    Attributes:
    --------
        q : float np.ndarray(1,n)
            The joint angles/configuration of the robot

    Returns
    -------
    T : float np.ndarray(4,4)
        The pose of the end-effector

    Examples
    --------
    >>> T = panda.fkine(np.array([1,1,1,1,1,1,1]))
    >>> T = panda.T

    See Also
    --------
    ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame 
    ropy.robot.jacob0 : Calculates the kinematic Jacobian in the world frame 
    ropy.robot.m : Calculates the manipulability index of the robot
    ropy.robot.Jm : Calculates the manipiulability Jacobian

    References
    --------
    - Kinematic Derivatives using the Elementary Transform Sequence,
      J. Haviland and P. Corke
    '''
    def fkine(self, q):
        return self._ets.fkine(q)



    """
    Calculates the manipulability index (scalar) robot at the joint 
    configuration q. It indicates dexterity, that is, how isotropic the robot's
    % motion is with respect to the 6 degrees of Cartesian motion. The measure
    is high when the manipulator is capable of equal motion in all directions
    and low when the manipulator is close to a singularity.
    
    Parameters
    ----------
    q : float np.ndarray(1,n)
        The joint angles/configuration of the robot

    Returns
    -------
    m : float
        The manipulability index

    Examples
    --------
    >>> m = panda.manip(np.array([1,1,1,1,1,1,1]))
    >>> m = panda.m
    
    See Also
    --------
    ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame 
    ropy.robot.jacob0 : Calculates the kinematic Jacobian in the world frame 
    ropy.robot.Jm : Calculates the manipiulability Jacobian
    ropy.robot.fkine : Calculates the forward kinematics of a robot

    References
    --------
    - Analysis and control of robot manipulators with redundancy,
      T. Yoshikawa,
      Robotics Research: The First International Symposium (M. Brady and R. Paul, eds.),
      pp. 735-747, The MIT press, 1984.
    """
    def manip(self, q):
        return self._ets.m(q)



    """
    Calculates the manipulability Jacobian. This measure relates the rate of 
    change of the manipulability to the joint velocities of the robot.
    
    Parameters
    ----------
    dq : float np.ndarray(1,n)
        The joint velocities of the robot

    Returns
    -------
    m : float
        The manipulability index

    Examples
    --------
    >>> Jm = panda.jacobm(np.array([1,1,1,1,1,1,1]))
    >>> Jm = panda.Jm
    
    See Also
    --------
    ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame 
    ropy.robot.jacob0 : Calculates the kinematic Jacobian in the world frame 
    ropy.robot.m : Calculates the manipulability index of the robot
    ropy.robot.fkine : Calculates the forward kinematics of a robot

    References
    --------
    - Maximising Manipulability in Resolved-Rate Motion Control,
      J. Haviland and P. Corke
    """
    def jacobm(self, q):
        return self._ets.Jm(q)