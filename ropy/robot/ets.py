#!/usr/bin/env python

import numpy as np
from multiprocessing import Pool, cpu_count
import ctypes
import time


class ets(object):
    """
    The Elementary Transform Sequence
    
    A superclass which represents the kinematics of a serial-link manipulator

    Attributes:
    --------
    n : int
        The number of transforms in the ETS

    See Also
    --------
    ropy.robot.SerialLink : A superclass for arm type robots
    ropy.robot.hessian0 : Calculates the kinematic Hessian in the world frame 
    ropy.robot.jacob0 : Calculates the kinematic Jacobian in the world frame 
    ropy.robot.m : Calculates the manipulability index of the robot
    ropy.robot.Jm : Calculates the manipiulability Jacobian
    ropy.robot.fkine : Calculates the forward kinematics of a robot
    ropy.robot.to_string : Creates a string representation of the ETS of a robot

    References
    --------
    - Kinematic Derivatives using the Elementary Transform Sequence,
      J. Haviland and P. Corke
    """
    
    def __init__(self, robot):

        super(ets, self).__init__()  

        self.pool = Pool(cpu_count())

        _ets = {}
        _ets_compact = {}
        _ets_string = ''
        self._q_idx = []

        conv = 180.0/np.pi
        
        i = 0
        self._joints = robot.n

        for j in range(self._joints):

            _ets[i] = {}
            L = robot.links[j]

            if robot.mdh:
                # Method for modified DH parameters
                
                # Append Tx(a)
                if L.a != 0:
                    _ets[i]['T'] = self._Tx(L.a)
                    _ets[i]['type'] = 'real'
                    _ets[i]['s'] = 'Tx'
                    _ets[i]['val'] = L.a
                    _ets[i]['val_deg'] = L.a
                    i += 1
                    _ets[i] = {}
                
                # Append Rx(alpha)
                if L.alpha != 0:
                    _ets[i]['T'] = self._Rx(L.alpha)
                    _ets[i]['type'] = 'real'
                    _ets[i]['s'] = 'Rx'
                    _ets[i]['val'] = L.alpha
                    _ets[i]['val_deg'] = L.alpha * conv
                    i += 1
                    _ets[i] = {}
                
                if L.is_revolute:

                    # Append Tz(d)
                    if L.d != 0:
                        _ets[i]['T'] = self._Tz(L.d)
                        _ets[i]['type'] = 'real'
                        _ets[i]['s'] = 'Tz'
                        _ets[i]['val'] = L.d
                        _ets[i]['val_deg'] = L.d
                        i += 1
                        _ets[i] = {}
                    
                    # Append Rz(q)
                    _ets[i]['T']   = self._R
                    _ets[i]['dT']  = self._dR
                    _ets[i]['ddT'] = self._ddR
                    _ets[i]['type'] = 'sym'
                    _ets[i]['s'] = 'Rz'
                    self._q_idx.append(i)
                    i += 1
                    _ets[i] = {}
                else:
                                       
                    # Append Tz(q)
                    _ets[i]['T']   = self._T
                    _ets[i]['dT']  = self._dT
                    _ets[i]['ddT'] = self._ddT
                    _ets[i]['type'] = 'sym'
                    _ets[i]['s'] = 'Tz'
                    self._q_idx.append(i)
                    i += 1
                    _ets[i] = {}

                    # Append Rz(theta)
                    if L.theta != 0:
                        _ets[i]['T'] = self._Rz(L.alpha)
                        _ets[i]['type'] = 'real'
                        _ets[i]['s'] = 'Rz'
                        _ets[i]['val'] = L.alpha
                        _ets[i]['val_deg'] = L.alpha * conv
                        i += 1
                        _ets[i] = {}
                        
            else:
                # Method for modified DH parameters
                
                if L.is_revolute:

                    # Append Rz(q)
                    _ets[i]['T']   = self._R
                    _ets[i]['dT']  = self._dR
                    _ets[i]['ddT'] = self._ddR
                    _ets[i]['type'] = 'sym'
                    _ets[i]['s'] = 'Rz'
                    self._q_idx.append(i)
                    i += 1
                    _ets[i] = {}

                    # Append Tz(d)
                    if L.d != 0:
                        _ets[i]['T'] = self._Tz(L.d)
                        _ets[i]['type'] = 'real'
                        _ets[i]['s'] = 'Tz'
                        _ets[i]['val'] = L.d
                        _ets[i]['val_deg'] = L.d
                        i += 1
                        _ets[i] = {}
                    
                else:
                    # Append Rz(theta)
                    if L.theta != 0:
                        _ets[i]['T'] = self._Rz(L.alpha)
                        _ets[i]['type'] = 'real'
                        _ets[i]['s'] = 'Rz'
                        _ets[i]['val'] = L.alpha
                        _ets[i]['val_deg'] = L.alpha * conv
                        i += 1
                        _ets[i] = {}
                    
                    # Append Tz(q)
                    _ets[i]['T']   = self._T
                    _ets[i]['dT']  = self._dT
                    _ets[i]['ddT'] = self._ddT
                    _ets[i]['type'] = 'sym'
                    _ets[i]['s'] = 'Tz'
                    self._q_idx.append(i)
                    i += 1
                    _ets[i] = {}

                # Append Tx(a)
                if L.a != 0:
                    _ets[i]['T'] = self._Tx(L.a)
                    _ets[i]['type'] = 'real'
                    _ets[i]['s'] = 'Tx'
                    _ets[i]['val'] = L.a
                    _ets[i]['val_deg'] = L.a
                    i += 1
                    _ets[i] = {}

                # Append Rx(alpha)
                if L.alpha != 0:
                    _ets[i]['T'] = self._Rx(L.alpha)
                    _ets[i]['type'] = 'real'
                    _ets[i]['s'] = 'Rx'
                    _ets[i]['val'] = L.alpha
                    _ets[i]['val_deg'] = L.alpha * conv
                    i += 1
                    _ets[i] = {}


        # Add the tool transform
        _ets[i]['T'] = robot.tool
        _ets[i]['type'] = 'real'
        _ets[i]['s'] = 'tool'
        _ets[i]['val'] = 1
        _ets[i]['val_deg'] = 1
        i += 1
        _ets[i] = {}

        # Number of transforms in the ETS
        self.n = i  

        # The ETS of the robot
        self._ets = _ets


    
    """
    The elementary transform sequence (ETS) represents a robot's forward 
    kinematics in terms of 6 elementary transfroms. This prints the ETS
    of a robot.
    
    Parameters
    ----------
    deg : Return the ETS using degrees instead of radians

    Returns
    -------
    s : String
        The ETS of the robot

    Examples
    --------
    >>> s = panda.ets
    
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
    """
    def to_string(self, deg = False):

        s = ''
        j = 1

        for i in range(self.n):

            tr = self._ets[i]['s']

            if self._ets[i]['type'] == 'sym':
                val = 'q{}'.format(j)
                j += 1

            else:
                if deg:
                    val = self._ets[i]['val_deg']
                else:
                    val = self._ets[i]['val']

            s = '{}{}({})'.format(s, tr, val)

        return s

    

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

        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        j = 0

        trans = np.eye(4)

        for i in range(self.n):

            if self._ets[i]['type'] == 'sym':
                s = self._ets[i]['s']
                T = self._ets[i]['T'](s, q[j])
                j += 1

            else:
                T = self._ets[i]['T']

            trans = trans @ T

        return trans

    

    """
    The manipulator Jacobian matrix maps joint velocity to end-effector 
    spatial velocity, expressed in the world-coordinate frame. This 
    function calulcates this based on the ETS of the robot.
    
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

        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        J = np.zeros((6,self._joints))

        T = self.fkine(q)
        R = T[0:3,0:3]

        dT = self._d_ets(q)

        for i in range(self._joints):

            # Linear velocity component of the Jacobian
            J[0:3,i] = dT[i][0:3,3]

            # Angular velocity component of the Jacobian
            J[3:7,i] = np.squeeze(self._vex( dT[i][0:3,0:3] @ np.transpose(R) ))

        return J



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

        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        H = np.zeros((6,self._joints,self._joints))

        t1 = time.time()

        T = self.fkine(q)
        R = T[0:3,0:3]

        dT = self._d_ets(q)
        ddT = self._dd_ets(q)



        for j in range(self._joints):
            for i in range(self._joints):

                

                # Linear velocity component of the Hessian
                H[0:3,i,j] = ddT[i][j][0:3,3]

                sw = ddT[j][i][0:3,0:3] @ np.transpose(R) + \
                    dT[i][0:3,0:3] @ np.transpose(dT[j][0:3,0:3])

                H[3:7,i,j] = np.squeeze(self._vex(sw))

        t2 = time.time()

        # print(t2-t1)

        # with Pool(self._joints) as p:
        #     p.map(self._hessi, [dT, ddT, R, H, range(self._joints)])

        return H



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
    def m(self, q):
        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        J = self.jacob0(q)

        return np.sqrt(np.linalg.det(J @ np.transpose(J)))



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
    def Jm(self, q):
        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        t0 = time.time()

        m = self.m(q)
        t00 = time.time()
        J = self.jacob0(q)
        # H = self.hessian0(q)
        t1 = time.time()

        H = self.hess(q)

        t2 = time.time()
        b = np.linalg.inv(J @ np.transpose(J))
        t3 = time.time()

        a = np.zeros((self._joints,1))

        for i in range(self._joints):
            c = J @ np.transpose(H[:,:,i])
            a[i,0] = m * np.transpose(c.flatten('F')) @ b.flatten('F')

        t4 = time.time()

        print(t00 - t0)
        print(t1 - t00)
        print(t2 - t1)
        print(t3 - t2)
        print(t4 - t3)

        print('\n')

        return a





    '''
    Private functions
    '''

    def _vex(self, T):
        if not isinstance(T, np.ndarray):
            raise TypeError('T matrix must be a numpy ndarray.')
        if T.shape != (3,3) and T.shape != (4,4):
            raise ValueError('T must be a 3x3 or 4x4 matrix')

        w = np.zeros((3,1))

        w[0,0] = 0.5 * (T[2,1] - T[1,2])
        w[1,0] = 0.5 * (T[0,2] - T[2,0])
        w[2,0] = 0.5 * (T[1,0] - T[0,1])

        return w


    # Calculates the partial derivative of an ETS wrt q
    def _d_ets(self, q):
        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        # The partial derivative of the pose T wrt each joint
        dT = []

        # Compute partial derivatives wrt each joint
        for k in range(self._joints):

            dT.append(np.eye(4))
            l = 0

            for i in range(self.n):

                if self._ets[i]['type'] == 'sym':
                    if i == self._q_idx[k]:
                        #  Multiply by the partial differentiation
                        s = self._ets[i]['s']
                        dT[k] = dT[k] @ self._ets[i]['dT'](s, q[k])
                        l += 1
                    else:
                        s = self._ets[i]['s']
                        dT[k] = dT[k] @ self._ets[i]['T'](s, q[l])
                        l += 1
                    
                else:
                    # The current T is static
                    dT[k] = dT[k] @ self._ets[i]['T']
        
        return dT

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict


    # Calculates the second partial derivative of an ETS wrt q
    def _dd_ets(self, q):
        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        ddT = np.zeros((self._joints, self._joints, 4, 4))
        thr = []
        for k in range(self._joints):
            thr.append(self.pool.apply_async(self._dd_ets_inner, [k, q]))

        for k in range(self._joints):
            ddT[k,:,:,:] =  thr[k].get()                

        return ddT

    
    def _dd_ets_inner(self, k, q):
        ddT = np.zeros((self._joints, 4, 4))

        for j in range(self._joints):

            # Initialise each matrix
            ddT[j,:,:] = np.eye(4)
            l = 0

            # Double partial derivative
            for i in range(self.n):

                if self._ets[i]['type'] == 'sym':

                    s = self._ets[i]['s']

                    if i == self._q_idx[k] and i == self._q_idx[j]:
                        # Multiply by the double partial differentiation
                        ddT[j,:,:] = ddT[j,:,:] @ self._ets[i]['ddT'](s, q[k])
                        l += 1
                    elif i == self._q_idx[k]: 
                        #  Multiply by the partial differentiation
                        ddT[j,:,:] = ddT[j,:,:] @ self._ets[i]['dT'](s, q[k])
                        l += 1
                    elif i == self._q_idx[j]:
                        #  Multiply by the partial differentiation
                        ddT[j,:,:] = ddT[j,:,:] @ self._ets[i]['dT'](s, q[j])
                        l += 1
                    else:
                        ddT[j,:,:] = ddT[j,:,:] @ self._ets[i]['T'](s, q[l])
                        l += 1
                    
                else:
                    # The current T is static
                    ddT[j,:,:] = ddT[j,:,:] @ self._ets[i]['T']

        return ddT



    # Helper functions
    def _R(self, axis, q):
        if axis == 'Rx':
            return self._Rx(q)
        elif axis == 'Ry':
            return self._Ry(q)
        elif axis == 'Rz':
            return self._Rz(q)
        else:
            raise ValueError('Not a valid axis, axis must be: "Rx", "Ry", or "Rz"')

    def _dR(self, axis, q):
        if axis == 'Rx':
            return self._dRx(q)
        elif axis == 'Ry':
            return self._dRy(q)
        elif axis == 'Rz':
            return self._dRz(q)
        else:
            raise ValueError('Not a valid axis, axis must be: "Rx", "Ry", or "Rz"')

    def _ddR(self, axis, q):
        if axis == 'Rx':
            return self._ddRx(q)
        elif axis == 'Ry':
            return self._ddRy(q)
        elif axis == 'Rz':
            return self._ddRz(q)
        else:
            raise ValueError('Not a valid axis, axis must be: "Rx", "Ry", or "Rz"')

    def _T(self, axis, q):
        if axis == 'Tx':
            return self._Tx(q)
        elif axis == 'Ty':
            return self._Ty(q)
        elif axis == 'Tz':
            return self._Tz(q)
        else:
            raise ValueError('Not a valid axis, axis must be: "Tx", "Ty", or "Tz"')

    def _dT(self, axis, q):
        if axis == 'Tx':
            return self._dTx(q)
        elif axis == 'Ty':
            return self._dTy(q)
        elif axis == 'Tz':
            return self._dTz(q)
        else:
            raise ValueError('Not a valid axis, axis must be: "Tx", "Ty", or "Tz"')

    def _ddT(self, axis, q):
        if axis == 'Tx':
            return self._ddTx(q)
        elif axis == 'Ty':
            return self._ddTy(q)
        elif axis == 'Tz':
            return self._ddTz(q)
        else:
            raise ValueError('Not a valid axis, axis must be: "Tx", "Ty", or "Tz"')

    # Each of the 6 elementary transforms as a function of q
    def _Rx(self, q):
        return np.array([
            [1,  0,          0,         0],
            [0,  np.cos(q), -np.sin(q), 0],
            [0,  np.sin(q),  np.cos(q), 0],
            [0,  0,          0,         1]
        ])

    def _Ry(self, q):
        return np.array([
            [ np.cos(q),  0, np.sin(q), 0],
            [ 0,          1, 0,         0],
            [-np.sin(q),  0, np.cos(q), 0],
            [ 0,          0, 0,         1]
        ])

    def _Rz(self, q):
        return np.array([
            [np.cos(q), -np.sin(q), 0, 0],
            [np.sin(q),  np.cos(q), 0, 0],
            [0,          0,         1, 0],
            [0,          0,         0, 1]
        ])

    def _Tx(self, q):
        return np.array([
            [1, 0, 0, q],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def _Ty(self, q):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, q],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

    def _Tz(self, q):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, q],
            [0, 0, 0, 1]
        ])

    # Precomputation of the first partial derivative wrt q of the 6  
    # elementary transforms
    # Partial derivative of R is skew(a) * R where skew(a) is the 
    # corresponding skew symmetric matrix for R
    def _dRx(self, q):
        return np.array([
            [0,  0,          0,         0],
            [0, -np.sin(q), -np.cos(q), 0],
            [0,  np.cos(q), -np.sin(q), 0],
            [0,  0,          0,         0]
        ])

    def _dRy(self, q):
        return np.array([
            [ -np.sin(q),  0,  np.cos(q), 0],
            [  0,          0,  0,         0],
            [ -np.cos(q),  0, -np.sin(q), 0],
            [  0,          0,  0,         0]
        ])

    def _dRz(self, q):
        return np.array([
            [ -np.sin(q), -np.cos(q), 0, 0],
            [  np.cos(q), -np.sin(q), 0, 0],
            [  0,          0,         0, 0],
            [  0,          0,         0, 0]
        ])

    def _dTx(self, q):
        return np.array([
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    def _dTy(self, q):
        return np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0]
        ])

    def _dTz(self, q):
        return np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0]
        ])

    # Precomputation of the second partial derivative wrt q of the 6  
    # elementary transforms
    # Second partial derivative of R is skew(a) * skew(a) * R where 
    # skew(a) is the corresponding skew symmetric matrix for R
    def _ddRx(self, q):
        return np.array([
            [0,  0,          0,         0],
            [0, -np.cos(q),  np.sin(q), 0],
            [0, -np.sin(q), -np.cos(q), 0],
            [0,  0,          0,         0]
        ])

    def _ddRy(self, q):
        return np.array([
            [ -np.cos(q),  0, -np.sin(q), 0],
            [  0,          0,  0,         0],
            [  np.sin(q),  0, -np.cos(q), 0],
            [  0,          0,  0,         0]
        ])

    def _ddRz(self, q):
        return np.array([
            [ -np.cos(q),  np.sin(q), 0, 0],
            [ -np.sin(q), -np.cos(q), 0, 0],
            [  0,          0,         0, 0],
            [  0,          0,         0, 0]
        ])

    def _ddTx(self, q):
        return np.zeros(4)

    def _ddTy(self, q):
        return np.zeros(4)

    def _ddTz(self, q):
        return np.zeros(4)



    def jac(self, q):

        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        Jt = np.zeros((3,self._joints))
        
        Jw = np.zeros((3,3,self._joints))

        t0 = time.time()

        T = self.fkine(q)
        R = T[0:3,0:3]

        t1 = time.time()

        dT = self._d_ets(q)

        t2 = time.time()

        for i in range(self._joints):

            # Linear velocity component of the Jacobian
            Jt[:,i] = dT[i][0:3,3]

            Jw[:,:,i] = dT[i][0:3,0:3] @ np.transpose(R)

        t3 = time.time()

        # print(t1-t0)
        # print(t2-t1)
        # print(t3-t2)

        return Jt, Jw




    def hess(self, q):

        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        H = np.zeros((6,self._joints,self._joints))

        Jt, Jw = self.jac(q)
      
        for j in range(self._joints):
            for i in range(j, self._joints):

                H[:3,i,j] = Jw[:,:,j] @ Jt[:3,i]

                H[3:,i,j] = np.squeeze(self._vex(
                    Jw[:,:,j] @ Jw[:,:,i] + Jw[:,:,i] @ -Jw[:,:,j]
                ))
                
                if i != j:
                    H[:3,j,i] = H[:3,i,j]

        return H

    def skew(self, a):

        return np.array([
            [ 0   , -a[2],  a[1]],
            [ a[2],  0   , -a[0]],
            [-a[1],  a[0],  0   ]
        ])


    def jac_fast(self, q):

        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        J = np.zeros((6,self._joints))
        Jt = np.zeros((3,self._joints))
        Jw = np.zeros((3,3,self._joints))

        T = self.fkine(q)
        R = T[0:3,0:3]

        dT = self._d_ets(q)

        for i in range(self._joints):

            # Linear velocity component of the Jacobian
            Jt[:,i] = dT[i][0:3,3]
            J[0:3,i] = dT[i][0:3,3]

            temp = dT[i][0:3,0:3] @ np.transpose(R)
            Jw[:,:,i] = temp
            J[3:7,i] = np.squeeze(self._vex( temp ))

        return J, Jt, Jw


    def hess_fast(self, q, Jt, Jw):

        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        H = np.zeros((6,self._joints,self._joints))
      
        for j in range(self._joints):
            for i in range(j, self._joints):

                H[:3,i,j] = Jw[:,:,j] @ Jt[:3,i]

                H[3:,i,j] = np.squeeze(self._vex(
                    Jw[:,:,j] @ Jw[:,:,i] + Jw[:,:,i] @ -Jw[:,:,j]
                ))
                
                if i != j:
                    H[:3,j,i] = H[:3,i,j]

        return H


    def m_fast(self, q, J):
        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        return np.sqrt(np.linalg.det(J @ np.transpose(J)))


    def Jm_fast(self, q):
        if not isinstance(q, np.ndarray):
            raise TypeError('q array must be a numpy ndarray.')
        if q.shape != (self._joints,):
            raise ValueError('q must be a 1 dim (n,) array')

        t0 = time.time()
        J, Jt, Jw = self.jac_fast(q)

        t1 = time.time()
        m = self.m_fast(q, J)

        t2 = time.time()
        H = self.hess_fast(q, Jt, Jw)

        t3 = time.time()
        b = np.linalg.inv(J @ np.transpose(J))

        t4 = time.time()
        a = np.zeros((self._joints,1))

        for i in range(self._joints):
            c = J @ np.transpose(H[:,:,i])
            a[i,0] = m * np.transpose(c.flatten('F')) @ b.flatten('F')

        t5 = time.time()

        # print(t1 - t0)
        # print(t2 - t1)
        # print(t3 - t2)
        # print(t4 - t3)
        # print(t5 - t4)

        # print('\n')

        return a
