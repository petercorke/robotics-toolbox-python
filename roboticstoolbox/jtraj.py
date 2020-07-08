import numpy as np
import spatialmath.base.argcheck as arg
from collections import namedtuple

def  jtraj(q0, q1, tv, qd0=None, qd1=None):
    """
    JTRAJ Compute a joint space trajectory
    
    :param q0: initial coordinate
    :type q0: array_like
    :param q1: final coordinate
    :type q1: array_like
    :param tv: time vector or number of steps
    :type tv: array_like or int
    :param qd0: initial velocity, defaults to zero
    :type qd0: array_like, optional
    :param qd1: final velocity, defaults to zero
    :type qd1: array_like, optional
    :return: trajectory of coordinates and optionally velocity and acceleration
    :return: trajectory of coordinates plus optionally velocity and acceleration
    :rtype: namedtuple


    ``Q = JTRAJ(Q0, QF, M)`` is a joint space trajectory ``Q`` (MxN) where the joint
    coordinates vary from ``Q0`` (1xN) to ``QF`` (1xN).  A quintic (5th order) polynomial is used 
    with default zero boundary conditions for velocity and acceleration.  
    Time is assumed to vary from 0 to 1 in ``M`` steps.  

    ``Q = JTRAJ(Q0, QF, M, QD0, QDF)`` as above but also specifies
    initial ``QD0`` (1xN) and final ``QDF`` (1xN) joint velocity for the trajectory.
    
    ``Q = JTRAJ(Q0, QF, T)`` as above but the number of steps in the
    trajectory is defined by the length of the time vector ``T`` (Mx1).    

    ``Q = JTRAJ(Q0, QF, T, QD0, QDF)`` as above but specifies initial and 
    final joint velocity for the trajectory and a time vector.
    
    The output ``Q`` is an MxN numpy array with one row per time step.
    
    Joint velocity and acceleration can be optionally returned by setting
    ``vel`` or ``accel`` to True. In this case the output is a named tuple
    with elements `q`, `qd` and `qdd`. The shape of the velocity and acceleration
    arrays is the same as for ``Q``.
    
    Notes:
        
    - When a time vector is provided the velocity and acceleration outputs
      are scaled assumign that the time vector starts at zero and increases
      linearly.
    
    See also QPLOT, CTRAJ, SerialLink.jtraj.
    
    Copyright (C) 1993-2017, by Peter I. Corke

    """
    if isinstance(tv, int):
        tscal = 1
        t = np.linspace(0, 1, tv) # normalized time from 0 -> 1
    else:
        tscal = max(tv)
        t = tv.flatten() / tscal

    q0 = arg.getvector(q0)
    q1 = arg.getvector(q1)
    assert len(q0) == len(q1), 'q0 and q1 must be same size'
    
    if qd0 is None:
        qd0 = np.zeros(q0.shape)
    else:
        qd0 = arg.getvector(qd0)
        assert len(qd0) == len(q0), 'qd0 has wrong size'
    if qd1 is None:
        qd1 = np.zeros(q0.shape)
    else:
        qd0 = arg.getvector(qd0)
        assert len(qd1) == len(q0), 'qd1 has wrong size'

    # compute the polynomial coefficients
    A =   6 * (q1 - q0) - 3 * (qd1 + qd0) * tscal
    B = -15 * (q1 - q0) + (8 * qd0 + 7 * qd1) * tscal
    C =  10 * (q1 - q0) - (6 * qd0 + 4 * qd1) * tscal
    E =       qd0 * tscal #  as the t vector has been normalized
    F =       q0
    
    n = len(q0)
    
    tt = np.array([t**5, t**4, t**3, t**2, t, np.ones(t.shape)]).T
    coeffs = np.array([A, B, C, np.zeros(A.shape), E, F])
    
    qt = tt @ coeffs
    
    # compute  velocity
    c = np.array([np.zeros(A.shape), 5 * A, 4 * B, 3 * C, np.zeros(A.shape), E])
    qdt = tt @ coeffs / tscal

    # compute  acceleration
    c = np.array([np.zeros(A.shape), np.zeros(A.shape), 20 * A, 12 * B, 6 * C, np.zeros(A.shape)])
    qddt = tt @ coeffs / tscal**2

    return namedtuple('jtraj', 't q qd qdd')(tt, qt, qdt, qddt)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    out = jtraj([0,1], [2, -1], 20)
    print(out)
    
    out = jtraj([0,1], [2, -1], 20, vel=True, accel=True)
    print(out)
    