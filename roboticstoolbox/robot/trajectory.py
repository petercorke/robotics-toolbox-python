"""
Trajectory primitives.

Python implementation by: Luis Fernando Lara Tobar and Peter Corke.
Based on original Robotics Toolbox for Matlab code by Peter Corke.
Permission to use and copy is granted provided that acknowledgement of
the authors is made.

@author: Luis Fernando Lara Tobar and Peter Corke

Edited 3/06 Samuel Drew
"""

import numpy as np
from spatialmath.base.argcheck import *


def jtraj(q0, q1, tv, qd0=None, qd1=None):
    """
    Compute a joint space trajectory between points C{q0} and C{q1}.
    The number of points is the length of the given time vector C{tv}.  If
    {tv} is a scalar it is taken as the number of points.
    
    A 7th order polynomial is used with default zero boundary conditions for
    velocity and acceleration.  Non-zero boundary velocities can be
    optionally specified as C{qd0} and C{qd1}.
    
    As well as the trajectory, M{q{t}}, its first and second derivatives
    M{qd(t)} and M{qdd(t)} are also computed.  All three are returned as a tuple.
    Each of these is an M{m x n} matrix, with one row per time step, and
    one column per joint parameter.

    @type q0: m-vector
    @param q0: initial state
    @type q1: m-vector
    @param q1: final state
    @type tv: n-vector or scalar
    @param tv: time step vector or number of steps
    @type qd0: m-vector
    @param qd0: initial velocity (default 0)
    @type qd1: m-vector
    @param qd1: final velocity (default 0)
    @rtype: tuple
    @return: (q, qd, qdd), a tuple of M{m x n} matrices
    @see: L{ctraj}

    """

    if isscalar(tv):
        tscal = 1
        t = np.array([range(0, tv)]).T/(tv-1) # Normalized time from 0 -> 1
    else:
        tv = getvector(tv)
        tscal = max(tv)
        t = tv[np.newaxis].T / tscal
    
    q0 = getvector(q0)
    q1 = getvector(q1)
    
    if qd0 == None:
        qd0 = np.zeros(q0.shape)
    else:
        qd0 = getvector(qd0);
    if qd1 == None:
        qd1 = np.zeros(q1.shape)
    else:
        qd1 = getvector(qd1)
    
    # compute the polynomial coefficients
    a = 6*(q1 - q0) - 3*(qd1 + qd0)*tscal
    b = -15*(q1 - q0) + (8*qd0 + 7*qd1)*tscal
    c = 10*(q1 - q0) - (6*qd0 + 4*qd1)*tscal
    e = qd0*tscal # as the t vector has been normalized
    f = q0

    tt = np.concatenate((np.power(t, 5),
                         np.power(t, 4),
                         np.power(t, 3),
                         np.power(t, 2),
                         t, np.ones(t.shape)),1)
    c = np.vstack((a, b, c, np.zeros(a.shape), e, f))
    qt = np.dot(tt, c)

    # # compute velocity
    # c = vstack((zeros(shape(a)),5*a,4*b,3*c,zeros(shape(a)),e))
    # qdt = tt * c / tscal
    #
    #
    # # compute acceleration
    # c = vstack((zeros(shape(a)),zeros(shape(a)),20*a,12*b,6*c,zeros(shape(a))))
    # qddt = tt * c / (tscal**2)

    # return qt,qdt,qddt
    return qt


# def ctraj(t0, t1, r):
#     """
#     Compute a Cartesian trajectory between poses C{t0} and C{t1}.
#     The number of points is the length of the path distance vector C{r}.
#     Each element of C{r} gives the distance along the path, and the
#     must be in the range [0 1].
#
#     If {r} is a scalar it is taken as the number of points, and the points
#     are equally spaced between C{t0} and C{t1}.
#
#     The trajectory is a list of transform matrices.
#
#     @type t0: homogeneous transform
#     @param t0: initial pose
#     @rtype: list of M{4x4} matrices
#     @return: Cartesian trajectory
#     @see: L{trinterp}, L{jtraj}
#     """
#
#     if isinstance(r,(int,int32,float,float64)):
#         i = mat(range(1,r+1))
#         r = (i-1.)/(r-1)
#     else:
#         r = arg2array(r);
#
#     if any(r>1) or any(r<0):
#         raise 'path position values (R) must 0<=R<=1'
#     traj = []
#     for s in r.T:
#         traj.append( T.trinterp(t0, t1, float(s)) )
#     return traj
    

