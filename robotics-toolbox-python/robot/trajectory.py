
"""
Trajectory primitives.

@author: Peter Corke
@copyright: Peter Corke
"""

from numpy import *
from utility import *
import transform as T

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

    if isinstance(tv,(int,int32,float,float64)):
        tscal = float(1)
        t = mat(range(0,tv)).T/(tv-1.) # Normalized time from 0 -> 1
    else:
        tv = arg2array(tv);
        tscal = float(max(tv))
        t = mat(tv).T / tscal
    
    q0 = arg2array(q0)
    q1 = arg2array(q1)
    
    if qd0 == None:
        qd0 = zeros((shape(q0)))
    else:
        qd0 = arg2array(qd0);
    if qd1 == None:
        qd1 = zeros((shape(q1)))
    else:
        qd1 = arg2array(qd1)

    print qd0
    print qd1
    
    # compute the polynomial coefficients
    A = 6*(q1 - q0) - 3*(qd1 + qd0)*tscal
    B = -15*(q1 - q0) + (8*qd0 + 7*qd1)*tscal
    C = 10*(q1 - q0) - (6*qd0 + 4*qd1)*tscal
    E = qd0*tscal # as the t vector has been normalized
    F = q0

    tt = concatenate((power(t,5),power(t,4),power(t,3),power(t,2),t,ones(shape(t))),1)
    c = vstack((A, B, C, zeros(shape(A)), E, F))
    qt = tt * c

    # compute velocity
    c = vstack((zeros(shape(A)),5*A,4*B,3*C,zeros(shape(A)),E))
    qdt = tt * c / tscal


    # compute acceleration
    c = vstack((zeros(shape(A)),zeros(shape(A)),20*A,12*B,6*C,zeros(shape(A))))
    qddt = tt * c / (tscal**2)

    return qt,qdt,qddt


def ctraj(t0, t1, r):
    """
    Compute a Cartesian trajectory between poses C{t0} and C{t1}.
    The number of points is the length of the path distance vector C{r}.
    Each element of C{r} gives the distance along the path, and the 
    must be in the range [0 1].

    If {r} is a scalar it is taken as the number of points, and the points 
    are equally spaced between C{t0} and C{t1}.

    The trajectory is a list of transform matrices.
    
    @type t0: homogeneous transform
    @param t0: initial pose
    @rtype: list of M{4x4} matrices
    @return: Cartesian trajectory
    @see: L{trinterp}, L{jtraj}
    """

    if isinstance(r,(int,int32,float,float64)):
        i = mat(range(1,r+1))
        r = (i-1.)/(r-1)
    else:
        r = arg2array(r);

    if any(r>1) or any(r<0):
        raise 'path position values (R) must 0<=R<=1'
    traj = []
    for s in r.T:
        traj.append( T.trinterp(t0, t1, float(s)) )
    return traj
    

