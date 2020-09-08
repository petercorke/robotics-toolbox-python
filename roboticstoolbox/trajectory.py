import numpy as np
import math
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

# -------------------------------------------------------------------------- #

def ctraj():
    pass

def cmstraj():
    pass

# -------------------------------------------------------------------------- #

def mstraj(viapoints, dt, tacc, qdmax=None, tsegment=None, q0=None, qd0=None, qdf=None, verbose=False):

    """
    MSTRAJ Multi-segment multi-axis trajectory
    
    :param viapoints: A set of viapoints, one per row
    :type viapoints: numpy.ndarray
    :param dt: time step
    :type dt: float (seconds)
    :param tacc: acceleration time (seconds)
    :type tacc: float
    :param qdmax: maximum joint speed, defaults to None
    :type qdmax: array_like or float, optional
    :param tsegment: maximum time of each motion segment (seconds), defaults to None
    :type tsegment: array_like, optional
    :param q0: initial joint coordinates, defaults to first row of viapoints
    :type q0: array_like, optional
    :param qd0: inital joint velocity, defaults to zero
    :type qd0: array_like, optional
    :param qdf: final joint velocity, defaults to zero
    :type qdf: array_like, optional
    :param verbose: print debug information, defaults to False
    :type verbose: bool, optional
    :return: trajectory plus extra info
    :rtype: namedtuple

     ``TRAJ = MSTRAJ(WP, QDMAX, TSEG, Q0, DT, TACC)`` is a trajectory
     (KxN) for N axes moving simultaneously through M segment.  Each segment
     is linear motion and polynomial blends connect the viapoints.  The axes
     start at ``Q0`` (1xN) if given and pass through the via points defined by the rows of
     the matrix WP (MxN), and finish at the point defined by the last row of WP.
     The  trajectory matrix has one row per time step, and one column per
     axis.  The number of steps in the trajectory K is a function of the
     number of via points and the time or velocity limits that apply.

     - WP (MxN) is a matrix of via points, 1 row per via point, one column 
       per axis.  The last via point is the destination.
     - QDMAX (1xN) are axis speed limits which cannot be exceeded,
     - TSEG (1xM) are the durations for each of the K viapoints
     - Q0 (1xN) are the initial axis coordinates
     - DT is the time step
     - TACC (1x1) is the acceleration time used for all segment transitions
     - TACC (1xM) is the acceleration time per segment, TACC(i) is the acceleration 
       time for the transition from segment i to segment i+1.  TACC(1) is also 
       the acceleration time at the start of segment 1.

     TRAJ = MSTRAJ(WP, QDMAX, TSEG, [], DT, TACC, OPTIONS) as above but the
     initial coordinates are taken from the first row of WP.

     TRAJ = MSTRAJ(WP, QDMAX, Q0, DT, TACC, QD0, QDF, OPTIONS) as above
     but additionally specifies the initial and final axis velocities (1xN).


     Notes::
     - Only one of QDMAX or TSEG can be specified, the other is set to [].
     - If no output arguments are specified the trajectory is plotted.
     - The path length K is a function of the number of via points, Q0, DT
       and TACC.
     - The final via point P(end,:) is the destination.
     - The motion has M viapoints from Q0 to P(1,:) to P(2,:) ... to P(end,:).
     - All axes reach their via points at the same time.
     - Can be used to create joint space trajectories where each axis is a joint
       coordinate.
     - Can be used to create Cartesian trajectories where the "axes"
       correspond to translation and orientation in RPY or Euler angle form.
     - If qdmax is a scalar then all axes are assumed to have the same
       maximum speed.

     See also MTRAJ, LSPB, CTRAJ.

     Copyright (C) 1993-2017, by Peter I. Corke
    """

    if q0 is None:
        q0 = viapoints[0,:]
        viapoints = viapoints[1:,:]
    else:
        assert viapoints.shape[1] == len(q0), 'WP and Q0 must have same number of columns'

    ns, nj = viapoints.shape
    Tacc = tacc
    
    assert not (qdmax is not None and tsegment is not None), 'cannot specify both qdmax and tsegment'
    if qdmax is None:
        assert tsegment is not None, 'tsegment must be given if qdmax is not'
        assert len(tsegment) == ns, 'Length of TSEG does not match number of viapoints'
    if tsegment is None:
        assert qdmax is not None, 'qdmax must be given if tsegment is not'
        if isinstance(qdmax, (int, float)):
            # if qdmax is a scalar assume all axes have the same speed
            qdmax = np.tile(qdmax, (nj,))
        else:
            assert len(qdmax) == nj, 'Length of QDMAX does not match number of axes'
    
    if isinstance(Tacc, (int, float)):
        Tacc = np.tile(Tacc, (ns,))
    else:
        assert len(Tacc) == ns, 'Tacc is wrong size'
    if qd0 is None:
        qd0 = np.zeros((nj,))
    else:
        assert len(qd0) == len(q0), 'qd0 is wrong size'
    if qdf is None:
        qdf = np.zeros((nj,))
    else:
        assert len(qdf) == len(q0), 'qdf is wrong size'

    # set the initial conditions
    q_prev = q0;
    qd_prev = qd0;

    clock = 0     # keep track of time
    arrive = np.zeros((ns,))   # record planned time of arrival at via points
    tg = np.zeros((0,nj))
    infolist = []
    info = namedtuple('mstraj_info', 'slowest segtime axtime clock')

    for seg in range(0, ns):
        if verbose:
            print('------------------- segment %d\n' % (seg,))

        # set the blend time, just half an interval for the first segment

        tacc = Tacc[seg]

        tacc = math.ceil(tacc / dt) * dt
        tacc2 = math.ceil(tacc / 2 / dt) * dt
        if seg == 0:
            taccx = tacc2
        else:
            taccx = tacc

        # estimate travel time
        #    could better estimate distance travelled during the blend
        q_next = viapoints[seg,:]    # current target
        dq = q_next - q_prev    # total distance to move this segment

        ## probably should iterate over the next section to get qb right...
        # while 1
        #   qd_next = (qnextnext - qnext)
        #   tb = abs(qd_next - qd) ./ qddmax;
        #   qb = f(tb, max acceleration)
        #   dq = q_next - q_prev - qb
        #   tl = abs(dq) ./ qdmax;

        if qdmax is not None:
            # qdmax is specified, compute slowest axis

            qb = taccx * qdmax / 2       # distance moved during blend
            tb = taccx

            # convert to time
            tl = abs(dq) / qdmax
            #tl = abs(dq - qb) / qdmax
            tl = np.ceil(tl / dt) * dt

            # find the total time and slowest axis
            tt = tb + tl
            slowest = np.argmax(tt)
            tseg = tt[slowest]

            infolist.append(info(slowest, tseg, tt, clock))

            # best if there is some linear motion component
            if tseg <= 2*tacc:
                tseg = 2 * tacc

        elif tsegment is not None:
            # segment time specified, use that
            tseg = tsegment[seg]
            slowest = math.nan

        # log the planned arrival time
        arrive[seg] = clock + tseg
        if seg > 0:
            arrive[seg] += tacc2

        if verbose:
            print('seg %d, slowest axis %d, time required %.4g\n' % (seg, slowest, tseg))

        ## create the trajectories for this segment

        # linear velocity from qprev to qnext
        qd = dq / tseg

        # add the blend polynomial
        print(jtraj)
        qb = jtraj.jtraj(q0, q_prev + tacc2 * qd, np.arange(0, taccx, dt), qd0=qd_prev, qd1=qd).q
        tg = np.vstack([tg, qb[1:,:]])

        clock = clock + taccx     # update the clock

        # add the linear part, from tacc/2+dt to tseg-tacc/2
        for t in np.arange(tacc2 + dt, tseg - tacc2, dt):
            s = t / tseg
            q0 = (1 - s) * q_prev + s * q_next       # linear step
            tg = np.vstack([tg, q0])
            clock += dt

        q_prev = q_next    # next target becomes previous target
        qd_prev = qd

    # add the final blend
    qb = jtraj.jtraj(q0, q_next, np.arange(0, tacc2, dt), qd0=qd_prev, qd1=qdf).q
    tg = np.vstack([tg, qb[1:,:]])

    print(info)

    infolist.append(info(None, tseg, None, clock))
    
    return namedtuple('mstraj', 't q arrive info via')(dt * np.arange(0, tg.shape[0]), tg, arrive, infolist, viapoints)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    path = np.array([
        [10, 10],
        [10, 60],
        [80, 80],
        [50, 10]
        ])
          
    out = mstraj(path, dt=0.1, tacc=5, qdmax=2.5, extra=True)
    print(out.q)
    
    plt.figure()
    plt.plot(out.t, out.q)
    plt.grid(True)
    plt.xlabel('time')
    plt.legend(('$q_0$', '$q_1$'))
    plt.plot(out.arrive, out.viapoints, 'bo')
    
    plt.figure()
    plt.plot(out.q[:,0], out.q[:,1])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    


    #plt.xaxis(t(1), t(end))
        
