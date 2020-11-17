import numpy as np
import math
from collections import namedtuple
import matplotlib.pyplot as plt
from spatialmath.base.argcheck import isvector, \
    getvector, assertmatrix


def tpoly(q0, qf, t, qd0=0, qdf=0):
    """
    Generate scalar polynomial trajectory

    :param q0: initial value
    :type q0: float
    :param qf: final value
    :type qf: float
    :param t: time
    :type t: float or array_like
    :param qd0: initial velocity, optional
    :type q0: float
    :param qdf: final velocity, optional
    :type q0: float
    :return: trajectory
    :rtype: namedtuple

    - ``tg = tpoly(q0, q1, t)`` is a scalar trajectory (Mx1) that varies
      smoothly from ``q0`` to ``qf`` using a quintic polynomial.  The initial
      and final velocity and acceleration are zero. Time ``t`` can be either:

        * an integer scalar, indicating the total number of timesteps

            - Velocity is in units of distance per trajectory step, not per
              second.
            - Acceleration is in units of distance per trajectory step squared,
              *not* per second squared.

        * an array_like, containing the time steps.

            - Results are scaled to units of time.

    - ``tg = tpoly(q0, q1, t, qd0, qdf)`` as above but specify the initial and
      final velocity. The initial and final acceleration are zero.

    The return value is a namedtuple (named ``tpoly``) with elements:

        - ``x``  the time coordinate as a numpy ndarray, shape=(M,)
        - ``y``  the position as a numpy ndarray, shape=(M,)
        - ``yd``  the velocity as a numpy ndarray, shape=(M,)
        - ``ydd``  the acceleration as a numpy ndarray, shape=(M,)

    .. note:: The time vector T is assumed to be monotonically increasing, and
        time scaling is based on the first and last element.

    References:

    - Robotics, Vision & Control, Chap 3,
      P. Corke, Springer 2011.

    :seealso: :func:`lspb`, :func:`t1plot`, :func:`jtraj`.
    """

    if isinstance(t, int):
        t = np.arange(0, t)
        istime = False
    elif isvector(t):
        t = getvector(t)
        istime = True
    else:
        raise TypeError('bad argument for time, must be int or vector')

    tf = max(t)
    # solve for the polynomial coefficients using least squares
    X = [
            [0,             0,           0,           0,       0,   1],
            [tf ** 5,       tf ** 4,     tf ** 3,     tf ** 2, tf,  1],
            [0,             0,           0,           0,       1,   0],
            [5 * tf ** 4,   4 * tf ** 3, 3 * tf ** 2, 2 * tf,  1,   0],
            [0,             0,           0,           2,       0,   0],
            [20 * tf ** 3, 12 * tf ** 2, 6 * tf,      2,       0,   0]
    ]
    coeffs, resid, rank, s = np.linalg.lstsq(
        X, np.r_[q0, qf, qd0, qdf, 0, 0], rcond=None)

    # coefficients of derivatives
    coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
    coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

    # evaluate the polynomials
    p = np.polyval(coeffs, t)
    pd = np.polyval(coeffs_d, t)
    pdd = np.polyval(coeffs_dd, t)

    return namedtuple('tpoly', 'x y yd ydd istime')(t, p, pd, pdd, istime)


# -------------------------------------------------------------------------- #

def lspb(q0, qf, t, V=None):
    """
    Scalar trapezoidal trajectory

    :param q0: initial value
    :type q0: float
    :param qf: final value
    :type qf: float
    :param t: time
    :type t: float or array_like
    :param V: velocity of linear segment, optional
    :type V: float
    :return: trajectory
    :rtype: namedtuple

    Computes a trapezoidal trajectory, which has a linear motion segment with
    parabolic blends.

    - ``tg = lspb(q0, qf, t)`` is a scalar trajectory (Mx1) that varies
      smoothly from ``q0`` to ``qf`` in M steps using a constant velocity
      segment and parabolic blends.  Time ``t`` can be either:

        * an integer scalar, indicating the total number of timesteps

            - Velocity is in units of distance per trajectory step, not per
              second.
            - Acceleration is in units of distance per trajectory step squared,
              *not* per second squared.

        * an array_like, containing the time steps.

            - Results are scaled to units of time.

    - ``tg = lspb(q0, q1, t, V)``  as above but specifies the velocity of the
      linear segment which is normally computed automatically.

    The return value is a namedtuple (named ``lspb``) with elements:

        - ``x``  the time coordinate as a numpy ndarray, shape=(M,)
        - ``y``  the position as a numpy ndarray, shape=(M,)
        - ``yd``  the velocity as a numpy ndarray, shape=(M,)
        - ``ydd``  the acceleration as a numpy ndarray, shape=(M,)

    .. note::

        - For some values of V no solution is possible and an error is flagged.
        - The time vector, if given, is assumed to be monotonically increasing,
          and time scaling is based on the first and last element.

    :References:

        - Robotics, Vision & Control, Chap 3,
        P. Corke, Springer 2011.

    :seealso: :func:`tpoly`, :func:`t1plot`, :func:`jtraj`.
    """

    if isinstance(t, int):
        t = np.arange(0, t)
        istime = False
    elif isvector(t):
        t = getvector(t)
        istime = True
    else:
        raise TypeError('bad argument for time, must be int or vector')

    tf = max(t)

    if V is None:
        # if velocity not specified, compute it
        V = (qf - q0) / tf * 1.5
    else:
        V = abs(V) * np.sign(qf - q0)
        if abs(V) < (abs(qf - q0) / tf):
            raise ValueError('V too small')
        elif abs(V) > (2 * abs(qf-q0) / tf):
            raise ValueError('V too big')

    if q0 == qf:
        # Commented these because they arent used anywhere
        # s = np.ones((len(t), len(t))) @ q0
        # sd = np.zeros((len(t), len(t)))
        # sdd = np.zeros((len(t), len(t)))
        return

    tb = (q0 - qf + V * tf) / V
    a = V / tb

    p = np.zeros((len(t),))
    pd = np.zeros((len(t),))
    pdd = np.zeros((len(t),))

    for i, _t in enumerate(t):

        if _t <= tb:
            # initial blend
            p[i] = q0 + a/2 * _t ** 2
            pd[i] = a * _t
            pdd[i] = a
        elif _t <= (tf - tb):
            # linear motion
            p[i] = (qf + q0 - V * tf) / 2 + V * _t
            pd[i] = V
            pdd[i] = 0
        else:
            # final blend
            p[i] = qf - a / 2 * tf ** 2 + a * tf * _t - a / 2 * _t ** 2
            pd[i] = a * tf - a * _t
            pdd[i] = -a

    return namedtuple(
        'lspb', 'x y yd ydd xblend istime')(t, p, pd, pdd, tb, istime)


# -------------------------------------------------------------------------- #

def jtraj(q0, qf, tv, qd0=None, qd1=None):
    """
    Compute a joint-space trajectory

    :param q0: initial joint coordinate
    :type q0: N-element array_like
    :param qf: final joint coordinate
    :type qf: N-element array_like
    :param tv: time vector or number of steps
    :type tv: array_like or int
    :param qd0: initial velocity, defaults to zero
    :type qd0: N-element array_like, optional
    :param qd1: final velocity, defaults to zero
    :type qd1: N-element array_like, optional
    :return: trajectory of coordinates and optionally velocity and acceleration
    :rtype: namedtuple

    - ``tg = jtraj(q0, qf, M)`` is a joint space trajectory where the joint
      coordinates vary from ``q0`` (N) to ``qf`` (N).  A quintic (5th order)
      polynomial is used with default zero boundary conditions for velocity and
      acceleration.
      Time is assumed to vary from 0 to 1 in ``M`` steps.

    - ``tg = jtraj(q0, qf, M, qd0, qdf)`` as above but also specifies initial
      ``qd0`` (N) and final ``qdf`` (N) joint velocity for the trajectory.

    - ``tg = jtraj(q0, qf, tv)``, as above but the number of steps in the
      trajectory is defined by the length of the time vector ``tv`` (M).

    - ``tg = jtraj(q0, qf, tv, qd0, qdf)`` as above but specifies initial and
      final joint velocity for the trajectory and a time vector.

    The return value is a namedtuple (named ``jtraj``) with elements:

        - ``t``  the time coordinate as a numpy ndarray, shape=(M,)
        - ``q``  the position as a numpy ndarray, shape=(M,N)
        - ``qd``  the velocity as a numpy ndarray, shape=(M,N)
        - ``qdd``  the acceleration as a numpy ndarray, shape=(M,N)

    Notes:

    - When a time vector is provided the velocity and acceleration outputs
      are scaled assuming that the time vector starts at zero and increases
      linearly.

    :seealso: :func:`ctraj`, :func:`qplot`, :func:`~SerialLink.jtraj`
    """
    # print(f"  --- jtraj: {q0} --> {q1} in {tv}")
    if isinstance(tv, int):
        tscal = 1.0
        t = np.linspace(0, 1, tv)  # normalized time from 0 -> 1
    else:
        tscal = max(tv)
        t = tv.flatten() / tscal

    q0 = getvector(q0)
    qf = getvector(qf)

    if not len(q0) == len(qf):
        raise ValueError('q0 and q1 must be same size')

    if qd0 is None:
        qd0 = np.zeros(q0.shape)
    else:
        qd0 = getvector(qd0)
        if not len(qd0) == len(q0):
            raise ValueError('qd0 has wrong size')
    if qd1 is None:
        qd1 = np.zeros(q0.shape)
    else:
        qd0 = getvector(qd0)
        if not len(qd1) == len(q0):
            raise ValueError('qd1 has wrong size')

    # compute the polynomial coefficients
    A = 6 * (qf - q0) - 3 * (qd1 + qd0) * tscal
    B = -15 * (qf - q0) + (8 * qd0 + 7 * qd1) * tscal
    C = 10 * (qf - q0) - (6 * qd0 + 4 * qd1) * tscal
    E = qd0 * tscal  # as the t vector has been normalized
    F = q0

    # n = len(q0)

    tt = np.array([t**5, t**4, t**3, t**2, t, np.ones(t.shape)]).T
    coeffs = np.array([A, B, C, np.zeros(A.shape), E, F])

    qt = tt @ coeffs

    # compute  velocity
    coeffs = np.array(
        [np.zeros(A.shape), 5 * A, 4 * B, 3 * C, np.zeros(A.shape), E])
    qdt = tt @ coeffs / tscal

    # compute  acceleration
    coeffs = np.array([
        np.zeros(A.shape), np.zeros(A.shape),
        20 * A, 12 * B, 6 * C, np.zeros(A.shape)])
    qddt = tt @ coeffs / tscal ** 2

    return namedtuple('jtraj', 't q qd qdd')(tt, qt, qdt, qddt)


def t1plot(tg, block=True):
    """
    Plot 1D trajectories

    :param tg: the trajectory to plot
    :type tg: namedtuple

    Plot the position, velocity and acceleration contained in the namedtuple
    ``tg``.

    :seealso: :func:`~tpoly`, :func:`~lspb`
    """

    plotargs = {'markersize': 3}
    textargs = {'fontsize': 12}

    plt.figure()
    ax = plt.subplot(3, 1, 1)

    # plot position
    if type(tg).__name__ == 'tpoly':
        ax.plot(tg.x, tg.y, '-o', **plotargs)

    elif type(tg).__name__ == 'lspb':
        # accel phase
        tf = tg.x[-1]
        k = tg.x <= tg.xblend
        ax.plot(tg.x[k], tg.y[k], 'ro-', **plotargs)

        # coast phase
        k = (tg.x > tg.xblend) & (tg.x <= (tf-tg.xblend))
        ax.plot(tg.x[k], tg.y[k], 'bo-', **plotargs)
        k = np.where(k)[0][0]
        ax.plot(tg.x[k-1:k+1], tg.y[k-1:k+1], 'b-', **plotargs)

        # decel phase
        k = tg.x > (tf - tg.xblend)
        ax.plot(tg.x[k], tg.y[k], 'go-', **plotargs)
        k = np.where(k)[0][0]
        ax.plot(tg.x[k-1:k+1], tg.y[k-1:k+1], 'g-', **plotargs)

        ax.grid(True)
    else:
        raise TypeError('unknown 1D trajectory tuple')

    ax.grid(True)

    if tg.istime:
        ax.set_ylabel('$s(t)$', **textargs)
    else:
        ax.set_ylabel('$s(k)$', **textargs)

    # plot velocity
    ax = plt.subplot(3, 1, 2)
    ax.plot(tg.x, tg.yd, '-o', **plotargs)
    ax.grid(True)

    if tg.istime:
        ax.set_ylabel('$ds/dt$', **textargs)
    else:
        ax.set_ylabel('$ds/dk$', **textargs)

    # plot acceleration
    ax = plt.subplot(3, 1, 3)
    ax.plot(tg.x, tg.ydd, '-o', **plotargs)
    ax.grid(True)
    if tg.istime:
        ax.set_ylabel('$ds^2/dt^2$', **textargs)
        ax.set_xlabel('t (seconds)')
    else:
        ax.set_ylabel('$ds^2/dk^2$', **textargs)
        ax.set_xlabel('k (step))')

    plt.show(block=block)


def qplot(q, t=None, block=True):
    """
    Plot robot joint angles

    :param q: joint angle trajectory
    :type q: numpy ndarray, shape=(M,N)
    :param t: time vector, optional
    :type t: numpy ndarray, shape=(M,)

    This is a convenience function to plot joint angle trajectories (MxN) for
    an N-axis robot, where each row represents one time step.

    - ``qplot(q)`` plots the joint angles versus row number.  If N==6 a
      conventional 6-axis robot is assumed, and the first three joints are
      shown as solid lines, the last three joints (wrist) are shown as dashed
      lines. A legend is also displayed.

    - ``qplot(q, t)`` as above but displays the joint angle trajectory versus
      time given the time vector T (Mx1).

    :seealso: :func:`jtraj`
    """
    assertmatrix(q)

    if t is None:
        t = np.arange(0, q.shape[0])

    n = q.shape[1]
    fig, ax = plt.subplots()
    if n == 6:
        plt.plot(t, q[:, 0:3])
        plt.plot(t, q[:, 3:6], '--')
    else:
        plt.plot(t, q)

    ax.legend([f"q{i+1}" for i in range(n)])

    plt.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint coordinates (rad,m)')
    ax.set_xlim(t[0], t[-1])

    plt.show(block=block)


# -------------------------------------------------------------------------- #

def ctraj(T0, T1, s):
    """
    Cartesian trajectory between two poses

    :param T0: initial pose
    :type T0: SE3
    :param T1: final pose
    :type T1: SE3
    :return T0: smooth path from ``T0`` to ``T1``
    :rtype: SE3

    ``ctraj(T0, T1, n)`` is a Cartesian trajectory from SE3 pose ``T0`` to
    ``T1`` with ``n`` points that follow a trapezoidal velocity profile along
    the path. The Cartesian trajectory is an SE3 instance containing ``n``
    values.

    ``ctraj(T0, T1, s)`` as above but the elements of ``s`` specify the
    fractional distance  along the path, and these values are in the
    range [0 1]. The i'th point corresponds to a distance ``s[i]`` along
    the path.

    Examples::

        >>> tg = ctraj(SE3.Rand(), SE3.Rand(), 20)
        >>> len(tg)
        20

    Notes:

    - In the second case ``s`` could be generated by a scalar trajectory
      generator such as ``tpoly`` or ``lspb`` (default).
    - Orientation interpolation is performed using unit-quaternion
      interpolation.

    Reference:

    - Robotics, Vision & Control, Sec 3.1.5,
      Peter Corke, Springer 2011

    :seealso: :func:`~roboticstoolbox.trajectory.lspb`,
        :func:`~spatialmath.unitquaternion.interp`
    """

    if isinstance(s, int):
        s = lspb(0, 1, s).y
    elif isvector(s):
        s = getvector(s)
    else:
        raise TypeError('bad argument for time, must be int or vector')

    return T1.interp(s, start=T0)


def cmstraj():
    pass


# -------------------------------------------------------------------------- #

def mstraj(
        viapoints, dt, tacc, qdmax=None, tsegment=None,
        q0=None, qd0=None, qdf=None, verbose=False):
    """
    Multi-segment multi-axis trajectory

    :param viapoints: A set of viapoints, one per row
    :type viapoints: ndarray(m,n)
    :param dt: time step
    :type dt: float (seconds)
    :param tacc: acceleration time (seconds)
    :type tacc: float
    :param qdmax: maximum speed, defaults to None
    :type qdmax: array_like(n) or float, optional
    :param tsegment: maximum time of each motion segment (seconds), defaults
        to None
    :type tsegment: array_like, optional
    :param q0: initial coordinates, defaults to first row of viapoints
    :type q0: array_like(n), optional
    :param qd0: inital  velocity, defaults to zero
    :type qd0: array_like(n), optional
    :param qdf: final  velocity, defaults to zero
    :type qdf: array_like(n), optional
    :param verbose: print debug information, defaults to False
    :type verbose: bool, optional
    :return: trajectory plus extra info
    :rtype: namedtuple

    Computes a trajectory for N axes moving smoothly through a set of
    viapoints.
    The motion comprises M segments:

    - The initial coordinates are the first row of ``viapoints`` or ``q0`` if
      provided.
    - The final coordinates are the last row of ``viapoints``
    - Each segment is linear motion and polynomial blends connect the
      viapoints.
    - All joints arrive at each via point at the same time, ie. the motion is
      coordinated across axes

    The time of the segments can be specified in two different ways:

    #. In terms of segment time where ``tsegment`` is an array of segment times
       which is the number of via points minus one::

            ``traj = mstraj(viapoints, dt, tacc, tsegment=TS)``

    #. Governed by the speed of the slowest axis for the segment.  The axis
       speed is a scalar (all axes have the same speed) or an N-vector of speed
       per axis::

            traj = mstraj(viapoints, dt, tacc, qdmax=SPEED)

    The return value is a namedtuple (named ``mstraj``) with elements:

        - ``t``  the time coordinate as a numpy ndarray, shape=(K,)
        - ``q``  the axis values as a numpy ndarray, shape=(K,N)
        - ``arrive`` a list of arrival times for each segment
        - ``info`` a list of named tuples, one per segment that describe the
          slowest axis, segment time,  and time stamp
        - ``via`` the passed set of via points

    The  trajectory proper is (``traj.t``, ``traj.q``).  The trajectory is a
    matrix has one row per time step, and one column per axis.

     Notes:

     - Only one of ``qdmag`` or ``tsegment`` can be specified
     - If ``tacc`` is greater than zero then the path smoothly accelerates
       between segments using a polynomial blend.  This means that the the via
       point is not actually reached.
     - The path length K is a function of the number of via
       points and the time or velocity limits that apply.
     - Can be used to create joint space trajectories where each axis is a
       joint coordinate.
     - Can be used to create Cartesian trajectories where the "axes"
       correspond to translation and orientation in RPY or Euler angle form.
     - If ``qdmax`` is a scalar then all axes are assumed to have the same
       maximum speed.

    :seealso: `lspb`, `ctraj`, `mtraj`
    """

    if q0 is None:
        q0 = viapoints[0, :]
        viapoints = viapoints[1:, :]
    else:
        q0 = getvector(q0)
        if not viapoints.shape[1] == len(q0):
            raise ValueError('WP and Q0 must have same number of columns')

    ns, nj = viapoints.shape
    Tacc = tacc

    if qdmax is not None and tsegment is not None:
        raise ValueError('cannot specify both qdmax and tsegment')

    if qdmax is None:
        if tsegment is None:
            raise ValueError('tsegment must be given if qdmax is not')

        if not len(tsegment) == ns:
            raise ValueError(
                'Length of TSEG does not match number of viapoints')

    if tsegment is None:

        # This is unreachable, left just in case
        if qdmax is None:  # pragma nocover
            raise ValueError('qdmax must be given if tsegment is not')

        if isinstance(qdmax, (int, float)):
            # if qdmax is a scalar assume all axes have the same speed
            qdmax = np.tile(qdmax, (nj,))
        else:
            qdmax = getvector(qdmax)

            if not len(qdmax) == nj:
                raise ValueError(
                    'Length of QDMAX does not match number of axes')

    if isinstance(Tacc, (int, float)):
        Tacc = np.tile(Tacc, (ns,))
    else:
        if not len(Tacc) == ns:
            raise ValueError('Tacc is wrong size')
    if qd0 is None:
        qd0 = np.zeros((nj,))
    else:
        if not len(qd0) == len(q0):
            raise ValueError('qd0 is wrong size')
    if qdf is None:
        qdf = np.zeros((nj,))
    else:
        if not len(qdf) == len(q0):
            raise ValueError('qdf is wrong size')

    # set the initial conditions
    q_prev = q0
    qd_prev = qd0

    clock = 0     # keep track of time
    arrive = np.zeros((ns,))   # record planned time of arrival at via points
    tg = np.zeros((0, nj))
    infolist = []
    info = namedtuple('mstraj_info', 'slowest segtime clock')

    def mrange(start, stop, step):
        """
        mrange(start, stop, step) behaves like MATLAB start:step:stop
        and includes the final value unlike range() or np.arange()
        """
        ret = []
        v = start
        while v <= stop:
            ret.append(v)
            v += step
        return np.r_[ret]

    for seg in range(0, ns):
        q_next = viapoints[seg, :]    # current target

        if verbose:  # pragma nocover
            print(f"------- segment {seg}: {q_prev} --> {q_next}")

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

        dq = q_next - q_prev    # total distance to move this segment

        # probably should iterate over the next section to get qb right...
        # while 1
        #   qd_next = (qnextnext - qnext)
        #   tb = abs(qd_next - qd) ./ qddmax;
        #   qb = f(tb, max acceleration)
        #   dq = q_next - q_prev - qb
        #   tl = abs(dq) ./ qdmax;

        if qdmax is not None:
            # qdmax is specified, compute slowest axis

            # qb = taccx * qdmax / 2       # distance moved during blend
            tb = taccx

            # convert to time
            tl = abs(dq) / qdmax
            # tl = abs(dq - qb) / qdmax
            tl = np.ceil(tl / dt) * dt

            # find the total time and slowest axis
            tt = tb + tl
            slowest = np.argmax(tt)
            tseg = tt[slowest]

            # best if there is some linear motion component
            if tseg <= 2*tacc:
                tseg = 2 * tacc

        elif tsegment is not None:
            # segment time specified, use that
            tseg = tsegment[seg]
            slowest = math.nan

        infolist.append(info(slowest, tseg, clock))

        # log the planned arrival time
        arrive[seg] = clock + tseg
        if seg > 0:
            arrive[seg] += tacc2

        if verbose:   # pragma nocover
            print(
                f"seg {seg}, distance {dq}, "
                "slowest axis {slowest}, time required {tseg}")

        # create the trajectories for this segment

        # linear velocity from qprev to qnext
        qd = dq / tseg

        # add the blend polynomial
        qb = jtraj(
            q0, q_prev + tacc2 * qd, mrange(0, taccx, dt),
            qd0=qd_prev, qd1=qd).q
        if verbose:    # pragma nocover
            print(qb)
        tg = np.vstack([tg, qb[1:, :]])

        clock = clock + taccx     # update the clock

        # add the linear part, from tacc/2+dt to tseg-tacc/2
        for t in mrange(tacc2 + dt, tseg - tacc2, dt):
            s = t / tseg
            q0 = (1 - s) * q_prev + s * q_next       # linear step
            print(t, s, q0)
            tg = np.vstack([tg, q0])
            clock += dt

        q_prev = q_next    # next target becomes previous target
        qd_prev = qd

    # add the final blend
    qb = jtraj(q0, q_next, mrange(0, tacc2, dt), qd0=qd_prev, qd1=qdf).q
    tg = np.vstack([tg, qb[1:, :]])

    infolist.append(info(None, tseg, clock))

    return namedtuple(
        'mstraj', 't q arrive info via')(
            dt * np.arange(0, tg.shape[0]), tg, arrive, infolist, viapoints)
