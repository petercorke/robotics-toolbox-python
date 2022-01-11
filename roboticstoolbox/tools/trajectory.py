import numpy as np
import math
from collections import namedtuple
import matplotlib.pyplot as plt
from spatialmath.base.argcheck import (
    isvector,
    getvector,
    assertmatrix,
    getvector,
    isscalar,
)


class Trajectory:
    """
    A container class for trajectory data.
    """

    def __init__(self, name, t, s, sd=None, sdd=None, istime=False):
        """
        Construct a new trajectory instance

        :param name: name of the function that created the trajectory
        :type name: str
        :param t: independent variable, eg. time or step
        :type t: ndarray(m)
        :param s: position
        :type s: ndarray(m) or ndarray(m,n)
        :param sd: velocity
        :type sd: ndarray(m) or ndarray(m,n)
        :param sdd: acceleration
        :type sdd: ndarray(m) or ndarray(m,n)
        :param istime: ``t`` is time, otherwise step number
        :type istime: bool
        :param tblend: blend duration (``lspb`` only)
        :type istime: float

        The object has attributes:

        - ``t``  the independent variable
        - ``s``  the position
        - ``sd``  the velocity
        - ``sdd``  the acceleration

        If ``t`` is time, ie. ``istime`` is True, then the units of ``sd`` and
        ``sdd`` are :math:`s^{-1}` and :math:`s^{-2}` respectively, otherwise
        with respect to ``t``.

        .. note:: Data is stored with timesteps as rows and axes as columns.
        """
        self.name = name
        self.t = t
        self.s = s
        self.sd = sd
        self.sdd = sdd
        self.istime = istime

    def __str__(self):
        s = f"Trajectory created by {self.name}: {len(self)} time steps x {self.naxes} axes"
        return s

    def __repr__(self):
        return str(self)

    def __len__(self):
        """
        Length of trajectory

        :return: number of steps in the trajectory
        :rtype: int
        """
        return self.s.shape[0]

    @property
    def q(self):
        """
        Position trajectory

        :return: trajectory with one row per timestep, one column per axis
        :rtype: ndarray(n,m)

        .. note:: This is a synonym for ``.s``, for compatibility with other
            applications.
        """
        return self.s

    @property
    def qd(self):
        """
        Velocity trajectory

        :return: trajectory velocity with one row per timestep, one column per axis
        :rtype: ndarray(n,m)

        .. note:: This is a synonym for ``.sd``, for compatibility with other
            applications.
        """
        return self.sd

    @property
    def qdd(self):
        """
        Acceleration trajectory

        :return: trajectory acceleration with one row per timestep, one column per axis
        :rtype: ndarray(n,m)

        .. note:: This is a synonym for ``.sdd``, for compatibility with other
            applications.
        """
        return self.sdd

    # @property
    # def t(self):
    #     """
    #     Trajectory time

    #     :return: trajectory time vector
    #     :rtype: ndarray(n)

    #     .. note:: This is a synonym for ``.t``, for compatibility with other
    #         applications.
    #     """
    #     return self.x

    @property
    def naxes(self):
        """
        Number of axes in the trajectory

        :return: number of axes or dimensions
        :rtype: int
        """
        if self.s.ndim == 1:
            return 1
        else:
            return self.s.shape[1]

    def plot(self, block=False, plotargs=None, textargs=None):
        """
        Plot trajectory

        :param block: wait till plot is dismissed
        :type block: bool


        Plot the position, velocity and acceleration data.  The format of the
        plot depends on the function that created it.

        - ``tpoly`` and ``lspb`` show the individual points with markers
        - ``lspb`` color code the different motion phases
        - ``jtraj`` general m-axis trajectory, show legend

        :seealso: :func:`~tpoly`, :func:`~lspb`
        """

        plotopts = {"marker": "o", "markersize": 3}
        if plotargs is not None:
            plotopts = {**plotopts, **plotargs}
        textopts = {"fontsize": 12}
        if textargs is not None:
            textopts = {**textopts, **textargs}

        plt.figure()
        ax = plt.subplot(3, 1, 1)

        # plot position
        if self.name == "tpoly":
            ax.plot(self.t, self.s, **plotopts)

        elif self.name == "lspb":
            # accel phase
            tf = self.t[-1]
            k = self.t <= self.tblend
            ax.plot(self.t[k], self.s[k], color="red", label="acceleration", **plotopts)

            # coast phase
            k = (self.t > self.tblend) & (self.t <= (tf - self.tblend))
            ax.plot(self.t[k], self.s[k], color="green", **plotopts)
            k = np.where(k)[0][0]
            ax.plot(
                self.t[k - 1 : k + 1], self.s[k - 1 : k + 1], color="green", label="linear", **plotopts
            )

            # decel phase
            k = self.t > (tf - self.tblend)
            ax.plot(self.t[k], self.s[k], color="blue", **plotopts)
            k = np.where(k)[0][0]
            ax.plot(
                self.t[k - 1 : k + 1], self.s[k - 1 : k + 1], color="blue", label="deceleration", **plotopts
            )

            ax.grid(True)
        else:
            ax.plot(self.t, self.s, **plotopts)

        if self.s.ndim > 1:
            ax.legend([f"q{i+1}" for i in range(self.naxes)])

        ax.grid(True)
        ax.set_xlim(0, max(self.t))

        if self.istime:
            if self.name in ("traj", "mtraj", "mstraj"):
                symbol = "q"
            else:
                symbol = "s"
            ax.set_ylabel(f"${symbol}(t)$", **textopts)
        else:
            ax.set_ylabel("$s(k)$", **textopts)

        # plot velocity
        ax = plt.subplot(3, 1, 2)
        ax.plot(self.t, self.sd, "-o", **plotopts)
        ax.grid(True)
        ax.set_xlim(0, max(self.t))

        if self.istime:
            ax.set_ylabel(f"$\dot{{{symbol}}}(t)$", **textopts)
        else:
            ax.set_ylabel("$ds/dk$", **textopts)

        # plot acceleration
        ax = plt.subplot(3, 1, 3)
        ax.plot(self.t, self.sdd, "-o", **plotopts)
        ax.grid(True)
        ax.set_xlim(0, max(self.t))

        if self.istime:
            ax.set_ylabel(f"$\ddot{{{symbol}}}(t)$", **textopts)
            ax.set_xlabel("t (seconds)")
        else:
            ax.set_ylabel("$d^2s/dk^2$", **textopts)
            ax.set_xlabel("k (step)")

        plt.show(block=block)

    def qplot(self, **kwargs):
        """
        Plot multi-axis trajectory

        :param **kwargs: optional arguments passed to ``qplot``

        Plots a multi-axis trajectory, held within the object, as position against time.

        :seealso: :func:`qplot`
        """
        qplot(self.t, self.q, **kwargs)


def tpoly(q0, qf, t, qd0=0, qdf=0):
    """
    Generate scalar polynomial trajectory

    :param q0: initial value
    :type q0: float
    :param qf: final value
    :type qf: float
    :param t: time
    :type t: int or array_like(m)
    :param qd0: initial velocity, optional
    :type q0: float
    :param qdf: final velocity, optional
    :type q0: float
    :return: trajectory
    :rtype: Trajectory instance

    - ``tg = tpoly(q0, q1, m)`` is a scalar trajectory (Mx1) that varies
      smoothly from ``q0`` to ``qf`` using a quintic polynomial.  The initial
      and final velocity and acceleration are zero. ``m`` is an integer scalar,
      indicating the total number of timesteps and

            - Velocity is in units of distance per trajectory step, not per
              second.
            - Acceleration is in units of distance per trajectory step squared,
              *not* per second squared.

    - ``tg = tpoly(q0, q1, t)`` as above but ``t`` is a uniformly-spaced time
      vector

            - Velocity is in units of distance per second.
            - Acceleration is in units of distance per second squared.

    The return value is an object that contains position, velocity and
    acceleration data.

    .. note:: The time vector T is assumed to be monotonically increasing, and
        time scaling is based on the first and last element.

    References:

    - Robotics, Vision & Control, Chap 3,
      P. Corke, Springer 2011.

    :seealso: :func:`lspb`, :func:`mtraj`.
    """
    if isinstance(t, int):
        t = np.arange(0, t)
        istime = False
    elif isvector(t):
        t = getvector(t)
        istime = True
    else:
        raise TypeError("bad argument for time, must be int or vector")
    tf = max(t)

    polyfunc = tpoly_func(q0, qf, tf, qd0, qdf)

    # evaluate the polynomials
    traj = polyfunc(t)
    p = traj[0]
    pd = traj[1]
    pdd = traj[2]

    return Trajectory("tpoly", t, p, pd, pdd, istime)


def tpoly_func(q0, qf, T, qd0=0, qdf=0):

    # solve for the polynomial coefficients using least squares
    X = [
        [0, 0, 0, 0, 0, 1],
        [T ** 5, T ** 4, T ** 3, T ** 2, T, 1],
        [0, 0, 0, 0, 1, 0],
        [5 * T ** 4, 4 * T ** 3, 3 * T ** 2, 2 * T, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [20 * T ** 3, 12 * T ** 2, 6 * T, 2, 0, 0],
    ]
    coeffs, resid, rank, s = np.linalg.lstsq(
        X, np.r_[q0, qf, qd0, qdf, 0, 0], rcond=None
    )

    # coefficients of derivatives
    coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
    coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

    return lambda x: (
        np.polyval(coeffs, x),
        np.polyval(coeffs_d, x),
        np.polyval(coeffs_dd, x),
    )


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
    :rtype: Trajectory instance

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

    The return value is an object that contains position, velocity and
    acceleration data.

    .. note::

        - For some values of V no solution is possible and an error is flagged.
        - The time vector, if given, is assumed to be monotonically increasing,
          and time scaling is based on the first and last element.
        - ``tg`` has an extra attribute ``xblend`` which is the blend duration.

    :References:

        - Robotics, Vision & Control, Chap 3,
        P. Corke, Springer 2011.

    :seealso: :func:`tpoly`, :func:`mtraj`.
    """

    if isinstance(t, int):
        t = np.arange(0, t)
        istime = False
    elif isvector(t):
        t = getvector(t)
        istime = True
    else:
        raise TypeError("bad argument for time, must be int or vector")

    tf = max(t)

    lspbfunc = lspb_func(q0, qf, tf, V)

    # evaluate the polynomials
    traj = lspbfunc(t)
    p = traj[0]
    pd = traj[1]
    pdd = traj[2]

    traj = Trajectory("lspb", t, p, pd, pdd, istime)
    traj.tblend = lspbfunc.tb
    return traj


def lspb_func(q0, qf, tf, V=None):

    if V is None:
        # if velocity not specified, compute it
        V = (qf - q0) / tf * 1.5
    else:
        V = abs(V) * np.sign(qf - q0)
        if abs(V) < (abs(qf - q0) / tf):
            raise ValueError("V too small")
        elif abs(V) > (2 * abs(qf - q0) / tf):
            raise ValueError("V too big")

    if q0 == qf:
        # Commented these because they arent used anywhere
        # s = np.ones((len(t), len(t))) @ q0
        # sd = np.zeros((len(t), len(t)))
        # sdd = np.zeros((len(t), len(t)))
        return

    tb = (q0 - qf + V * tf) / V
    a = V / tb

    def lspbfunc(t):

        p = []
        pd = []
        pdd = []

        if isscalar(t):
            t = [t]
        for tk in t:
            if tk <= tb:
                # initial blend
                pk = q0 + a / 2 * tk ** 2
                pdk = a * tk
                pddk = a
            elif tk <= (tf - tb):
                # linear motion
                pk = (qf + q0 - V * tf) / 2 + V * tk
                pdk = V
                pddk = 0
            else:
                # final blend
                pk = qf - a / 2 * tf ** 2 + a * tf * tk - a / 2 * tk ** 2
                pdk = a * tf - a * tk
                pddk = -a
            p.append(pk)
            pd.append(pdk)
            pdd.append(pddk)
        return (np.array(p), np.array(pd), np.array(pdd))

    # return the function, but add some computed parameters as attributes
    # as a way of returning extra values without a tuple return
    func = lspbfunc
    func.tb = tb
    func.V = V

    return func


# -------------------------------------------------------------------------- #


def jtraj(q0, qf, t, qd0=None, qd1=None):
    """
    Compute a joint-space trajectory

    :param q0: initial joint coordinate
    :type q0: array_like(n)
    :param qf: final joint coordinate
    :type qf: array_like(n)
    :param t: time vector or number of steps
    :type t: array_like or int
    :param qd0: initial velocity, defaults to zero
    :type qd0: array_like(n), optional
    :param qd1: final velocity, defaults to zero
    :type qd1: array_like(n), optional
    :return: trajectory
    :rtype: Trajectory instance

    - ``tg = jtraj(q0, qf, N)`` is a joint space trajectory where the joint
      coordinates vary from ``q0`` (M) to ``qf`` (M).  A quintic (5th order)
      polynomial is used with default zero boundary conditions for velocity and
      acceleration.  Time is assumed to vary from 0 to 1 in ``N`` steps.

    - ``tg = jtraj(q0, qf, t)`` as above but ``t`` is a uniformly-spaced time
      vector

    The return value is an object that contains position, velocity and
    acceleration data.

    Notes:

    - The time vector, if given, scales the velocity and acceleration outputs
      assuming that the time vector starts at zero and increases
      linearly.

    :seealso: :func:`ctraj`, :func:`qplot`, :func:`~SerialLink.jtraj`
    """
    # print(f"  --- jtraj: {q0} --> {q1} in {tv}")
    if isinstance(t, int):
        tscal = 1.0
        ts = np.linspace(0, 1, t)  # normalized time from 0 -> 1
    else:
        tscal = max(t)
        ts = t.flatten() / tscal

    q0 = getvector(q0)
    qf = getvector(qf)

    if not len(q0) == len(qf):
        raise ValueError("q0 and q1 must be same size")

    if qd0 is None:
        qd0 = np.zeros(q0.shape)
    else:
        qd0 = getvector(qd0)
        if not len(qd0) == len(q0):
            raise ValueError("qd0 has wrong size")
    if qd1 is None:
        qd1 = np.zeros(q0.shape)
    else:
        qd1 = getvector(qd1)
        if not len(qd1) == len(q0):
            raise ValueError("qd1 has wrong size")

    # compute the polynomial coefficients
    A = 6 * (qf - q0) - 3 * (qd1 + qd0) * tscal
    B = -15 * (qf - q0) + (8 * qd0 + 7 * qd1) * tscal
    C = 10 * (qf - q0) - (6 * qd0 + 4 * qd1) * tscal
    E = qd0 * tscal  # as the t vector has been normalized
    F = q0

    # n = len(q0)

    tt = np.array([ts ** 5, ts ** 4, ts ** 3, ts ** 2, ts, np.ones(ts.shape)]).T
    coeffs = np.array([A, B, C, np.zeros(A.shape), E, F])  # 6xN

    qt = tt @ coeffs

    # compute  velocity
    coeffs = np.array([np.zeros(A.shape), 5 * A, 4 * B, 3 * C, np.zeros(A.shape), E])
    qdt = tt @ coeffs / tscal

    # compute  acceleration
    coeffs = np.array(
        [np.zeros(A.shape), np.zeros(A.shape), 20 * A, 12 * B, 6 * C, np.zeros(A.shape)]
    )
    qddt = tt @ coeffs / tscal ** 2

    return Trajectory("jtraj", t, qt, qdt, qddt, istime=True)


def mtraj(tfunc, q0, qf, t):
    """
    Multi-axis trajectory

    :param tfunc: a 1D trajectory function, eg. ``tpoly`` or ``lspb``
    :type tfunc: callable
    :param q0: initial configuration
    :type q0: ndarray(m)
    :param qf: final configuration
    :type qf: ndarray(m)
    :param t: time vector or number of steps
    :type t: array_like or int
    :raises TypeError: ``tfunc`` is not callable
    :raises ValueError: length of ``q0`` and ``qf`` are different
    :return: trajectory
    :rtype: Trajectory instance

    - ``tg = mtraj(func, q0, qf, n)`` is a multi-axis trajectory varying
      from configuration ``q0`` (M) to ``qf`` (M) according to the scalar trajectory
      function ``tfunc`` in ``n``` steps.

    - ``tg = mtraj(func, q0, qf, t)`` as above but ``t`` is a uniformly-spaced time
      vector

    The scalar trajectory function is applied to each axis::

            tg = tfunc(s0, sF, n)

    and possible values of TFUNC include ``lspb`` for a trapezoidal trajectory, or
    ``tpoly`` for a polynomial trajectory.

    The return value is an object that contains position, velocity and
    acceleration data.

    .. note:: The time vector, if given, is assumed to be monotonically increasing, and
        time scaling is based on the first and last element.

    :seealso: :func:`tpoly`, :func:`lspb`
    """

    if not callable(tfunc):
        raise TypeError("first argument must be a function reference")

    q0 = getvector(q0)
    qf = getvector(qf)
    if len(q0) != len(qf):
        raise ValueError("must be same number of elements in q0 and qf")

    traj = []
    for i in range(len(q0)):
        # for each axis
        traj.append(tfunc(q0[i], qf[i], t))

    x = traj[0].t
    y = np.array([tg.s for tg in traj]).T
    yd = np.array([tg.sd for tg in traj]).T
    ydd = np.array([tg.sdd for tg in traj]).T

    istime = traj[0].istime

    return Trajectory("mtraj", x, y, yd, ydd, istime)


def qplot(
    x,
    y=None,
    wrist=False,
    unwrap=False,
    block=False,
    labels=None,
    loc=None,
    grid=True,
    stack=False,
    **kwargs,
):
    """
    Plot trajectory data

    :param q: trajectory, one row per timestep
    :type q: ndarray(m,n)
    :param t: time vector, optional
    :type t: numpy ndarray, shape=(M,)
    :param wrist: distinguish arm and wrist joints with line styles
    :type wrist: bool
    :param unwrap: unwrap joint angles so that they smoothly increase or
        decrease when they pass through :math:`\pm \pi`
    :type unwrap: bool
    :param block: block until the plot is closed
    :type block: bool
    :param labels: legend labels
    :type labels: list of str, or single string with space separated labels
    :param kwargs: options passed to pyplot.plot
    :param loc: legend location as per pyplot.legend
    :type loc: str

    This is a convenience function to plot trajectories, where each row represents one time step.

    - ``qplot(q)`` plots the joint angles versus row number.  If N==6 a
      conventional 6-axis robot is assumed, and the first three joints are
      shown as solid lines, the last three joints (wrist) are shown as dashed
      lines. A legend is also displayed.

    - ``qplot(t, q)`` as above but displays the joint angle trajectory versus
      time given the time vector T (Mx1).

    Example::

        >>> qplot(q, x, labels='x y z')

    :seealso: :func:`jtraj`, :func:`numpy.unwrap`
    """
    if y is None:
        q = x
        t = np.arange(0, q.shape[0])
    else:
        t = x
        q = y

    if t.ndim != 1 or q.shape[0] != t.shape[0]:
        raise ValueError("dimensions of arguments are not consistent")

    if unwrap:
        q = np.unwrap(q, axis=0)

    n = q.shape[1]

    if labels is None:
        labels = [f"q{i}" for i in range(n)]
    elif isinstance(labels, str):
        labels = labels.split(" ")
    elif not isinstance(labels, (tuple, list)):
        raise TypeError("wrong type for labels")

    fig, ax = plt.subplots()

    if stack:
        for i in range(n):
            ax = plt.subplot(n, 1, i + 1)

            plt.plot(t, q[:, i], **kwargs)

            plt.grid(grid)
            ax.set_ylabel(labels[i])
            ax.set_xlim(t[0], t[-1])

        ax.set_xlabel("Time (s)")

    else:
        if n == 6 and wrist:
            plt.plot(t, q[:, 0:3], **kwargs)
            plt.plot(t, q[:, 3:6], "--", **kwargs)
        else:
            plt.plot(t, q, **kwargs)

        ax.legend(labels, loc=loc)

        plt.grid(grid)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint coordinates (rad,m)")
        ax.set_xlim(t[0], t[-1])

    plt.show(block=block)

    return fig.get_axes()


# -------------------------------------------------------------------------- #


def ctraj(T0, T1, t=None, s=None):
    """
    Cartesian trajectory between two poses

    :param T0: initial pose
    :type T0: SE3
    :param T1: final pose
    :type T1: SE3
    :param t: number of samples or time vector
    :type t: int or ndarray(n)
    :param s: array of distance along the path, in the interval [0, 1]
    :type s: ndarray(s)
    :return T0: smooth path from ``T0`` to ``T1``
    :rtype: SE3

    ``ctraj(T0, T1, n)`` is a Cartesian trajectory from SE3 pose ``T0`` to
    ``T1`` with ``n`` points that follow a trapezoidal velocity profile along
    the path. The Cartesian trajectory is an SE3 instance containing ``n``
    values.

    ``ctraj(T0, T1, t)`` as above but the trajectory is sampled at
    the  points in the array ``t``.

    ``ctraj(T0, T1, s=s)`` as above but the elements of ``s`` specify the
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

    if isinstance(t, int):
        s = lspb(0, 1, t).s
    elif isvector(t):
        t = getvector(t)
        s = lspb(0, 1, t / np.max(t)).s
    elif isvector(s):
        s = getvector(s)
    else:
        raise TypeError("bad argument for time, must be int or vector")

    return T0.interp(T1, s)


def cmstraj():
    pass


# -------------------------------------------------------------------------- #


def mstraj(
    viapoints,
    dt,
    tacc,
    qdmax=None,
    tsegment=None,
    q0=None,
    qd0=None,
    qdf=None,
    verbose=False,
):
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
    :return: trajectory
    :rtype: Trajectory instance

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
    - ``tg`` has extra attributes ``arrive``, ``info`` and ``via``


    :seealso: `lspb`, `ctraj`, `mtraj`
    """

    if q0 is None:
        q0 = viapoints[0, :]
        viapoints = viapoints[1:, :]
    else:
        q0 = getvector(q0)
        if not viapoints.shape[1] == len(q0):
            raise ValueError("WP and Q0 must have same number of columns")

    ns, nj = viapoints.shape
    Tacc = tacc

    if qdmax is not None and tsegment is not None:
        raise ValueError("cannot specify both qdmax and tsegment")

    if qdmax is None:
        if tsegment is None:
            raise ValueError("tsegment must be given if qdmax is not")

        if not len(tsegment) == ns:
            raise ValueError("Length of TSEG does not match number of viapoints")

    if tsegment is None:

        # This is unreachable, left just in case
        if qdmax is None:  # pragma nocover
            raise ValueError("qdmax must be given if tsegment is not")

        if isinstance(qdmax, (int, float)):
            # if qdmax is a scalar assume all axes have the same speed
            qdmax = np.tile(qdmax, (nj,))
        else:
            qdmax = getvector(qdmax)

            if not len(qdmax) == nj:
                raise ValueError("Length of QDMAX does not match number of axes")

    if isinstance(Tacc, (int, float)):
        Tacc = np.tile(Tacc, (ns,))
    else:
        if not len(Tacc) == ns:
            raise ValueError("Tacc is wrong size")
    if qd0 is None:
        qd0 = np.zeros((nj,))
    else:
        if not len(qd0) == len(q0):
            raise ValueError("qd0 is wrong size")
    if qdf is None:
        qdf = np.zeros((nj,))
    else:
        if not len(qdf) == len(q0):
            raise ValueError("qdf is wrong size")

    # set the initial conditions
    q_prev = q0
    qd_prev = qd0

    clock = 0  # keep track of time
    arrive = np.zeros((ns,))  # record planned time of arrival at via points
    tg = np.zeros((0, nj))
    infolist = []
    info = namedtuple("mstraj_info", "slowest segtime clock")

    def mrange(start, stop, step):
        """
        mrange(start, stop, step) behaves like MATLAB start:step:stop
        and includes the final value unlike range() or np.arange()
        """
        # ret = []
        istart = round(start / step)
        istop = round(stop / step)
        return np.arange(istart, istop + 1) * step

    for seg in range(0, ns):
        q_next = viapoints[seg, :]  # current target

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

        dq = q_next - q_prev  # total distance to move this segment

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
            if tseg <= 2 * tacc:
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

        if verbose:  # pragma nocover
            print(
                f"seg {seg}, distance {dq}, "
                "slowest axis {slowest}, time required {tseg}"
            )

        # create the trajectories for this segment

        # linear velocity from qprev to qnext
        qd = dq / tseg

        # add the blend polynomial
        qb = jtraj(q0, q_prev + tacc2 * qd, mrange(0, taccx, dt), qd0=qd_prev, qd1=qd).s
        if verbose:  # pragma nocover
            print(qb)
        tg = np.vstack([tg, qb[1:, :]])

        clock = clock + taccx  # update the clock

        # add the linear part, from tacc/2+dt to tseg-tacc/2
        for t in mrange(tacc2 + dt, tseg - tacc2, dt):
            s = t / tseg
            q0 = (1 - s) * q_prev + s * q_next  # linear step
            if verbose:  # pragma nocover
                print(t, s, q0)
            tg = np.vstack([tg, q0])
            clock += dt

        q_prev = q_next  # next target becomes previous target
        qd_prev = qd

    # add the final blend
    qb = jtraj(q0, q_next, mrange(0, tacc2, dt), qd0=qd_prev, qd1=qdf).s
    tg = np.vstack([tg, qb[1:, :]])

    infolist.append(info(None, tseg, clock))

    traj = Trajectory("mstraj", dt * np.arange(0, tg.shape[0]), tg)
    traj.arrive = arrive
    traj.info = infolist
    traj.via = viapoints

    return traj
    # return namedtuple(
    #     'mstraj', 't q arrive info via')(
    #         dt * np.arange(0, tg.shape[0]), tg, arrive, infolist, viapoints)


if __name__ == "__main__":

    t = tpoly(0, 1, 50)
    t.plot()

    t = tpoly(0, 1, np.linspace(0, 1, 50))
    t.plot()

    t = lspb(0, 1, 50)
    t.plot()
    t = lspb(0, 1, np.linspace(0, 1, 50))
    t.plot(block=True)
