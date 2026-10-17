import math
import warnings
from collections import namedtuple
from collections.abc import Callable
from typing import Any, Literal, overload

import matplotlib.pyplot as plt
from roboticstoolbox.tools.plot import apply_block
import numpy as np
from spatialmath import SE3
from spatialmath.base.argcheck import (
    # assertmatrix,
    getvector,
    isscalar,
    isvector,
)

from roboticstoolbox.tools.types import ArrayLike, NDArray


class Trajectory:
    """
    A container class for trajectory data.
    """

    #: blend duration, set by :func:`trapezoidal` only
    tblend: float
    #: arrival time at each via point, set by :func:`mstraj` only
    arrive: NDArray
    #: one named tuple per segment (slowest axis, segment time, clock), set by
    #: :func:`mstraj` only
    info: list[Any]
    #: the via points, set by :func:`mstraj` only
    via: NDArray

    def __init__(
        self,
        name: str,
        t: NDArray,
        s: NDArray,
        sd: NDArray | None = None,
        sdd: NDArray | None = None,
        istime: bool = False,
    ) -> None:
        """
        Construct a new trajectory instance

        :param name: name of the function that created the trajectory
        :param t: independent variable, eg. time or step
        :type t: ndarray(m)
        :param s: position
        :type s: ndarray(m) or ndarray(m,n)
        :param sd: velocity
        :type sd: ndarray(m) or ndarray(m,n)
        :param sdd: acceleration
        :type sdd: ndarray(m) or ndarray(m,n)
        :param istime: ``t`` is time, otherwise step number

        The object has attributes:

        - ``t``  the independent variable
        - ``s``  the position
        - ``sd``  the velocity
        - ``sdd``  the acceleration

        and, set by the function that created it, ``tblend`` for
        :func:`trapezoidal` and ``arrive``, ``info`` and ``via`` for
        :func:`mstraj`.

        If ``t`` is time, ie. ``istime`` is True, then the units of ``sd`` and
        ``sdd`` are :math:`s^{-1}` and :math:`s^{-2}` respectively, otherwise
        with respect to ``t``.

        .. note:: Data is stored with timesteps as rows and axes as columns.

        :References:

            - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 3.
        """
        self.name = name
        self.t = t
        self.s = s
        self.sd = sd
        self.sdd = sdd
        self.istime = istime

    def __str__(self) -> str:
        s = (
            f"Trajectory created by {self.name}: {len(self)} time steps x"
            f" {self.naxes} axes"
        )
        return s

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        """
        Length of trajectory

        :return: number of steps in the trajectory
        :rtype: int
        """
        return self.s.shape[0]

    @property
    def q(self) -> NDArray:
        """
        Position trajectory

        :return: trajectory with one row per timestep, one column per axis
        :rtype: ndarray(n,m)

        .. note:: This is a synonym for ``.s``, for compatibility with other
            applications.
        """
        return self.s

    @property
    def qd(self) -> NDArray:
        """
        Velocity trajectory

        :return: trajectory velocity with one row per timestep, one column per axis
        :rtype: ndarray(n,m)
        :raises ValueError: the trajectory has no velocity data

        .. note:: This is a synonym for ``.sd``, for compatibility with other
            applications.
        """
        if self.sd is None:
            raise ValueError("trajectory has no velocity data")
        return self.sd

    @property
    def qdd(self) -> NDArray:
        """
        Acceleration trajectory

        :return: trajectory acceleration with one row per timestep, one column per axis
        :rtype: ndarray(n,m)
        :raises ValueError: the trajectory has no acceleration data

        .. note:: This is a synonym for ``.sdd``, for compatibility with other
            applications.
        """
        if self.sdd is None:
            raise ValueError("trajectory has no acceleration data")
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
    def naxes(self) -> int:
        """
        Number of axes in the trajectory

        :return: number of axes or dimensions
        :rtype: int
        """
        if self.s.ndim == 1:
            return 1
        else:
            return self.s.shape[1]

    def plot(
        self,
        block: bool = False,
        plotargs: dict[str, Any] | None = None,
        textargs: dict[str, Any] | None = None,
    ) -> None:
        """
        Plot trajectory

        :param block: ``None`` leaves showing the figure to the environment (no
            ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
            it and waits until it is closed; ignored by notebook backends,
            defaults to False
        :param plotargs: Matplotlib options for the plotted lines
        :param textargs: Matplotlib options for the axis labels

        Plot the position, velocity and acceleration data.  The format of the
        plot depends on the function that created it.

        - ``quintic`` and ``trapezoidal`` show the individual points with markers
        - ``trapezoidal`` color code the different motion phases
        - ``jtraj`` general m-axis trajectory, show legend

        :seealso: :func:`~quintic`, :func:`~trapezoidal`

        .. versionchanged:: 1.5.0
            ``block=None`` does not call ``plt.show``, as documented. It used to call
            ``plt.show(block=None)``, which waits in a script. The default, ``False``, is unchanged.
        """

        plotopts: dict[str, Any] = {"marker": "o", "markersize": 3}
        if plotargs is not None:
            plotopts = {**plotopts, **plotargs}
        textopts: dict[str, Any] = {"fontsize": 12}
        if textargs is not None:
            textopts = {**textopts, **textargs}

        nplots = 3
        if self.name == "mstraj":
            nplots = 1

        plt.figure()
        ax = plt.subplot(nplots, 1, 1)

        # plot position
        if self.name in "quintic":
            ax.plot(self.t, self.s, **plotopts)

        elif self.name == "trapezoidal":
            elapsed = self.t - self.t[0]
            tf = elapsed[-1]
            # accel phase
            k = elapsed <= self.tblend
            ax.plot(self.t[k], self.s[k], color="red", label="acceleration", **plotopts)

            # coast phase
            k = (elapsed > self.tblend) & (elapsed <= (tf - self.tblend))
            ax.plot(self.t[k], self.s[k], color="green", **plotopts)
            k = np.where(k)[0][0]
            ax.plot(
                self.t[k - 1 : k + 1],
                self.s[k - 1 : k + 1],
                color="green",
                label="linear",
                **plotopts,
            )

            # decel phase
            k = elapsed > (tf - self.tblend)
            ax.plot(self.t[k], self.s[k], color="blue", **plotopts)
            k = np.where(k)[0][0]
            ax.plot(
                self.t[k - 1 : k + 1],
                self.s[k - 1 : k + 1],
                color="blue",
                label="deceleration",
                **plotopts,
            )

            ax.grid(True)
        else:
            ax.plot(self.t, self.s, **plotopts)

        if self.s.ndim > 1:
            ax.legend([f"q{i + 1}" for i in range(self.naxes)])

        ax.grid(True)
        ax.set_xlim(self.t[0], self.t[-1])

        if self.istime:
            ax.set_ylabel("$q(t)$", **textopts)
        else:
            ax.set_ylabel("$q(k)$", **textopts)

        if nplots > 1:
            # plot velocity
            ax = plt.subplot(3, 1, 2)
            ax.plot(self.t, self.sd, **plotopts)
            ax.grid(True)
            ax.set_xlim(self.t[0], self.t[-1])

            if self.istime:
                ax.set_ylabel(r"$\dot{{q}}(t)$", **textopts)
            else:
                ax.set_ylabel("$dq/dk$", **textopts)

            # plot acceleration
            ax = plt.subplot(3, 1, 3)
            ax.plot(self.t, self.sdd, **plotopts)
            ax.grid(True)
            ax.set_xlim(self.t[0], self.t[-1])

            if self.istime:
                ax.set_ylabel(rf"$\ddot{{q}}(t)$", **textopts)
                ax.set_xlabel("t (seconds)")
            else:
                ax.set_ylabel("$d^2q/dk^2$", **textopts)
                ax.set_xlabel("k (step)")

        apply_block(block)

    def qplot(self, **kwargs: Any) -> None:
        """
        Plot multi-axis trajectory

        :param \*\*kwargs: optional arguments passed to :func:`~roboticstoolbox.tools.plot.xplot`

        Plots a multi-axis trajectory, held within the object, as position against time.

        :seealso: :func:`~roboticstoolbox.tools.plot.xplot`
        """
        # deferred, the plot module is imported after this one by the package
        from roboticstoolbox.tools.plot import xplot

        xplot(self.t, self.q, **kwargs)


# -------------------------------------------------------------------------- #


def quintic(
    q0: float, qf: float, t: int | ArrayLike, qd0: float = 0, qdf: float = 0
) -> Trajectory:
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
    :rtype: :class:`Trajectory` instance

    - ``tg = quintic(q0, q1, m)`` is a scalar trajectory (Mx1) that varies
      smoothly from ``q0`` to ``qf`` using a quintic polynomial.  The initial
      and final velocity and acceleration are zero. ``m`` is an integer scalar,
      indicating the total number of timesteps and

            - Velocity is in units of distance per trajectory step, not per
              second.
            - Acceleration is in units of distance per trajectory step squared,
              *not* per second squared.

    - ``tg = quintic(q0, q1, t)`` as above but ``t`` is a uniformly-spaced time
      vector

            - Velocity is in units of distance per second.
            - Acceleration is in units of distance per second squared.

    The return value is an object that contains position, velocity and
    acceleration data.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import quintic
        >>> tg = quintic(1, 2, 10)
        >>> tg
        >>> len(tg)
        >>> tg.q
        >>> tg.plot()

    .. plot::

        from roboticstoolbox import quintic
        tg = quintic(1, 2, 10)
        tg.plot()


    .. note:: The time vector T is assumed to be monotonically increasing, and
        time scaling is based on the first and last element. The returned
        trajectory retains the supplied time vector.

    .. versionchanged:: 1.5.0
        Accurate for long and short durations, see :func:`quintic_func`.  A time
        vector, or an integer, giving fewer than two samples raises
        ``ValueError``; it previously returned the mean of ``q0`` and ``qf``.

    :References:

        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 3.


    :seealso: :func:`trapezoidal`, :func:`mtraj`.
    """
    tv: NDArray
    if isinstance(t, int):
        tv = np.arange(0, t)
        istime = False
    elif isvector(t):
        tv = getvector(t)
        istime = True
    else:
        raise TypeError("bad argument for time, must be int or vector")
    if len(tv) < 2:
        raise ValueError("a trajectory needs at least two time samples")
    tf = tv[-1] - tv[0]

    polyfunc = quintic_func(q0, qf, tf, qd0, qdf)

    # evaluate the polynomials
    traj = polyfunc(tv - tv[0])
    p = traj[0]
    pd = traj[1]
    pdd = traj[2]

    return Trajectory("quintic", tv, p, pd, pdd, istime)


def quintic_func(
    q0: float,
    qf: float,
    T: float,
    qd0: float = 0,
    qdf: float = 0,
    method: Literal["lstsq", "hermite"] = "lstsq",
) -> Callable[[ArrayLike], tuple[NDArray, NDArray, NDArray]]:
    r"""
    Quintic scalar polynomial as a function

    :param q0: initial value
    :param qf: final value
    :param T: trajectory duration, a finite non-zero value
    :param qd0: initial velocity, defaults to 0
    :param qdf: final velocity, defaults to 0
    :param method: how the polynomial is found, ``"lstsq"`` (default) solves the
        linear system, ``"hermite"`` uses the closed-form Hermite basis
    :returns: polynomial function :math:`f: t \mapsto (q(t), \dot{q}(t), \ddot{q}(t))`
    :raises ValueError: ``T`` is zero or not finite, ``method`` is unknown, or an
        endpoint value is not a scalar

    Returns a function which computes the specific quintic polynomial, and its
    derivatives, as described by the parameters.  Velocity and acceleration are
    with respect to the units of ``T``, and the acceleration is zero at both ends.

    The polynomial :math:`q(t) = a_5 t^5 + a_4 t^4 + a_3 t^3 + a_2 t^2 + a_1 t + a_0`
    is found from the position, velocity and acceleration at the start and end

    .. math::

        \begin{pmatrix}
            0 & 0 & 0 & 0 & 0 & 1 \\
            T^5 & T^4 & T^3 & T^2 & T & 1 \\
            0 & 0 & 0 & 0 & 1 & 0 \\
            5T^4 & 4T^3 & 3T^2 & 2T & 1 & 0 \\
            0 & 0 & 0 & 2 & 0 & 0 \\
            20T^3 & 12T^2 & 6T & 2 & 0 & 0
        \end{pmatrix}
        \begin{pmatrix} a_5 \\ a_4 \\ a_3 \\ a_2 \\ a_1 \\ a_0 \end{pmatrix}
        =
        \begin{pmatrix} q_0 \\ q_f \\ \dot{q}_0 \\ \dot{q}_f \\ 0 \\ 0 \end{pmatrix}

    This matrix is badly scaled, since its entries range from 1 to :math:`T^5`.
    Its condition number is about 10\ :sup:`3` for :math:`T=1` but 10\ :sup:`15`
    for :math:`T=1000` and 10\ :sup:`16` for :math:`T=0.001`, at which point a
    least-squares solver silently discards a boundary condition: the initial
    position of ``quintic(1, 2, 1000)`` came out near zero rather than one.

    For ``method="lstsq"`` the system is therefore solved in normalized time
    :math:`\tau = t/T`, for which the matrix above is used with :math:`T=1`.  It
    is the same for every trajectory, with a condition number of 762, and the
    velocity boundary values are multiplied by :math:`T`.  The derivatives are
    divided by :math:`T` and :math:`T^2` to return to the units of ``t``.

    For ``method="hermite"`` the solution of the same system is written down
    directly, as a sum of Hermite basis polynomials in :math:`\tau`.  It is
    equivalent, and is useful as an independent check.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import quintic_func
        >>> f = quintic_func(1, 2, 1000)
        >>> f(0)
        >>> f(1000)
        >>> f(500)

    .. note:: The Hermite form is due to Afloat16, who found the loss of
        accuracy for long and short durations in PR #736.

    .. versionchanged:: 1.5.0
        The system is solved in normalized time.  Previously the result was
        silently wrong for long and short durations, for example the initial
        value of ``quintic_func(1, 2, 1000)`` was close to zero.  Added
        ``method``.  A zero or non-finite ``T`` raises ``ValueError``, where it
        previously gave a degenerate result.

    :seealso: :func:`quintic` :func:`trapezoidal_func`
    """
    if method not in ("lstsq", "hermite"):
        raise ValueError(f"unknown method {method!r}, must be 'lstsq' or 'hermite'")
    if not np.isfinite(T) or T == 0:
        raise ValueError("trajectory duration must be finite and non-zero")
    boundary = np.r_[q0, qf, qd0, qdf]
    if boundary.shape != (4,):
        raise ValueError("q0, qf, qd0 and qdf must be scalars")

    if method == "hermite":
        return _quintic_hermite_func(q0, qf, T, qd0, qdf)

    # solve for the polynomial coefficients in normalized time, tau = t / T,
    # where the matrix is that of the docstring with T = 1
    # fmt: off
    X = np.array([
        [ 0.0,   0.0,   0.0,  0.0,  0.0,  1.0],
        [ 1.0,   1.0,   1.0,  1.0,  1.0,  1.0],
        [ 0.0,   0.0,   0.0,  0.0,  1.0,  0.0],
        [ 5.0,   4.0,   3.0,  2.0,  1.0,  0.0],
        [ 0.0,   0.0,   0.0,  2.0,  0.0,  0.0],
        [20.0,  12.0,   6.0,  2.0,  0.0,  0.0],
    ])
    # fmt: on
    # velocities with respect to tau are T times those with respect to t
    coeffs = np.linalg.solve(X, np.r_[q0, qf, qd0 * T, qdf * T, 0, 0])

    # coefficients of derivatives
    coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
    coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

    def polynomial(x: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        tau = np.asarray(x) / T
        return (
            np.polyval(coeffs, tau),
            np.polyval(coeffs_d, tau) / T,
            np.polyval(coeffs_dd, tau) / T**2,
        )

    return polynomial


def _quintic_hermite_func(
    q0: float, qf: float, T: float, qd0: float, qdf: float
) -> Callable[[ArrayLike], tuple[NDArray, NDArray, NDArray]]:
    """
    Quintic scalar polynomial using a Hermite basis

    The polynomial is constructed and evaluated in normalized time to avoid
    ill-conditioning for short or long durations.  The derivatives are factored
    so that zero endpoint acceleration is not lost through polynomial
    cancellation.  Contributed by Afloat16 in PR #736.
    """
    # velocities with respect to normalized time
    v0 = qd0 * T
    vf = qdf * T
    delta = qf - q0

    def polynomial(x: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
        tau = np.asarray(x, dtype=float) / T
        remaining = 1 - tau
        blend = tau**3 * (10 + tau * (-15 + 6 * tau))
        return (
            q0 * (1 - blend)
            + qf * blend
            + v0 * tau * remaining**3 * (1 + 3 * tau)
            - vf * tau**3 * remaining * (4 - 3 * tau),
            (
                30 * delta * tau**2 * remaining**2
                + v0 * remaining**2 * (1 + 2 * tau - 15 * tau**2)
                + vf * tau**2 * (1 + 2 * remaining - 15 * remaining**2)
            )
            / T,
            12
            * tau
            * remaining
            * (5 * delta * (1 - 2 * tau) - v0 * (3 - 5 * tau) - vf * (2 - 5 * tau))
            / T
            / T,
        )

    return polynomial


# -------------------------------------------------------------------------- #


def lspb(*args: Any, **kwargs: Any) -> Trajectory:
    """
    .. warning:: Deprecated, use ``trapezoidal`` instead.
    """
    warnings.warn("lsp is deprecated, use trapezoidal", FutureWarning)
    return trapezoidal(*args, **kwargs)


# -------------------------------------------------------------------------- #


def trapezoidal(
    q0: float, qf: float, t: int | ArrayLike, V: float | None = None
) -> Trajectory:
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
    :rtype: :class:`Trajectory` instance

    Computes a trapezoidal trajectory, which has a linear motion segment with
    parabolic blends.

    - ``tg = trapezoidal(q0, qf, t)`` is a scalar trajectory (Mx1) that varies
      smoothly from ``q0`` to ``qf`` in M steps using a constant velocity
      segment and parabolic blends.  Time ``t`` can be either:

        * an integer scalar, indicating the total number of timesteps

            - Velocity is in units of distance per trajectory step, not per
              second.
            - Acceleration is in units of distance per trajectory step squared,
              *not* per second squared.

        * an array_like, containing the time steps.

            - Results are scaled to units of time.

    - ``tg = trapezoidal(q0, q1, t, V)``  as above but specifies the velocity of the
      linear segment which is normally computed automatically.

    The return value is an object that contains position, velocity and
    acceleration data.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import trapezoidal
        >>> tg = trapezoidal(1, 2, 10)
        >>> tg
        >>> len(tg)
        >>> tg.q

    .. plot::

        from roboticstoolbox import trapezoidal
        tg = trapezoidal(1, 2, 10)
        tg.plot()

    .. note::

        - For some values of ``V`` no solution is possible and an error is flagged.
        - The time vector, if given, is assumed to be monotonically increasing,
          and time scaling is based on the first and last element. The returned
          trajectory retains the supplied time vector.
        - ``tg`` has an extra attribute ``tblend`` which is the blend duration.

    :References:

        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 3.

    :seealso: :func:`quintic`, :func:`trapezoidal_func` :func:`mtraj`.
    """

    tv: NDArray
    if isinstance(t, int):
        tv = np.arange(0, t)
        istime = False
    elif isvector(t):
        tv = getvector(t)
        istime = True
    else:
        raise TypeError("bad argument for time, must be int or vector")

    tf = tv[-1] - tv[0]

    trapezoidalfunc = trapezoidal_func(q0, qf, tf, V)

    # evaluate the polynomials
    p, pd, pdd = trapezoidalfunc(tv - tv[0])

    traj = Trajectory("trapezoidal", tv, p, pd, pdd, istime)
    traj.tblend = trapezoidalfunc.tb
    return traj


class TrapezoidalProfile:
    r"""
    Trapezoidal scalar profile

    :param q0: initial value
    :param qf: final value
    :param T: duration
    :param V: velocity of the linear segment, defaults to 1.5 times the average
        velocity :math:`(q_f - q_0) / T`
    :raises ValueError: ``V`` is too small or too big for a solution to exist

    An object that is called with a time, or an array of times, and returns the
    position, velocity and acceleration :math:`(q(t), \dot{q}(t), \ddot{q}(t))`
    of a trapezoidal velocity profile: a parabolic blend, a linear segment, and
    a parabolic blend.  A scalar time gives scalar values, an array of times
    gives arrays of the same shape.

    The object has attributes:

    - ``tb``, the duration of each blend
    - ``V``, the velocity of the linear segment, signed

    Instances are created by :func:`trapezoidal_func`.
    """

    tb: float
    V: float

    def __init__(self, q0: float, qf: float, T: float, V: float | None = None) -> None:
        if V is None:
            # if velocity not specified, compute it
            vel = (qf - q0) / T * 1.5
        else:
            vel = abs(V) * float(np.sign(qf - q0))
            if abs(vel) < (abs(qf - q0) / T):
                raise ValueError("V too small")
            elif abs(vel) > (2 * abs(qf - q0) / T):
                raise ValueError("V too big")

        if vel == 0:
            self.tb = np.inf
            self._a = 0.0
        else:
            self.tb = (q0 - qf + vel * T) / vel
            self._a = vel / self.tb

        self.V = vel
        self._q0 = q0
        self._qf = qf
        self._T = T

    @overload
    def __call__(self, t: float) -> tuple[float, float, float]: ...

    @overload
    def __call__(self, t: ArrayLike) -> tuple[NDArray, NDArray, NDArray]: ...

    def __call__(self, t: Any) -> Any:
        """
        Evaluate the profile

        :param t: time, or array of times
        :return: position, velocity and acceleration, each with the shape of ``t``
        """
        q0, qf, T, V, tb, a = self._q0, self._qf, self._T, self.V, self.tb, self._a

        times = np.asarray(t, dtype=float)
        p = np.empty(times.shape)
        pd = np.empty(times.shape)
        pdd = np.empty(times.shape)
        for k, tk in enumerate(times.flat):
            if tk < 0:
                pk, pdk, pddk = q0, 0.0, 0.0
            elif tk <= tb:
                # initial blend
                pk = q0 + a / 2 * tk**2
                pdk = a * tk
                pddk = a
            elif tk <= (T - tb):
                # linear motion
                pk = (qf + q0 - V * T) / 2 + V * tk
                pdk = V
                pddk = 0.0
            elif tk <= T:
                # final blend
                pk = qf - a / 2 * T**2 + a * T * tk - a / 2 * tk**2
                pdk = a * T - a * tk
                pddk = -a
            else:
                pk, pdk, pddk = qf, 0.0, 0.0
            p.flat[k] = pk
            pd.flat[k] = pdk
            pdd.flat[k] = pddk

        if times.ndim == 0:
            return p[()], pd[()], pdd[()]
        return p, pd, pdd


def trapezoidal_func(
    q0: float, qf: float, T: float, V: float | None = None
) -> TrapezoidalProfile:
    r"""
    Trapezoidal scalar profile as a function

    :param q0: initial value
    :param qf: final value
    :param T: maximum time
    :param V: velocity of linear segment, defaults to 1.5 times the average
    :return: trapezoidal profile function :math:`f: t \mapsto (q(t), \dot{q}(t), \ddot{q}(t))`
    :rtype: :class:`TrapezoidalProfile`
    :raises ValueError: ``V`` is too small or too big for a solution to exist

    Returns a function which computes the specific trapezoidal profile, and its
    derivatives, as described by the parameters.  The result is called with a
    time, or an array of times, and has the attributes ``tb``, the blend
    duration, and ``V``, the velocity of the linear segment.

    A scalar time gives scalar values and an array of times gives arrays of the
    same shape, as for :func:`quintic_func`.

    .. versionchanged:: 1.5.0
        A scalar time gives scalar values, it previously gave arrays with one
        element.  The result is a :class:`TrapezoidalProfile` object, which is
        called in the same way as the function it replaces.

    Example:

    .. runblock:: pycon

        >>> from roboticstoolbox import trapezoidal_func
        >>> f = trapezoidal_func(1, 2, 5)
        >>> f(0)
        >>> f(5)
        >>> f(2.5)
        >>> f.tb
    """
    return TrapezoidalProfile(q0, qf, T, V)


# -------------------------------------------------------------------------- #


def mtraj(
    tfunc: Callable[..., Trajectory], q0: ArrayLike, qf: ArrayLike, t: int | ArrayLike
) -> Trajectory:
    """
    Multi-axis trajectory

    :param tfunc: a 1D trajectory function, eg. :func:`quintic` or :func:`trapezoidal`
    :param q0: initial configuration
    :type q0: array_like(m)
    :param qf: final configuration
    :type qf: array_like(m)
    :param t: time vector or number of steps
    :raises TypeError: ``tfunc`` is not callable
    :raises ValueError: length of ``q0`` and ``qf`` are different
    :return: trajectory

    - ``tg = mtraj(func, q0, qf, n)`` is a multi-axis trajectory varying
      from configuration ``q0`` (M) to ``qf`` (M) according to the scalar trajectory
      function ``tfunc`` in ``n`` steps.

    - ``tg = mtraj(func, q0, qf, t)`` as above but ``t`` is a uniformly-spaced time
      vector

    The scalar trajectory function is applied to each axis::

            tg = tfunc(s0, sF, n)

    and possible values of ``tfunc`` include ``trapezoidal`` for a trapezoidal trajectory, or
    ``quintic`` for a polynomial trajectory.

    The return value is an object that contains position, velocity and
    acceleration data.

    .. note:: The time vector, if given, is assumed to be monotonically increasing, and
        time scaling is based on the first and last element.

    :References:
        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 3.

    :seealso: :func:`quintic`, :func:`trapezoidal`
    """

    if not callable(tfunc):
        raise TypeError("first argument must be a function reference")

    q0_ = getvector(q0)
    qf_ = getvector(qf)
    if len(q0_) != len(qf_):
        raise ValueError("must be same number of elements in q0 and qf")

    traj = []
    for i in range(len(q0_)):
        # for each axis
        traj.append(tfunc(q0_[i], qf_[i], t))

    x = traj[0].t
    y = np.array([tg.s for tg in traj]).T
    yd = np.array([tg.sd for tg in traj]).T
    ydd = np.array([tg.sdd for tg in traj]).T

    istime = traj[0].istime

    return Trajectory("mtraj", x, y, yd, ydd, istime)


# -------------------------------------------------------------------------- #


def jtraj(
    q0: ArrayLike,
    qf: ArrayLike,
    t: int | ArrayLike,
    qd0: ArrayLike | None = None,
    qd1: ArrayLike | None = None,
) -> Trajectory:
    """
    Compute a joint-space trajectory

    :param q0: initial joint coordinate
    :type q0: array_like(n)
    :param qf: final joint coordinate
    :type qf: array_like(n)
    :param t: time vector or number of steps
    :param qd0: initial velocity, defaults to zero
    :type qd0: array_like(n), optional
    :param qd1: final velocity, defaults to zero
    :type qd1: array_like(n), optional
    :raises ValueError: ``q0``, ``qf``, ``qd0`` and ``qd1`` are not the same size
    :return: trajectory

    - ``tg = jtraj(q0, qf, N)`` is a joint space trajectory where the joint
      coordinates vary from ``q0`` (M) to ``qf`` (M).  A quintic (5th order)
      polynomial is used with default zero boundary conditions for velocity and
      acceleration.  Time is assumed to vary from 0 to 1 in ``N`` steps.

    - ``tg = jtraj(q0, qf, t)`` as above but ``t`` is a uniformly-spaced time
      vector

    The return value is an object that contains position, velocity and
    acceleration data.

    .. note:: The time vector, if given, scales the velocity and acceleration outputs
        assuming that the time vector starts at zero and increases linearly.

    :References:

        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 3.

    :seealso: :func:`ctraj`, :func:`qplot`, :func:`~SerialLink.jtraj`
    """
    # print(f"  --- jtraj: {q0} --> {q1} in {tv}")
    if isinstance(t, int):
        tscal = 1.0
        ts = np.linspace(0, 1, t)  # normalized time from 0 -> 1
        tv = ts * t
    else:
        tarray = np.asarray(t, dtype=float)
        tv = tarray.flatten()
        tscal = float(np.max(tarray))
        ts = tv / tscal

    q0_ = getvector(q0)
    qf_ = getvector(qf)

    if not len(q0_) == len(qf_):
        raise ValueError("q0 and q1 must be same size")

    if qd0 is None:
        qd0_ = np.zeros(q0_.shape)
    else:
        qd0_ = getvector(qd0)
        if not len(qd0_) == len(q0_):
            raise ValueError("qd0 has wrong size")
    if qd1 is None:
        qd1_ = np.zeros(q0_.shape)
    else:
        qd1_ = getvector(qd1)
        if not len(qd1_) == len(q0_):
            raise ValueError("qd1 has wrong size")

    # compute the polynomial coefficients
    A = 6 * (qf_ - q0_) - 3 * (qd1_ + qd0_) * tscal
    B = -15 * (qf_ - q0_) + (8 * qd0_ + 7 * qd1_) * tscal
    C = 10 * (qf_ - q0_) - (6 * qd0_ + 4 * qd1_) * tscal
    E = qd0_ * tscal  # as the t vector has been normalized
    F = q0_

    # n = len(q0)

    tt = np.array([ts**5, ts**4, ts**3, ts**2, ts, np.ones(ts.shape)]).T
    coeffs = np.array([A, B, C, np.zeros(A.shape), E, F])  # 6xN

    qt = tt @ coeffs

    # compute  velocity
    coeffs = np.array([np.zeros(A.shape), 5 * A, 4 * B, 3 * C, np.zeros(A.shape), E])
    qdt = tt @ coeffs / tscal

    # compute  acceleration
    coeffs = np.array(
        [np.zeros(A.shape), np.zeros(A.shape), 20 * A, 12 * B, 6 * C, np.zeros(A.shape)]
    )
    qddt = tt @ coeffs / tscal**2

    return Trajectory("jtraj", tv, qt, qdt, qddt, istime=True)


# -------------------------------------------------------------------------- #


def ctraj(
    T0: SE3, T1: SE3, t: int | ArrayLike | None = None, s: ArrayLike | None = None
) -> SE3:
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

    .. note::

        - In the second case ``s`` could be generated by a scalar trajectory
          generator such as ``quintic`` or ``trapezoidal`` (default).
        - Orientation interpolation is performed using unit-quaternion
          interpolation.

    :References:
        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 3.

    :seealso: :func:`~roboticstoolbox.trajectory.trapezoidal`,
        :func:`~spatialmath.unitquaternion.interp`
    """

    if isinstance(t, int):
        sv = trapezoidal(0, 1, t).s
    elif t is not None and isvector(t):
        tv = getvector(t)
        tv = tv / np.max(tv)
        # Preserve sampling of the zero-origin profile, including unordered samples.
        sv = trapezoidal_func(0, 1, float(np.max(tv)))(tv)[0]
    elif s is not None and isvector(s):
        sv = getvector(s)
    else:
        raise TypeError("bad argument for time, must be int or vector")

    return T0.interp(T1, sv)


def cmstraj() -> None:
    # Cartesian version of mstraj, via points are SE3
    # perhaps refactor mstraj to allow interpolation of any type
    pass


# -------------------------------------------------------------------------- #


def mstraj(
    viapoints: np.ndarray,
    dt: float,
    tacc: float | ArrayLike,
    qdmax: float | ArrayLike | None = None,
    tsegment: ArrayLike | None = None,
    q0: ArrayLike | None = None,
    qd0: ArrayLike | None = None,
    qdf: ArrayLike | None = None,
    verbose: bool = False,
) -> Trajectory:
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
    :param qd0: initial velocity, defaults to zero
    :type qd0: array_like(n), optional
    :param qdf: final velocity, defaults to zero
    :type qdf: array_like(n), optional
    :param verbose: print debug information, defaults to False
    :type verbose: bool, optional
    :return: trajectory
    :rtype: :class:`Trajectory` instance

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

            traj = mstraj(viapoints, dt, tacc, tsegment=TS)

    #. Governed by the speed of the slowest axis for the segment.  The axis
       speed is a scalar (all axes have the same speed) or an N-vector of speed
       per axis::

            traj = mstraj(viapoints, dt, tacc, qdmax=SPEED)

    The returned :class:`Trajectory` has attributes:

        - ``t`` the time coordinate in seconds, shape=(K,)
        - ``q`` the axis values, shape=(K,N)
        - ``qd`` the axis velocities in units per second, shape=(K,N)
        - ``qdd`` the axis accelerations in units per second squared, shape=(K,N)
        - ``arrive`` an array of arrival times for each segment
        - ``info`` a list of named tuples, one per segment, that describe the
          slowest axis, segment time,  and time stamp
        - ``via`` the passed set of via points (excluding the initial row if
          ``q0`` is omitted)

    Trajectory arrays have one row per time step and one column per axis.

    .. note::

        - Only one of ``qdmax`` or ``tsegment`` can be specified
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
        - With ``tacc=0``, velocity changes at via points are instantaneous;
          the sampled acceleration is zero within each linear segment.
        - The returned trajectory also has ``arrive``, ``info`` and ``via``.

    :References:
        - Robotics, Vision & Control in Python, 3e, P. Corke, Springer 2023, Chap 3.

    :seealso: :func:`trapezoidal`, :func:`ctraj`, :func:`mtraj`
    """

    # q0 is overwritten below by the last point of the linear motion, as it
    # always was, so keep it as a variable of its own
    q0v: NDArray
    if q0 is None:
        q0v = viapoints[0, :]
        viapoints = viapoints[1:, :]
    else:
        q0v = getvector(q0)
        if not viapoints.shape[1] == len(q0v):
            raise ValueError("WP and Q0 must have same number of columns")

    ns, nj = viapoints.shape
    tseg_given: NDArray | None = None if tsegment is None else getvector(tsegment)
    qdmax_v: NDArray | None = None

    if qdmax is not None and tsegment is not None:
        raise ValueError("cannot specify both qdmax and tsegment")

    if qdmax is None:
        if tseg_given is None:
            raise ValueError("tsegment must be given if qdmax is not")

        if not len(tseg_given) == ns:
            raise ValueError("Length of TSEG does not match number of viapoints")

    if tseg_given is None:
        # This is unreachable, left just in case
        if qdmax is None:  # pragma nocover
            raise ValueError("qdmax must be given if tsegment is not")

        if isinstance(qdmax, (int, float)):
            # if qdmax is a scalar assume all axes have the same speed
            qdmax_v = np.tile(qdmax, (nj,))
        else:
            qdmax_v = getvector(qdmax)

            if not len(qdmax_v) == nj:
                raise ValueError("Length of QDMAX does not match number of axes")

    Tacc: NDArray
    if isinstance(tacc, (int, float)):
        Tacc = np.tile(tacc, (ns,))
    else:
        Tacc = getvector(tacc)
        if not len(Tacc) == ns:
            raise ValueError("Tacc is wrong size")
    qd0v: NDArray
    if qd0 is None:
        qd0v = np.zeros((nj,))
    else:
        qd0v = getvector(qd0)
        if not len(qd0v) == len(q0v):
            raise ValueError("qd0 is wrong size")
    qdfv: NDArray
    if qdf is None:
        qdfv = np.zeros((nj,))
    else:
        qdfv = getvector(qdf)
        if not len(qdfv) == len(q0v):
            raise ValueError("qdf is wrong size")

    # set the initial conditions
    q_prev = q0v
    qd_prev = qd0v

    clock: float = 0  # keep track of time
    arrive = np.zeros((ns,))  # record planned time of arrival at via points
    q_parts = []
    qd_parts = []
    qdd_parts = []
    infolist = []
    mstraj_info = namedtuple("mstraj_info", "slowest segtime clock")

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

        tacc_seg = math.ceil(Tacc[seg] / dt) * dt
        tacc2 = math.ceil(tacc_seg / 2 / dt) * dt
        if seg == 0:
            taccx = tacc2
        else:
            taccx = tacc_seg

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

        slowest: int | float | None
        tseg: float
        if qdmax_v is not None:
            # qdmax is specified, compute slowest axis

            # qb = taccx * qdmax / 2       # distance moved during blend
            tb = taccx

            # convert to time
            tl = abs(dq) / qdmax_v
            # tl = abs(dq - qb) / qdmax
            tl = np.ceil(tl / dt) * dt

            # find the total time and slowest axis
            tt = tb + tl
            slowest = int(np.argmax(tt))
            tseg = tt[slowest]

            # best if there is some linear motion component
            if tseg <= 2 * tacc_seg:
                tseg = 2 * tacc_seg

        elif tseg_given is not None:
            # segment time specified, use that
            tseg = tseg_given[seg]
            slowest = math.nan

        infolist.append(mstraj_info(slowest, tseg, clock))

        # log the planned arrival time
        arrive[seg] = clock + tseg
        if seg > 0:
            arrive[seg] += tacc2

        if verbose:  # pragma nocover
            print(
                f"seg {seg}, distance {dq}, "
                f"slowest axis {slowest}, time required {tseg}"
            )

        # create the trajectories for this segment

        # linear velocity from qprev to qnext
        qd = dq / tseg

        # add the blend polynomial
        if taccx > 0:
            blend = jtraj(
                q0v, q_prev + tacc2 * qd, mrange(0, taccx, dt), qd0=qd_prev, qd1=qd
            )
            if verbose:  # pragma nocover
                print(blend.q)
            q_parts.append(blend.q[1:, :])
            qd_parts.append(blend.qd[1:, :])
            qdd_parts.append(blend.qdd[1:, :])

        clock = clock + taccx  # update the clock

        # add the linear part, from tacc/2+dt to tseg-tacc/2
        for t in mrange(tacc2 + dt, tseg - tacc2, dt):
            s = t / tseg
            q0v = (1 - s) * q_prev + s * q_next  # linear step
            if verbose:  # pragma nocover
                print(t, s, q0v)
            q_parts.append(q0v[np.newaxis, :])
            qd_parts.append(qd[np.newaxis, :])
            qdd_parts.append(np.zeros((1, nj)))
            clock += dt

        q_prev = q_next  # next target becomes previous target
        qd_prev = qd

    # add the final blend
    if tacc2 > 0:
        blend = jtraj(q0v, q_next, mrange(0, tacc2, dt), qd0=qd_prev, qd1=qdfv)
        q_parts.append(blend.q[1:, :])
        qd_parts.append(blend.qd[1:, :])
        qdd_parts.append(blend.qdd[1:, :])

    infolist.append(mstraj_info(None, tseg, clock))

    q = np.vstack(q_parts)
    qd = np.vstack(qd_parts)
    qdd = np.vstack(qdd_parts)
    traj = Trajectory("mstraj", dt * np.arange(0, q.shape[0]), q, qd, qdd, istime=True)
    traj.arrive = arrive
    traj.info = infolist
    traj.via = viapoints

    return traj


if __name__ == "__main__":
    # t = quintic(0, 1, 50)
    # t.plot()

    # t = quintic(0, 1, np.linspace(0, 1, 50))
    # t.plot()

    # t = trapezoidal(0, 1, 50)
    # t.plot()
    # t = trapezoidal(0, 1, np.linspace(0, 1, 50))
    # t.plot(block=True)

    from spatialmath import SO2

    from roboticstoolbox import *

    # puma = models.DH.Puma560()

    # traj = jtraj(puma.qz, puma.qr, 100)
    # traj.plot(block=True)

    via = SO2(30, unit="deg") * np.array([[-1, 1, 1, -1, -1], [1, 1, -1, -1, 1]])
    traj0 = mstraj(via.T, dt=0.2, tacc=0.5, qdmax=[2, 1])
    xplot(traj0.q[:, 0], traj0.q[:, 1], color="red")
    traj0.plot(block=True)
