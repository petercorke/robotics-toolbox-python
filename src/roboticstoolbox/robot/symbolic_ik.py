r"""
Prototype: symbolic (closed-form) inverse kinematics for serial-link robots.

Descended from ``@SerialLink/ikine_sym.m`` in RTB for MATLAB, which implements
the classical approach based on Pieper's method.  This is the seed of an
IKFast-like capability: take a robot model and derive analytical,
non-iterative joint-angle expressions from it.

This file is deliberately self-contained (code and tests together) while the
design is evolving.  Run the tests with::

    pytest src/roboticstoolbox/robot/symbolic_ik.py

Approach
--------

The forward kinematics :math:`T = A_1 A_2 \cdots A_N` is rearranged by moving
inverse link transforms to the left and/or right of :math:`T`, see :func:`_pieper`,
giving many *equivalent* sets of scalar equations in which the joint variables
are separated in different ways.  All of them go into one pool.  Each joint
:math:`j` appears only through the symbols ``Sj`` and ``Cj`` (:math:`\sin q_j`
and :math:`\cos q_j`), or ``qj`` for a prismatic joint, so the equations are
polynomials.

The solver keeps track of which joints are unknown and repeats:

1. an equation with a single unknown joint is solved for it, which makes that
   joint known and so reduces the number of unknowns in other equations;
2. otherwise, eliminate a joint from a pair of equations that both contain it,
   to create a new equation with fewer unknowns, preferring the cheapest pairs.

If no equation can be found the method fails and returns ``None``.

An equation that is true but of no use, because the coefficients of the unknown
are zero whenever the other joints satisfy their own equations, is rejected.
This can't be seen symbolically without the relations, so it is tested
numerically on a few random configurations of the robot.  Testing it
symbolically needs a Groebner basis, which was about 160 s for a UR5.

Differences from the MATLAB original:

- ``pieper`` used ``nx`` instead of ``nz`` in the third row of the symbolic
  pose matrix, which is fixed here.
- The original solved joints 1, 2, 3... in order, using only the *left* form of
  the equations.  Here the left, right and mixed forms are pooled and the order
  of solution emerges.
- Diagnostic ``fprintf`` output is under a ``verbose`` flag.
- The ``'file'`` option / code generation is not ported yet.

Status: of the 6-axis DH models in ``roboticstoolbox.models.DH`` seven are
solved (IRB140, KR5, Puma560, Stanford, UR3, UR5, UR10) and checked against
forward kinematics.  Jaco and Mico, which have no wrist centre, are not.

Known limitations:

- Only standard DH robots are handled, since link transforms come from
  ``Link.A``.
- Elimination needs equations that are linear in ``Sj``, ``Cj``, which is why
  the general 6R case (Jaco, Mico) fails.  That needs a resultant based
  elimination, and the roots of a univariate polynomial in ``tan(q/2)``, see
  Raghavan and Roth, J. Mech. Design 115(3), 1993.
- Spurious branches are not removed: some combinations of the alternatives for
  each joint do not reach the pose.  The equations that are fully determined
  once all joints are solved could be used to prune them.
- Solutions are expressions in terms of the pose elements, the link parameters
  and ``Si``, ``Ci`` of other joints, and are not simplified.
"""

from __future__ import annotations

import re
import signal
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import numpy as np
import sympy as sp
from sympy.printing.pycode import PythonCodePrinter

from roboticstoolbox.robot.DHRobot import DHRobot

# symbolic elements of the homogeneous transform
#      nx ox ax tx
#      ny oy ay ty
#      nz oz az tz
_POSE_SYMBOLS = sp.symbols("nx ox ax tx ny oy ay ty nz oz az tz", real=True)

# the pose elements that are specified for a given number of DOF
_SPECIFIED = {
    2: {"tx", "ty"},
    3: {"tx", "ty", "tz"},
    6: {s.name for s in _POSE_SYMBOLS},
}


def _pose_matrix() -> sp.Matrix:
    """Symbolic 4x4 end-effector pose with elements ``nx ... tz``."""
    nx, ox, ax, tx, ny, oy, ay, ty, nz, oz, az, tz = _POSE_SYMBOLS
    return sp.Matrix(
        [
            [nx, ox, ax, tx],
            [ny, oy, ay, ty],
            [nz, oz, az, tz],
            [0, 0, 0, 1],
        ]
    )


def _joint_var(j: int) -> sp.Symbol:
    """Joint variable ``qj``, j is 1-based."""
    return sp.Symbol(f"q{j}", real=True)


def _sin_cos_symbols(j: int) -> tuple[sp.Symbol, sp.Symbol]:
    """Symbols ``Sj``, ``Cj`` that stand in for sin(qj), cos(qj), j is 1-based."""
    return sp.Symbol(f"S{j}", real=True), sp.Symbol(f"C{j}", real=True)


def _inv_se3(T: sp.Matrix) -> sp.Matrix:
    """Inverse of a homogeneous transform, exploiting its structure."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = sp.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T * t
    return Ti


# ------------------------------------------------------------------------- #
#                         making a model symbolic                           #
# ------------------------------------------------------------------------- #


def _symbolize(robot: DHRobot) -> tuple[DHRobot, dict[str, float]]:
    r"""
    Rationalize a numeric DH model

    :param robot: DH robot, standard convention
    :returns: ``(symbolic_robot, values)``

    Nonzero link lengths ``a`` and ``d`` become symbols ``a1``, ``d2`` etc.,
    zero lengths become an exact zero.  Angles (``alpha``, ``offset``) become
    exact multiples of :math:`\pi`.  Parameters that are already symbolic are
    left alone.  ``values`` maps the names of the symbols created to their
    numeric values.

    The base and tool transforms are not handled, and are assumed to be the
    identity.
    """
    from roboticstoolbox import PrismaticDH, RevoluteDH

    values: dict[str, float] = {}

    def length(x, name: str):
        if isinstance(x, sp.Basic):
            return x
        if x == 0:
            return sp.Integer(0)
        values[name] = float(x)
        return sp.Symbol(name, real=True)

    def angle(x):
        if isinstance(x, sp.Basic):
            return x
        r = sp.nsimplify(float(x), [sp.pi], tolerance=1e-9)
        if r.has(sp.Float):
            raise ValueError(f"cannot express angle {x} as a multiple of pi")
        return r

    links = []
    for j, link in enumerate(robot.links, 1):
        if link.mdh:
            raise NotImplementedError("modified DH links are not supported")
        kw = dict(
            a=length(link.a, f"a{j}"),
            alpha=angle(link.alpha),
            offset=angle(link.offset),
            flip=link.isflip,
        )
        if link.isrevolute:
            links.append(RevoluteDH(d=length(link.d, f"d{j}"), **kw))
        else:
            links.append(PrismaticDH(theta=angle(link.theta), **kw))

    return DHRobot(links, name=robot.name), values


def _link_matrices(robot: DHRobot) -> list[sp.Matrix]:
    """Link matrices in terms of the symbols ``Sj``, ``Cj`` (or ``qj``)."""
    A = []
    for j, link in enumerate(robot.links, 1):
        q = _joint_var(j)
        Aj = sp.Matrix(link.A(q).A)
        if link.isrevolute:
            # sin(q + offset) -> sin(q) cos(offset) + ... so that only sin(q)
            # and cos(q) appear, then replace those by symbols
            Sj, Cj = _sin_cos_symbols(j)
            Aj = Aj.applyfunc(sp.expand_trig).subs({sp.sin(q): Sj, sp.cos(q): Cj})
        A.append(Aj)
    return A


# ------------------------------------------------------------------------- #
#                         equations and unknowns                            #
# ------------------------------------------------------------------------- #

_JOINT_SYMBOL = re.compile(r"([SCq])(\d+)")


def _joints_in(expr: sp.Expr) -> set[int]:
    """Joints (1-based) whose symbols ``Sj``, ``Cj`` or ``qj`` appear in expr."""
    out = set()
    for s in expr.free_symbols:
        m = _JOINT_SYMBOL.fullmatch(s.name)
        if m:
            out.add(int(m.group(2)))
    return out


def _pieper(
    A: list[sp.Matrix],
    T: sp.Matrix,
    nl: int,
    nr: int = 0,
) -> tuple[sp.Matrix, sp.Matrix]:
    r"""
    Equations from Pieper's method

    :param A: link matrices :math:`A_1 \ldots A_N`
    :param T: symbolic end-effector pose
    :param nl: number of link transforms moved to the left of :math:`T`
    :param nr: number of link transforms moved to the right of :math:`T`

    Return the :math:`4 \times 4` matrices ``(left, right)`` for which
    ``left == right`` is the set of 12 equations

    .. math::

        A_{n_l}^{-1} \cdots A_1^{-1} \, T \, A_N^{-1} \cdots A_{N-n_r+1}^{-1}
        = A_{n_l+1} \cdots A_{N-n_r}

    with :math:`n_l + n_r \le N`.  Joints :math:`1 \ldots n_l` and
    :math:`N-n_r+1 \ldots N` are on the left-hand side and the rest are on the
    right.

    - ``nr = 0`` is the *left* form, and works from the base towards the tip
    - ``nl = 0`` is the *right* form, and works from the tip towards the base
    - both nonzero is the *mixed* form, which among other things isolates the
      wrist centre :math:`p - d_6 a` of a robot with a wrist

    All the forms are the same constraint, :math:`T = A_1 \cdots A_N`, but they
    separate the joint variables differently, so different scalar equations are
    simple.
    """
    N = len(A)
    if nl < 0 or nr < 0 or nl + nr > N:
        raise ValueError("require nl, nr >= 0 and nl + nr <= number of links")

    left = T
    for j in range(nl):
        left = _inv_se3(A[j]) * left
    for j in range(nr):
        left = left * _inv_se3(A[N - 1 - j])

    right = sp.eye(4)
    for j in range(nl, N - nr):
        right = right * A[j]

    return left, right


def _equation_pool(
    A: list[sp.Matrix], dof: int, mixed: bool = True
) -> list[sp.Expr]:
    """
    All distinct scalar equations from the left, right and mixed forms

    Each equation is an expanded expression that equals zero.  Only equations
    whose pose elements are all specified (for the given DOF) are kept.  If
    ``mixed`` is False only the pure left and pure right forms are used.
    """
    specified = _SPECIFIED[dof]
    pool: list[sp.Expr] = []
    seen: set[sp.Expr] = set()

    N = len(A)
    for nl in range(N + 1):
        for nr in range(N + 1 - nl):
            if not mixed and nl > 0 and nr > 0:
                continue
            left, right = _pieper(A, _pose_matrix(), nl, nr)
            for e in (left - right)[:3, :]:
                e = sp.expand(e)
                if e == 0 or e in seen:
                    continue
                used = {s.name for s in e.free_symbols if s in _POSE_SYMBOLS}
                if not used <= specified:
                    continue
                seen.add(e)
                seen.add(sp.expand(-e))
                pool.append(e)
    return pool


def _reduce(expr: sp.Expr, joints: list[int] | set[int]) -> sp.Expr:
    """Replace powers ``Sj**k`` (k >= 2) using ``Sj**2 = 1 - Cj**2``."""
    for j in joints:
        sj, cj = _sin_cos_symbols(j)
        expr = expr.replace(
            lambda e: e.is_Pow and e.base == sj and e.exp.is_Integer and e.exp >= 2,
            lambda e: (1 - cj**2) ** (e.exp // 2) * sj ** (e.exp % 2),
        )
    return sp.expand(expr)


def _linear_form(eq: sp.Expr, j: int, revolute: bool) -> tuple | None:
    r"""
    Coefficients of an equation that is linear in the variable of joint j

    :returns: for a revolute joint ``(a, b, k)`` such that the equation is
        :math:`a C_j + b S_j + k = 0`, for a prismatic joint ``(a, k)`` such
        that it is :math:`a q_j + k = 0`, or ``None`` if the equation has any
        other form
    """
    if revolute:
        sj, cj = _sin_cos_symbols(j)
        P = sp.Poly(_reduce(eq, [j]), sj, cj)
        if P.total_degree() > 1:
            return None
        return P.coeff_monomial(cj), P.coeff_monomial(sj), P.coeff_monomial(1)
    else:
        qj = _joint_var(j)
        P = sp.Poly(eq, qj)
        if P.total_degree() > 1:
            return None
        return P.coeff_monomial(qj), P.coeff_monomial(1)


def _samples(
    A: list[sp.Matrix],
    revolute: list[bool],
    values: dict[str, float],
    count: int = 3,
    seed: int = 0,
) -> list[dict[str, float]]:
    """
    Numeric instances of the problem, for testing equations

    :param A: link matrices
    :param revolute: whether each joint is revolute
    :param values: numeric values for the link parameter symbols, by name.  Any
        other symbols are given random values.
    :returns: a list of dicts, each mapping symbol names to values, for random
        joint values (as ``Sj``, ``Cj``, or ``qj`` if prismatic), the link
        parameters and the pose elements ``nx ... tz`` which result.

    All of the equations are true for these values.
    """
    rng = np.random.default_rng(seed)
    others = {
        s.name
        for m in A
        for s in m.free_symbols
        if not _JOINT_SYMBOL.fullmatch(s.name) and s.name not in values
    }
    out = []
    for _ in range(count):
        vals = dict(values) | {name: rng.uniform(0.2, 1.0) for name in others}
        for j, rev in enumerate(revolute, 1):
            if rev:
                q = rng.uniform(-1.5, 1.5)
                vals[f"S{j}"], vals[f"C{j}"] = np.sin(q), np.cos(q)
            else:
                vals[f"q{j}"] = rng.uniform(0.2, 0.6)
        T = np.eye(4)
        for m in A:
            sub = {s: vals[s.name] for s in m.free_symbols}
            T = T @ np.array(m.xreplace(sub), dtype=float)
        vals |= {sym.name: T[i // 4, i % 4] for i, sym in enumerate(_POSE_SYMBOLS)}
        out.append(vals)
    return out


def _is_degenerate(
    coefficients: tuple[sp.Expr, ...], samples: list[dict[str, float]]
) -> bool:
    """
    True if every coefficient is zero for all of the numeric samples

    An equation like ``A cos(q) + B sin(q) = C`` can be true but of no use
    because ``A`` and ``B`` are zero whenever the other joints satisfy their own
    equations.  This can't be seen symbolically without knowing the relations,
    so test it numerically.
    """
    for vals in samples:
        for c in coefficients:
            sub = {s: vals[s.name] for s in c.free_symbols}
            if abs(float(c.xreplace(sub))) > 1e-7:
                return False
    return True


def _solve_single(
    eq: sp.Expr,
    j: int,
    revolute: bool,
    samples: list[dict[str, float]] | None = None,
) -> list[sp.Expr] | None:
    r"""
    Solve an equation in a single unknown joint

    :param eq: expression which equals zero
    :param j: joint number (1-based), the only unknown joint in ``eq``
    :param revolute: joint is revolute
    :param samples: numeric instances, see :func:`_samples`, used to reject an
        equation whose coefficients are all zero
    :returns: solutions for ``qj``, or ``None`` if the equation can't be solved

    A revolute joint requires the form

    .. math::

        A \cos q_j + B \sin q_j = C

    where :math:`A`, :math:`B` and :math:`C` are arbitrarily complex expressions
    that do not contain :math:`q_j`.  The solutions are

    - :math:`C = 0`: :math:`\mathrm{atan2}(-A, B)` and :math:`\mathrm{atan2}(A, -B)`
    - otherwise: :math:`\mathrm{atan2}(C, \pm r) - \phi` where
      :math:`r = \sqrt{A^2 + B^2 - C^2}` and :math:`\phi = \mathrm{atan2}(A, B)`

    A prismatic joint requires a linear equation.
    """
    form = _linear_form(eq, j, revolute)
    if form is None:
        return None

    if not revolute:
        a, k = form
        if a == 0 or (samples and _is_degenerate((a,), samples)):
            return None
        return [-k / a]

    A, B, k = form
    if A == 0 and B == 0:
        return None
    if samples and _is_degenerate((A, B), samples):
        return None
    C = -k
    if C == 0:
        return [sp.atan2(-A, B), sp.atan2(A, -B)]
    r = sp.sqrt(A**2 + B**2 - C**2)
    phi = sp.atan2(A, B)
    return [sp.atan2(C, r) - phi, sp.atan2(C, -r) - phi]


def _eliminate(
    e1: sp.Expr, e2: sp.Expr, j: int, revolute: bool
) -> list[sp.Expr] | None:
    r"""
    Eliminate a joint from a pair of equations

    :param e1: first equation, expression which equals zero
    :param e2: second equation
    :param j: joint to eliminate
    :param revolute: joint is revolute
    :returns: new equations that are free of joint ``j``, or ``None``

    Both equations must be linear in the variable of the joint.  For a revolute
    joint they are

    .. math::

        a_i C_j + b_i S_j + k_i = 0, \quad i=1,2

    which is solved for :math:`C_j` and :math:`S_j` by Cramer's rule.  Then
    :math:`C_j^2 + S_j^2 = 1` gives an equation without :math:`q_j`.  The
    result is divided by the determinant (a degenerate case) as many times as
    it goes exactly, and the quotient is the new equation.
    """
    f1 = _linear_form(e1, j, revolute)
    f2 = _linear_form(e2, j, revolute)
    if f1 is None or f2 is None:
        return None

    if revolute:
        (a1, b1, k1), (a2, b2, k2) = f1, f2
        det = sp.expand(a1 * b2 - a2 * b1)
        if det == 0:
            return None
        R = sp.expand((-k1 * b2 + k2 * b1) ** 2 + (-a1 * k2 + a2 * k1) ** 2 - det**2)
    else:
        (a1, k1), (a2, k2) = f1, f2
        det = sp.Integer(0)
        R = sp.expand(a2 * k1 - a1 * k2)

    if R == 0:
        return None

    # R = 0 means that at least one factor is zero, not all of them, so the
    # factors cannot be used as separate equations.  The vanishing of the
    # determinant is a degenerate case, so divide it out (as many times as it
    # goes).  If it is not a factor then R is as good as any of its factors
    if revolute and det.free_symbols:
        while R.free_symbols:
            quotient, remainder = sp.div(R, det)
            if remainder != 0:
                break
            R = sp.expand(quotient)

    return [R] if R.free_symbols else None


def _clean(expr: sp.Expr, joints: set[int]) -> sp.Expr:
    """Tidy a solution: remove powers of ``Sj`` using ``Sj**2 = 1 - Cj**2``."""
    for _ in range(3):
        expr = _reduce(expr, joints)
    return expr


class _Timeout(Exception):
    pass


@contextmanager
def _time_limit(seconds: float | None) -> Iterator[None]:
    """
    Raise ``_Timeout`` in the body after a number of seconds

    Individual SymPy operations (expand, factor) can run for a very long time
    and can't be interrupted from within, so this uses a timer signal.  It only
    works on the main thread on Unix, elsewhere there is no limit.
    """
    usable = (
        seconds is not None
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not usable:
        yield
        return

    def handler(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def ikine_sym(
    robot: DHRobot,
    n: int | None = None,
    verbose: bool = False,
    max_time: float | None = 120.0,
    mixed: bool = True,
) -> list[list[sp.Expr]] | None:
    r"""
    Symbolic inverse kinematics

    :param robot: DH robot model
    :param n: number of degrees of freedom to solve, one of 2, 3 or 6.  Must
        equal the number of joints, defaults to that.
    :param verbose: print diagnostic information
    :param max_time: give up after this many seconds (Unix, main thread only),
        ``None`` for no limit
    :param mixed: include the mixed left/right forms of Pieper's equations
    :returns: for each joint, a list of alternative symbolic solutions, or
        ``None`` if a solution could not be found
    :raises ValueError: if ``n`` is not 2, 3 or 6, or is not the number of joints

    The desired end-effector pose is represented by the symbolic matrix

    .. math::

        \begin{pmatrix}
        n_x & o_x & a_x & t_x \\
        n_y & o_y & a_y & t_y \\
        n_z & o_z & a_z & t_z \\
        0 & 0 & 0 & 1
        \end{pmatrix}

    where the first three columns specify orientation and the last column
    translation.  ``n`` can only have specific values:

    - 2 solve for translation :math:`t_x` and :math:`t_y`
    - 3 solve for translation :math:`t_x`, :math:`t_y` and :math:`t_z`
    - 6 solve for translation and orientation

    The model is first made symbolic by :func:`_symbolize`, so numeric models
    can be passed.  The nonzero link lengths ``a`` and ``d`` become symbols
    named ``a1``, ``d2`` etc.

    The result ``Q[j]`` is a list of alternative expressions for joint
    :math:`q_{j+1}`.  The solution for a joint depends on the joints solved
    before it through the symbols ``Si`` and ``Ci`` which represent
    :math:`\sin q_i` and :math:`\cos q_i`.

    Example::

        >>> robot = rtb.models.DH.Planar2(symbolic=True)
        >>> Q = ikine_sym(robot)
        >>> len(Q[0])  # two solutions for q1
        2
    """
    if n is None:
        n = robot.n
    if n not in _SPECIFIED:
        raise ValueError("Can only solve for 2, 3, 6 DOF")
    if n != robot.n:
        raise ValueError(f"robot has {robot.n} joints, n must match")

    try:
        with _time_limit(max_time):
            return _ikine_sym(robot, n, verbose, mixed)
    except _Timeout:
        if verbose:
            print("timed out")
        return None



# limits on the search for eliminations in tier 2
_MAX_UNKNOWNS = 4  # only consider equations with at most this many unknown joints
_PER_GROUP = 4  # cheapest equations to use from each group of unknown joints
_MAX_NEW = 20  # stop after finding this many new equations, and reconsider


def _ikine_sym(
    robot: DHRobot, n: int, verbose: bool, mixed: bool
) -> list[list[sp.Expr]] | None:
    """Implementation of :func:`ikine_sym`."""
    robot, values = _symbolize(robot)
    revolute = {j: link.isrevolute for j, link in enumerate(robot.links, 1)}
    A = _link_matrices(robot)
    samples = _samples(A, list(revolute.values()), values)
    pool = _equation_pool(A, n, mixed)
    seen = set(pool) | {sp.expand(-e) for e in pool}
    if verbose:
        print(f"{len(pool)} equations")

    Q: dict[int, list[sp.Expr]] = {}
    failed: set[tuple[sp.Expr, int]] = set()  # equation can't be solved for j
    tried: set[tuple[sp.Expr, sp.Expr, int]] = set()  # eliminations done

    def unknown(e: sp.Expr) -> set[int]:
        return _joints_in(e) - Q.keys()

    while len(Q) < n:
        # tier 1: an equation with a single unknown joint
        singles = sorted((e for e in pool if len(unknown(e)) == 1), key=sp.count_ops)
        for e in singles:
            (j,) = unknown(e)
            if (e, j) in failed:
                continue
            sols = _solve_single(e, j, revolute[j], samples)
            if sols is None:
                failed.add((e, j))
                continue
            if verbose:
                print(f"solved joint {j}: {len(sols)} solutions")
            Q[j] = sols
            break
        else:
            # tier 2: eliminate a joint from a pair of equations that both
            # contain it.  The new equation has the unknowns of both except
            # that joint, so prefer pairs for which that set is small, and
            # don't make equations with more unknowns than we started with.
            # Do the cheapest ones first and stop as soon as one produces an
            # equation with a single unknown so that tier 1 can use it
            cost: dict[sp.Expr, int] = {}
            by_joint: dict[int, dict[frozenset[int], list[sp.Expr]]] = {}
            for e in pool:
                u = frozenset(unknown(e))
                if 2 <= len(u) <= _MAX_UNKNOWNS:
                    cost[e] = sp.count_ops(e)
                    for j in u:
                        by_joint.setdefault(j, {}).setdefault(u, []).append(e)

            candidates = []
            for j, groups in by_joint.items():
                # the few cheapest equations from each group of unknowns
                eqs = [
                    (frozenset(u), e)
                    for u, g in groups.items()
                    for e in sorted(g, key=cost.get)[:_PER_GROUP]
                ]
                for i1, (u1, e1) in enumerate(eqs):
                    for u2, e2 in eqs[i1 + 1 :]:
                        r = len(u1 | u2) - 1
                        if r <= max(len(u1), len(u2)) and (e1, e2, j) not in tried:
                            candidates.append((r, cost[e1] + cost[e2], e1, e2, j))
            candidates.sort(key=lambda c: c[:2])
            if verbose:
                sizes = sorted(len(unknown(e)) for e in pool if unknown(e))
                print(
                    f"  pass: {len(candidates)} candidate eliminations, "
                    f"{len(tried)} done, pool {len(pool)}, "
                    f"fewest unknowns {sizes[:6]}"
                )

            new: list[sp.Expr] = []
            for _, _, e1, e2, j in candidates:
                tried.add((e1, e2, j))
                for r in _eliminate(e1, e2, j, revolute[j]) or []:
                    if r not in seen:
                        seen.add(r)
                        seen.add(sp.expand(-r))
                        new.append(r)
                        if verbose:
                            print(
                                f"  eliminate q{j} from {sorted(unknown(e1))} "
                                f"& {sorted(unknown(e2))} -> {sorted(unknown(r))}"
                            )
                if any(len(unknown(r)) <= 1 for r in new) or len(new) >= _MAX_NEW:
                    break
            if not new:
                if verbose:
                    print(f"stuck, solved joints {sorted(Q)}")
                return None
            if verbose:
                print(f"eliminated, {len(new)} new equations")
            pool.extend(new)

    done = set(Q)
    return [[_clean(s, done) for s in Q[j]] for j in range(1, n + 1)]


# ------------------------------------------------------------------------- #
#                            code generation                                #
# ------------------------------------------------------------------------- #


def _py(expr: sp.Expr) -> str:
    """Python source for an expression, square roots via ``safe_sqrt``."""
    expr = expr.replace(
        lambda e: e.is_Pow and e.exp == sp.S.Half,
        lambda e: sp.Function("safe_sqrt")(e.base),
    )
    printer = PythonCodePrinter({"user_functions": {"safe_sqrt": "safe_sqrt"}})
    return printer.doprint(expr)


def generate_python(robot: DHRobot, name: str = "ikine") -> str:
    """
    Generate Python source for the inverse kinematics of a robot

    :param robot: DH robot model with numeric parameters
    :param name: name of the generated function
    :returns: Python source code

    The code defines ``name(T, tol=1e-6)`` which takes the end-effector pose, a
    4x4 array-like, and returns a list of tuples, each of which is a joint
    configuration that reaches the pose.  It depends only on the standard
    library.  The link parameters are constants in the code.

    The joints are solved in the order found by :func:`ikine_sym`, with a nested
    loop over the alternative solutions for each joint.  Each configuration is
    checked against the forward kinematics and only those that reach the pose,
    to within ``tol``, are returned.

    :raises ValueError: if no solution is found, or the model has symbolic
        parameters
    """
    Q = ikine_sym(robot, max_time=None)
    if Q is None:
        raise ValueError("no symbolic solution for this robot")

    symrobot, values = _symbolize(robot)
    n = robot.n
    revolute = [link.isrevolute for link in symrobot.links]

    def numeric(e: sp.Expr) -> sp.Expr:
        sub = {s: sp.Float(values[s.name]) for s in e.free_symbols if s.name in values}
        e = e.xreplace(sub)
        pose = {s.name for s in _POSE_SYMBOLS}
        free = {
            s.name
            for s in e.free_symbols
            if s.name not in pose and not _JOINT_SYMBOL.fullmatch(s.name)
        }
        if free:
            raise ValueError(f"model has symbolic parameters: {sorted(free)}")
        return e

    # forward kinematics, to check each configuration
    T = sp.eye(4)
    for A in _link_matrices(symrobot):
        T = T * A
    fk = [numeric(sp.expand(T[i, j])) for i in range(3) for j in range(4)]
    fk_repl, fk_red = sp.cse(fk, symbols=sp.numbered_symbols("y"))
    pose = [s.name for s in _POSE_SYMBOLS]

    lines = [
        '"""Inverse kinematics generated by roboticstoolbox symbolic_ik."""',
        "",
        "import math",
        "",
        "",
        "def safe_sqrt(x):",
        "    # round-off can make a zero argument slightly negative",
        "    return math.nan if x < -1e-9 else math.sqrt(max(x, 0.0))",
        "",
        "",
        f"def {name}(T, tol=1e-6):",
        f'    """All joint configurations (q1..q{n}) for which the pose is T."""',
        "    " + ", ".join(pose[0:4]) + " = T[0][0], T[0][1], T[0][2], T[0][3]",
        "    " + ", ".join(pose[4:8]) + " = T[1][0], T[1][1], T[1][2], T[1][3]",
        "    " + ", ".join(pose[8:12]) + " = T[2][0], T[2][1], T[2][2], T[2][3]",
        "    solutions = []",
    ]

    order = _solve_order(Q)

    def emit(step: int, indent: str) -> None:
        if step == n:
            for sym, e in fk_repl:
                lines.append(f"{indent}{sym} = {_py(e)}")
            terms = ", ".join(f"abs({_py(e)} - {p})" for e, p in zip(fk_red, pose))
            lines.append(f"{indent}if max({terms}) < tol:")
            qs = ", ".join(f"q{j}" for j in range(1, n + 1))
            lines.append(f"{indent}    solutions.append(({qs},))")
            return

        j = order[step]
        exprs = [numeric(e) for e in Q[j]]
        repl, red = sp.cse(exprs, symbols=sp.numbered_symbols(f"x{j + 1}_"))
        for sym, e in repl:
            lines.append(f"{indent}{sym} = {_py(e)}")
        lines.append(f"{indent}for q{j + 1} in ({', '.join(_py(e) for e in red)},):")
        lines.append(f"{indent}    if q{j + 1} != q{j + 1}:  # NaN, not reachable")
        lines.append(f"{indent}        continue")
        if revolute[j]:
            lines.append(f"{indent}    S{j + 1} = math.sin(q{j + 1})")
            lines.append(f"{indent}    C{j + 1} = math.cos(q{j + 1})")
        emit(step + 1, indent + "    ")

    emit(0, "    ")
    lines.append("    return solutions")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------- #
#                                  tests                                    #
# ------------------------------------------------------------------------- #


def _solve_order(Q: list[list[sp.Expr]]) -> list[int]:
    """
    Joint indices (0-based) in an order in which each only depends on those
    before it, from the ``Si``, ``Ci`` symbols in their solutions.
    """
    deps = {j: {i - 1 for e in sols for i in _joints_in(e)} for j, sols in enumerate(Q)}
    order: list[int] = []
    while len(order) < len(Q):
        j = next(j for j in deps if j not in order and deps[j] <= set(order))
        order.append(j)
    return order


def _safe_sqrt(x):
    """Square root that tolerates round-off: tiny negative values give 0, larger
    negative values (an unreachable branch) give NaN."""
    x = np.asarray(x, dtype=float)
    return np.where(x < -1e-9, np.nan, np.sqrt(np.maximum(x, 0.0)))


def _compile(expr: sp.Expr):
    """Compile a solution expression to ``(symbols, function)``."""
    safe = sp.Function("safe_sqrt")
    expr = expr.replace(
        lambda e: e.is_Pow and e.exp == sp.S.Half, lambda e: safe(e.base)
    )
    args = sorted(expr.free_symbols, key=lambda s: s.name)
    func = sp.lambdify(args, expr, modules=[{"safe_sqrt": _safe_sqrt}, "numpy"], cse=True)
    return args, func


def _numeric_solutions(
    Q: list[list[sp.Expr]],
    values: dict[str, float],
    revolute: list[bool] | None = None,
) -> list[np.ndarray]:
    """
    Evaluate symbolic IK solutions for every combination of branches

    :param values: numeric value of symbols, keyed by symbol *name* since the
        symbols in a model may have been created with different assumptions
    :param revolute: whether each joint is revolute, default all

    Solutions are evaluated in dependency order.  The resulting sin and cos (or
    the joint value for a prismatic joint) are fed to the solutions for the
    joints that depend on them.  A branch that is not real, because it is
    outside the workspace, is dropped.
    """
    n = len(Q)
    revolute = revolute or [True] * n
    order = _solve_order(Q)
    compiled = [[_compile(e) for e in sols] for sols in Q]

    def recurse(step: int, vals: dict[str, float], qs: dict[int, float]) -> list[dict]:
        if step == n:
            return [qs]
        j = order[step]
        out = []
        for args, func in compiled[j]:
            qj = float(func(*[vals[a.name] for a in args]))
            if np.isnan(qj):
                continue
            if revolute[j]:
                new = {f"S{j + 1}": np.sin(qj), f"C{j + 1}": np.cos(qj)}
            else:
                new = {f"q{j + 1}": qj}
            out += recurse(step + 1, {**vals, **new}, {**qs, j: qj})
        return out

    return [np.array([x[j] for j in range(n)]) for x in recurse(0, dict(values), {})]


def _wrap(q: np.ndarray) -> np.ndarray:
    """Wrap angles to the interval (-pi, pi]."""
    return np.arctan2(np.sin(q), np.cos(q))


def _pose_values(T: np.ndarray) -> dict[str, float]:
    """Map the elements of a numeric 4x4 pose to the symbolic pose elements."""
    return {s.name: v for s, v in zip(_POSE_SYMBOLS, T[:3, :].flatten())}


def test_planar2():
    import roboticstoolbox as rtb

    robot = rtb.models.DH.Planar2(symbolic=True)
    Q = ikine_sym(robot)

    assert Q is not None
    assert len(Q) == 2
    assert all(len(s) == 2 for s in Q)

    # check against the numeric model
    numeric = rtb.models.DH.Planar2()  # a1 = a2 = 1
    q0 = np.array([0.3, 0.7])
    T = numeric.fkine(q0).A

    values = {"a1": 1.0, "a2": 1.0, **_pose_values(T)}
    sols = _numeric_solutions(Q, values)
    assert len(sols) == 4

    # not every branch combination is a valid solution, since a solution for
    # one joint can be spurious once the others are chosen.  Valid ones reach
    # the target, there are two (elbow up and down), and one is the original
    valid = [qs for qs in sols if np.allclose(numeric.fkine(qs).t[:2], T[:2, 3])]
    assert len(valid) == 2
    assert any(np.allclose(_wrap(qs), q0, atol=1e-6) for qs in valid)


def _elbow_robot() -> DHRobot:
    """3-link arm with a vertical waist, as for the first 3 joints of a Puma."""
    from roboticstoolbox import RevoluteDH

    return DHRobot(
        [
            RevoluteDH(d=0.5, a=0, alpha=np.pi / 2),
            RevoluteDH(a=0.4),
            RevoluteDH(a=0.3),
        ],
        name="elbow",
    )


def test_elbow3():
    robot = _elbow_robot()
    Q = ikine_sym(robot)

    assert Q is not None
    assert len(Q) == 3

    _, values = _symbolize(robot)
    q0 = np.array([0.4, 0.6, -0.9])
    T = robot.fkine(q0).A

    sols = _numeric_solutions(Q, {**values, **_pose_values(T)})

    # every branch combination is not necessarily a valid solution (a
    # solution for one joint can be spurious), but valid ones must reach the
    # target and the original must be found
    valid = [qs for qs in sols if np.allclose(robot.fkine(qs).t, T[:3, 3], atol=1e-6)]
    assert len(valid) >= 2
    assert any(np.allclose(_wrap(qs), q0, atol=1e-6) for qs in valid)


def test_pieper_forms():
    """All forms of the equations are the same constraint T = A1 ... AN."""
    numeric = _elbow_robot()
    robot, values = _symbolize(numeric)
    A = _link_matrices(robot)
    q0 = [0.4, 0.6, -0.9]
    T = numeric.fkine(q0).A

    subs = {s: v for s, v in zip(_POSE_SYMBOLS, T[:3, :].flatten())}
    subs |= {sp.Symbol(k, real=True): v for k, v in values.items()}
    for j, qj in enumerate(q0, 1):
        sj, cj = _sin_cos_symbols(j)
        subs |= {sj: np.sin(qj), cj: np.cos(qj)}

    for nl in range(4):
        for nr in range(4 - nl):
            left, right = _pieper(A, _pose_matrix(), nl, nr)
            residual = np.array((left - right).subs(subs).evalf(), dtype=float)
            np.testing.assert_allclose(residual, 0, atol=1e-12)


def test_bad_n():
    import pytest
    import roboticstoolbox as rtb

    robot = rtb.models.DH.Planar2(symbolic=True)
    with pytest.raises(ValueError):
        ikine_sym(robot, 4)


def _dh_models_6() -> list[str]:
    """Names of the 6-axis standard DH models in ``roboticstoolbox.models.DH``."""
    import inspect

    import roboticstoolbox as rtb

    names = []
    for name, cls in inspect.getmembers(rtb.models.DH, inspect.isclass):
        if issubclass(cls, DHRobot):
            robot = cls()
            if robot.n == 6 and not any(link.mdh for link in robot.links):
                names.append(name)
    return names


# models for which a symbolic solution is expected.  The others are allowed to
# return None (not yet solved) but must not raise.  Update as the solver improves
_EXPECTED_SOLVED: set[str] = {
    "IRB140",
    "KR5",
    "Puma560",
    "Stanford",
    "UR3",
    "UR5",
    "UR10",
}


def _wrap_joints(q: np.ndarray, revolute: list[bool]) -> np.ndarray:
    """Wrap the revolute joint angles to (-pi, pi]."""
    q = np.array(q, dtype=float)
    r = np.array(revolute)
    q[r] = _wrap(q[r])
    return q


def _numeric_check(
    robot: DHRobot, Q: list[list[sp.Expr]], trials: int = 2, seed: int = 0
) -> list[int]:
    """
    Check a symbolic IK solution against forward kinematics

    For random joint configurations, compute the pose, evaluate all branches of
    the symbolic solution, and check that some of them reach the pose, and that
    the original configuration is among those that do.

    :returns: number of valid solutions for each trial
    """
    rng = np.random.default_rng(seed)
    _, values = _symbolize(robot)
    revolute = [link.isrevolute for link in robot.links]

    counts = []
    for _ in range(trials):
        q0 = np.where(
            revolute, rng.uniform(-1.5, 1.5, robot.n), rng.uniform(0.2, 0.6, robot.n)
        )
        T = robot.fkine(q0).A
        sols = _numeric_solutions(Q, {**values, **_pose_values(T)}, revolute)

        valid = [q for q in sols if np.abs(robot.fkine(q).A - T).max() < 1e-6]
        assert valid, "no valid solution"
        target = _wrap_joints(q0, revolute)
        assert any(
            np.allclose(_wrap_joints(q, revolute), target, atol=1e-5) for q in valid
        ), "original configuration not found"
        counts.append(len(valid))
    return counts


def test_dh_models():
    """
    Solve each 6-axis DH model.  Where a solution is found check its form, and
    check it numerically against forward kinematics.
    """
    import roboticstoolbox as rtb

    for name in _dh_models_6():
        robot = getattr(rtb.models.DH, name)()
        t0 = time.monotonic()
        Q = ikine_sym(robot, max_time=120 if name in _EXPECTED_SOLVED else 15)
        print(f"{name:10s} {'solved' if Q else 'None  '} {time.monotonic() - t0:.1f}s")

        if name in _EXPECTED_SOLVED:
            assert Q is not None, name
        if Q is None:
            continue

        assert len(Q) == 6
        assert all(len(s) > 0 for s in Q)
        # only prismatic joint variables ``qj`` may remain in a solution
        prismatic = {
            f"q{j}" for j, link in enumerate(robot.links, 1) if link.isprismatic
        }
        leftover = {
            s.name
            for sols in Q
            for e in sols
            for s in e.free_symbols
            if re.fullmatch(r"q\d+", s.name)
        }
        assert leftover <= prismatic, (name, leftover)

        counts = _numeric_check(robot, Q)
        print(f"{'':10s} valid solutions per trial: {counts}")


def test_generate_python():
    """The generated code runs and gives the same solutions as the symbolic ones."""
    import roboticstoolbox as rtb

    rng = np.random.default_rng(0)
    for name in ("Puma560", "Stanford", "UR5"):
        robot = getattr(rtb.models.DH, name)()
        code = generate_python(robot, "ikine")
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102

        revolute = [link.isrevolute for link in robot.links]
        q0 = np.where(revolute, rng.uniform(-1.5, 1.5, 6), rng.uniform(0.2, 0.6, 6))
        T = robot.fkine(q0).A
        solutions = namespace["ikine"](T)

        assert solutions, name
        for q in solutions:
            assert np.abs(robot.fkine(np.array(q)).A - T).max() < 1e-6
        target = _wrap_joints(q0, revolute)
        assert any(
            np.allclose(_wrap_joints(np.array(q), revolute), target, atol=1e-5)
            for q in solutions
        ), name
        print(f"{name:10s} generated {code.count(chr(10))} lines, {len(solutions)} solutions")


if __name__ == "__main__":  # pragma nocover
    import pytest

    pytest.main([__file__, "-v", "-s"])
