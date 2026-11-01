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
forward kinematics.  Jaco and Mico, which have no wrist centre, are not, but are
solved by the general 6R method of :mod:`symbolic_ik_rr`, which also chooses
between the two methods, see :func:`symbolic_ik_rr.make_solver`.

Speed: the polynomial arithmetic is done by FLINT, via ``python-flint``, if it is
installed (it is imported when first needed, and a warning says so if it is not),
which is much faster than SymPy (for the elimination step more than
100 times on the Jaco).  Without it SymPy is used and the results are the same.
Irrational constants such as ``sqrt(3)/2`` are replaced by symbols while solving
so that every equation is a polynomial with rational coefficients.

Known limitations:

- Only standard DH robots are handled, since link transforms come from
  ``Link.A``.
- Elimination needs equations that are linear in ``Sj``, ``Cj``, which is why
  the general 6R case (Jaco, Mico) fails.  Greedy elimination of pairs of
  equations saturates: the equations in the same two joints are dependent, so
  their resultant is zero.  What is needed is the elimination of many equations at
  once, which is the method of Raghavan and Roth, J. Mech. Design 115(3), 1993,
  in :mod:`symbolic_ik_rr`.
- Spurious branches are not removed: some combinations of the alternatives for
  each joint do not reach the pose.  The equations that are fully determined
  once all joints are solved could be used to prune them.
- Solutions are expressions in terms of the pose elements, the link parameters
  and ``Si``, ``Ci`` of other joints, and are not simplified.  Reducing them
  using a Groebner basis of the known relations was tried: it was slow (minutes
  for the basis, much longer to reduce a large expression) and the result was
  larger, because the normal form is fully expanded.  Common subexpression
  elimination in the generated code is what keeps them manageable.
"""

from __future__ import annotations

import enum
import re
import signal
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Iterator, Literal

import numpy as np
import sympy as sp
from sympy.printing.c import C99CodePrinter
from sympy.printing.cxx import CXX17CodePrinter
from sympy.printing.pycode import PythonCodePrinter
from sympy.printing.rust import RustCodePrinter

from roboticstoolbox.robot.DHRobot import DHRobot
from roboticstoolbox.tools.types import ArrayLike

# python-flint is optional, and makes the elimination step much faster.  It is
# imported the first time it is needed, and if it is missing a warning says what is
# lost.  ``None`` means not yet tried, ``False`` means not to be used, otherwise
# the pair ``(fmpq, fmpq_mpoly_ctx)``.
_flint: tuple | bool | None = None


def _get_flint() -> tuple | None:
    """
    The FLINT classes used for the polynomial steps, or ``None`` to use SymPy

    Warns, once, if python-flint is not installed.
    """
    global _flint
    if _flint is None:
        try:
            from flint import fmpq, fmpq_mpoly_ctx

            _flint = (fmpq, fmpq_mpoly_ctx)
        except ImportError:
            _flint = False
            warnings.warn(
                "python-flint is not installed, so the symbolic derivation uses "
                "SymPy for the polynomial steps, which gives the same results but "
                "can be 100 times slower for some arms.  Install it with "
                "'pip install python-flint'.",
                RuntimeWarning,
                stacklevel=3,
            )
    return _flint if isinstance(_flint, tuple) else None

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


def _abstract_constants(
    A: list[sp.Matrix],
) -> tuple[list[sp.Matrix], dict[sp.Symbol, sp.Expr]]:
    """
    Replace irrational constants by symbols

    :param A: link matrices
    :returns: ``(A, constants)``, where ``constants`` maps each new symbol
        ``k1``, ``k2``, ... to the constant it replaced

    A twist angle like ``alpha = pi/3`` leaves constants such as ``sqrt(3)/2`` or
    ``cos(pi/18)`` in the equations.  Replacing them with symbols makes every
    equation a polynomial with rational coefficients, which can be handled by
    fast polynomial libraries.  The symbols carry no relations between them, so
    some simplifications are missed.  Rational numbers are left alone.
    """
    constants: dict[sp.Symbol, sp.Expr] = {}
    lookup: dict[sp.Expr, sp.Symbol] = {}

    def symbol_for(e: sp.Expr) -> sp.Symbol:
        if e not in lookup:
            lookup[e] = sp.Symbol(f"k{len(lookup) + 1}", real=True)
            constants[lookup[e]] = e
        return lookup[e]

    def is_constant(e: sp.Expr) -> bool:
        return (
            (e.is_Pow or isinstance(e, (sp.sin, sp.cos)))
            and e.is_number
            and not e.is_Rational
        )

    return [m.replace(is_constant, symbol_for) for m in A], constants


# ------------------------------------------------------------------------- #
#                         equations and unknowns                            #
# ------------------------------------------------------------------------- #

_JOINT_SYMBOL = re.compile(r"([SCq])(\d+)")


@lru_cache(maxsize=None)
def _joints_in(expr: sp.Expr) -> frozenset[int]:
    """Joints (1-based) whose symbols ``Sj``, ``Cj`` or ``qj`` appear in expr."""
    out = set()
    for s in expr.free_symbols:
        m = _JOINT_SYMBOL.fullmatch(s.name)
        if m:
            out.add(int(m.group(2)))
    return frozenset(out)


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


def _linear_form_flint(eq: sp.Expr, j: int) -> tuple | None:
    """
    The revolute joint case of :func:`_linear_form` using FLINT

    :raises sympy.polys.polyerrors.PolynomialError: if ``eq`` is not a polynomial
        with rational coefficients

    ``Sj**2`` is replaced by ``1 - Cj**2`` by dividing by ``Sj**2 + Cj**2 - 1``
    using lexicographic order with ``Sj`` first, so the remainder has no term
    divisible by ``Sj**2``.
    """
    sj, cj = _sin_cos_symbols(j)
    others = sorted(eq.free_symbols - {sj, cj}, key=lambda s: s.name)
    gens = [sj, cj, *others]
    flint = _get_flint()
    assert flint is not None  # nosec B101: narrows the type for mypy
    fmpq, fmpq_mpoly_ctx = flint
    ctx = fmpq_mpoly_ctx.get(tuple(g.name for g in gens), "lex")

    poly = sp.Poly(eq, *gens, domain="QQ")
    P = ctx.from_dict({m: fmpq(int(c.p), int(c.q)) for m, c in poly.terms()})
    S, C = ctx.gens()[0], ctx.gens()[1]
    _, P = divmod(P, S**2 + C**2 - 1)

    # group by the powers of Sj and Cj, which must be 1, Cj, Sj or a constant
    parts: dict[tuple[int, int], list] = {(0, 0): [], (0, 1): [], (1, 0): []}
    for m, c in P.to_dict().items():
        if (m[0], m[1]) not in parts:
            return None
        term = sp.Rational(int(c.p), int(c.q)) * sp.Mul(
            *[g**e for g, e in zip(others, m[2:]) if e]
        )
        parts[(m[0], m[1])].append(term)
    return sp.Add(*parts[(0, 1)]), sp.Add(*parts[(1, 0)]), sp.Add(*parts[(0, 0)])


@lru_cache(maxsize=None)
def _linear_form(eq: sp.Expr, j: int, revolute: bool) -> tuple | None:
    r"""
    Coefficients of an equation that is linear in the variable of joint j

    :returns: for a revolute joint ``(a, b, k)`` such that the equation is
        :math:`a C_j + b S_j + k = 0`, for a prismatic joint ``(a, k)`` such
        that it is :math:`a q_j + k = 0`, or ``None`` if the equation has any
        other form
    """
    if revolute and _get_flint() is not None:
        try:
            return _linear_form_flint(eq, j)
        except sp.polys.polyerrors.PolynomialError:
            pass  # not a polynomial with rational coefficients, use SymPy

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


@dataclass(frozen=True)
class PolyRoots:
    r"""
    The solutions for a joint are given by the roots of a polynomial

    :param coeffs: coefficients :math:`c_i` of the polynomial, lowest power first

    The solutions are :math:`q = 2 \arctan t` for the real roots :math:`t` of
    :math:`\sum_i c_i t^i = 0`, where :math:`t = \tan(q/2)`.  If the highest
    coefficient is zero the polynomial has lost degree, which means that
    :math:`t = \infty` is a root, and :math:`q = \pi` is also a solution.  If all
    the coefficients are zero the joint is undetermined.

    There is in general no closed form for the roots of a polynomial, so the
    generated code finds them numerically.  This takes the place of the list of
    expressions that a joint otherwise has.
    """

    coeffs: tuple[sp.Expr, ...]

    @property
    def free_symbols(self) -> set[sp.Symbol]:
        return set().union(*(c.free_symbols for c in self.coeffs))

    def xreplace(self, rule: dict) -> PolyRoots:
        return PolyRoots(tuple(c.xreplace(rule) for c in self.coeffs))


def _solve_single(
    eq: sp.Expr,
    j: int,
    revolute: bool,
    samples: list[dict[str, float]] | None = None,
    polynomial: bool = False,
) -> list[sp.Expr | PolyRoots] | None:
    r"""
    Solve an equation in a single unknown joint

    :param eq: expression which equals zero
    :param j: joint number (1-based), the only unknown joint in ``eq``
    :param revolute: joint is revolute
    :param samples: numeric instances, see :func:`_samples`, used to reject an
        equation whose coefficients are all zero
    :param polynomial: give the solution as the roots of a polynomial in
        :math:`\tan(q/2)`, see :class:`PolyRoots`, rather than in closed form
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
    if polynomial:
        # A cos(q) + B sin(q) + k = 0 with cos(q) = (1 - t^2) / (1 + t^2) and
        # sin(q) = 2 t / (1 + t^2), multiplied by 1 + t^2
        return [PolyRoots((A + k, 2 * B, k - A))]
    C = -k
    if C == 0:
        return [sp.atan2(-A, B), sp.atan2(A, -B)]
    r = sp.sqrt(A**2 + B**2 - C**2)
    phi = sp.atan2(A, B)
    return [sp.atan2(C, r) - phi, sp.atan2(C, -r) - phi]


def _eliminate_flint(f1: tuple, f2: tuple) -> list[sp.Expr] | None:
    """
    The revolute joint case of :func:`_eliminate` using FLINT

    :param f1: ``(a, b, k)`` of the first equation, see :func:`_linear_form`
    :param f2: ``(a, b, k)`` of the second equation
    :raises sympy.polys.polyerrors.PolynomialError: if the coefficients are not
        polynomials with rational coefficients

    Multivariate polynomial arithmetic in FLINT is far faster than in SymPy.
    """
    coeffs = [sp.sympify(c) for c in (*f1, *f2)]
    gens = sorted(set().union(*(c.free_symbols for c in coeffs)), key=lambda s: s.name)
    flint = _get_flint()
    assert flint is not None  # nosec B101: narrows the type for mypy
    fmpq, fmpq_mpoly_ctx = flint
    ctx = fmpq_mpoly_ctx.get(tuple(g.name for g in gens), "deglex")

    def to_flint(expr: sp.Expr):
        poly = sp.Poly(expr, *gens, domain="QQ")
        return ctx.from_dict({m: fmpq(int(c.p), int(c.q)) for m, c in poly.terms()})

    a1, b1, k1, a2, b2, k2 = (to_flint(c) for c in coeffs)
    det = a1 * b2 - a2 * b1
    if det.is_zero():
        return None
    R = (-k1 * b2 + k2 * b1) ** 2 + (-a1 * k2 + a2 * k1) ** 2 - det**2
    if R.is_zero():
        return None

    # the vanishing of the determinant is a degenerate case, so divide it out
    # (as many times as it goes exactly)
    if not det.is_constant():
        while not R.is_constant():
            quotient, remainder = divmod(R, det)
            if not remainder.is_zero():
                break
            R = quotient

    if R.is_constant():
        return None
    terms = [
        sp.Rational(int(c.p), int(c.q))
        * sp.Mul(*[g**e for g, e in zip(gens, monomial) if e])
        for monomial, c in R.to_dict().items()
    ]
    return [sp.Add(*terms)]


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

    if revolute and _get_flint() is not None:
        try:
            return _eliminate_flint(f1, f2)
        except sp.polys.polyerrors.PolynomialError:
            pass  # not a polynomial with rational coefficients, use SymPy

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


def _clean(expr: sp.Expr | PolyRoots, joints: set[int]) -> sp.Expr | PolyRoots:
    """Tidy a solution: remove powers of ``Sj`` using ``Sj**2 = 1 - Cj**2``."""
    if isinstance(expr, PolyRoots):
        return PolyRoots(tuple(_clean(c, joints) for c in expr.coeffs))
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
    assert seconds is not None  # nosec B101: narrows the type for mypy
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
    polynomial: bool = False,
) -> list[list[sp.Expr | PolyRoots]] | None:
    r"""
    Symbolic inverse kinematics

    :param robot: DH robot model
    :param n: number of degrees of freedom to solve, one of 2, 3 or 6.  Must
        equal the number of joints, defaults to that.
    :param verbose: print diagnostic information
    :param max_time: give up after this many seconds (Unix, main thread only),
        ``None`` for no limit
    :param mixed: include the mixed left/right forms of Pieper's equations
    :param polynomial: give every revolute joint as the roots of a polynomial,
        see :class:`PolyRoots`, rather than in closed form.  For testing.
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
            return _ikine_sym(robot, n, verbose, mixed, polynomial)[0]
    except _Timeout:
        if verbose:
            print("timed out")
        return None



# limits on the search for eliminations in tier 2
_MAX_UNKNOWNS = 4  # only consider equations with at most this many unknown joints
_PER_GROUP = 4  # cheapest equations to use from each group of unknown joints
_MAX_NEW = 20  # stop after finding this many new equations, and reconsider


def _ikine_sym(
    robot: DHRobot, n: int, verbose: bool, mixed: bool, polynomial: bool = False
) -> tuple[list[list[sp.Expr | PolyRoots]] | None, list[sp.Expr]]:
    """Implementation of :func:`ikine_sym`, also returns the pool of equations."""
    robot, values = _symbolize(robot)
    revolute = {j: link.isrevolute for j, link in enumerate(robot.links, 1)}
    A, constants = _abstract_constants(_link_matrices(robot))
    values = values | {k.name: float(v) for k, v in constants.items()}
    samples = _samples(A, list(revolute.values()), values)
    pool = _equation_pool(A, n, mixed)
    seen = set(pool) | {sp.expand(-e) for e in pool}
    if verbose:
        print(f"{len(pool)} equations")

    Q: dict[int, list[sp.Expr]] = {}
    failed: set[tuple[sp.Expr, int]] = set()  # equation can't be solved for j
    tried: set[tuple[sp.Expr, sp.Expr, int]] = set()  # eliminations done

    def unknown(e: sp.Expr) -> set[int]:
        return set(_joints_in(e) - Q.keys())

    while len(Q) < n:
        # tier 1: an equation with a single unknown joint
        singles = sorted((e for e in pool if len(unknown(e)) == 1), key=sp.count_ops)
        for e in singles:
            (j,) = unknown(e)
            if (e, j) in failed:
                continue
            sols = _solve_single(e, j, revolute[j], samples, polynomial)
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
                    for e in sorted(g, key=lambda e: cost[e])[:_PER_GROUP]
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
                return None, pool
            if verbose:
                print(f"eliminated, {len(new)} new equations")
            pool.extend(new)

    done = set(Q)
    # put the constants back
    result = [
        [_clean(s, done).xreplace(constants) for s in Q[j]] for j in range(1, n + 1)
    ]
    return result, [e.xreplace(constants) for e in pool]


# ------------------------------------------------------------------------- #
#                                 results                                   #
# ------------------------------------------------------------------------- #


class IKStatus(enum.Enum):
    """Outcome of an inverse kinematics solution, as reported by generated code."""

    OK = 0  #: solutions were found
    SINGULAR = 1  #: one or more branches were dropped, a joint was undetermined
    UNREACHABLE = 2  #: no solution, the pose is outside the workspace


@dataclass
class IKSolutions:
    """
    All the inverse kinematic solutions for a pose

    :param q: joint configurations, one per row, shape (m, n)
    :param status: why there are, or are not, solutions
    :param revolute: whether each joint is revolute, shape (n,).  The difference
        between two angles is wrapped to (-pi, pi].

    ``status`` distinguishes a pose that cannot be reached from one where a
    singularity caused solutions to be dropped, which an empty ``q`` does not.
    """

    q: np.ndarray
    status: IKStatus
    revolute: np.ndarray

    def __post_init__(self) -> None:
        self.revolute = np.asarray(self.revolute, dtype=bool)
        self.q = np.asarray(self.q, dtype=float).reshape(-1, len(self.revolute))

    @classmethod
    def from_generated(cls, result, revolute: ArrayLike) -> IKSolutions:
        """
        Wrap the result of generated Python code

        :param result: what the generated function returned
        :param revolute: whether each joint is revolute
        """
        return cls(
            np.array(result.solutions), IKStatus[result.status.name], np.array(revolute)
        )

    def __len__(self) -> int:
        return len(self.q)

    def distance(self, q0, weights=None) -> np.ndarray:
        """
        Weighted distance of each solution from a configuration

        :param q0: configuration to compare with, shape (n,)
        :param weights: weight for each joint, default all 1.  The difference for
            each joint is multiplied by its weight, so a weight of zero ignores
            that joint.
        :returns: the distances, shape (m,)

        The distance is :math:`\\sqrt{\\sum_j (w_j \\Delta q_j)^2}`, where the
        difference :math:`\\Delta q_j` of a revolute joint is wrapped to
        :math:`(-\\pi, \\pi]`.
        """
        n = len(self.revolute)
        q0 = np.asarray(q0, dtype=float).reshape(n)
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float).reshape(n)
        d = self.q - q0
        d[:, self.revolute] = _wrap(d[:, self.revolute])
        return np.sqrt(((w * d) ** 2).sum(axis=1))

    def nearest(self, q0, weights=None) -> np.ndarray | None:
        """
        The solution nearest to a configuration

        :param q0: configuration to compare with, shape (n,)
        :param weights: weight for each joint, default all 1, see :meth:`distance`
        :returns: the nearest solution, shape (n,), or None if there are none
        """
        if len(self) == 0:
            return None
        return self.q[np.argmin(self.distance(q0, weights))]


# ------------------------------------------------------------------------- #
#                            code generation                                #
# ------------------------------------------------------------------------- #

# an atan2 whose two arguments are both smaller than this is a singularity
SINGULARITY_TOL = 1e-9


_REPOSITORY = "https://github.com/petercorke/robotics-toolbox-python"


def _dh_info(robot: DHRobot) -> dict:
    """
    The name and standard DH parameters of a robot, for the header of generated code

    The ``table`` is the first block of ``str(robot)``: the description and the DH
    table, with any offsets and flips shown in the theta column.  It is rounded
    for display, so the exact parameters are also kept.
    """
    table = re.sub(r"\x1b\[[0-9;]*m", "", str(robot)).split("\n\n")[0]
    return {
        "name": robot.name,
        "table": [line.rstrip() for line in table.splitlines()],
        "alpha": [float(link.alpha) for link in robot.links],
        "a": [float(link.a) for link in robot.links],
        "d": [float(link.d) for link in robot.links],
        "theta": [float(link.theta) for link in robot.links],
        "offset": [float(link.offset) for link in robot.links],
        "flip": [bool(link.isflip) for link in robot.links],
        "prismatic": [bool(link.isprismatic) for link in robot.links],
    }


def _provenance(info: dict, method: str) -> list[str]:
    """
    Lines of text for the header of generated code

    :param info: from :func:`_dh_info`
    :param method: description of how the solution was found
    :returns: lines without comment markers, which must not contain ``*/`` or
        three double quotes
    """
    import roboticstoolbox as rtb

    now = datetime.now().astimezone()
    lines = [
        "Generated by Robotics Toolbox for Python by Peter Corke",
        _REPOSITORY,
        f"Robotics Toolbox for Python {rtb.__version__}, MIT licence",
        "",
        f"Generated {now.strftime('%Y-%m-%d %H:%M:%S %Z')} (UTC{now.strftime('%z')})",
        f"Method: {method}",
        "",
        "The robot, as printed by Robotics Toolbox.  Standard DH parameters, with the",
        "joint angle shifted by any offset shown and negated if shown as -q.  The",
        "joint limits are not used.  This file is UTF-8.",
        "",
        *info["table"],
        "",
        "The table is rounded for display.  The parameters, exactly.  For a revolute",
        "joint d is fixed and theta is the variable, for a prismatic joint theta is",
        "fixed and d is the variable, and the value of the variable is not used:",
        f"  alpha  = {info['alpha']}",
        f"  a      = {info['a']}",
        f"  d      = {info['d']}",
        f"  theta  = {info['theta']}",
        f"  offset = {info['offset']}",
        f"  flip   = {info['flip']}",
    ]
    return lines


def _comment(lines: list[str], language: str) -> list[str]:
    """Lines of text as a comment, or a module docstring for Python."""
    if language == "python":
        return ['"""', *lines, '"""']
    if language == "c":
        return ["/*", *[(" * " + line).rstrip() for line in lines], " */"]
    if language == "c++":
        return [("// " + line).rstrip() for line in lines]
    if language == "rust":
        return [("//! " + line).rstrip() for line in lines]
    raise ValueError(language)


@dataclass
class _Step:
    """Solving for one joint: shared subexpressions, then the alternatives."""

    joint: int  # 1-based
    revolute: bool
    assigns: list[tuple[sp.Symbol, sp.Expr]]
    alts: list[sp.Expr]  # or, if poly, the coefficients of the polynomial
    checks: list[sp.Expr]  # equations that are zero for a valid choice of alts
    poly: bool = False  # the solutions are the roots of a polynomial, see PolyRoots


@dataclass
class _Plan:
    """Language independent description of the generated solver."""

    n: int
    pose: list[str]  # names of the pose elements, nx ... tz
    steps: list[_Step]  # in the order the joints are solved
    fk_assigns: list[tuple[sp.Symbol, sp.Expr]]  # forward kinematics check
    fk_exprs: list[sp.Expr]
    capacity: int  # upper bound on the number of solutions
    verify: bool  # check each solution against the forward kinematics
    info: dict  # robot name and DH parameters, see _dh_info
    method: str  # how the solution was found


def _pruning_checks(
    Q: list[list[sp.Expr]],
    pool: list[sp.Expr],
    order: list[int],
    A: list[sp.Matrix],
    revolute: list[bool],
    values: dict[str, float],
    per_step: int = 3,
) -> dict[int, list[sp.Expr]]:
    """
    Choose equations that rule out wrong alternatives as soon as possible

    :param Q: symbolic solution
    :param pool: all the equations
    :param order: joints (0-based) in the order they are solved
    :param per_step: most equations to use at each step
    :returns: for each joint (0-based) that has more than one alternative, the
        equations to check once it is chosen

    Once a joint is solved there are equations whose unknowns are all solved, so
    they are zero for a valid choice of the alternatives so far.  They are of use
    only if they distinguish: the correct alternative satisfies them and the
    wrong one does not, since some equations are satisfied by both roots.  This
    is tested numerically, on random configurations, and equations are chosen
    greedily to cover as many of the wrong alternatives as possible.
    """
    samples = _samples(A, revolute, values, count=6, seed=1)
    solved: set[int] = set()
    out: dict[int, list[sp.Expr]] = {}

    for j in order:
        solved.add(j + 1)
        if len(Q[j]) < 2 or not revolute[j]:
            continue
        cands = sorted(
            (e for e in pool if j + 1 in _joints_in(e) and _joints_in(e) <= solved),
            key=sp.count_ops,
        )[:40]
        alt_funcs = [_compile(e) for e in Q[j]]
        cand_funcs = [_compile(e) for e in cands]

        # which (sample, wrong alternative) pairs each candidate rules out
        covers: list[set[tuple[int, int]]] = [set() for _ in cands]
        for si, vals in enumerate(samples):
            true_q = np.arctan2(vals[f"S{j + 1}"], vals[f"C{j + 1}"])
            for ai, (args, f) in enumerate(alt_funcs):
                qa = float(f(*[vals[a.name] for a in args]))
                if np.isnan(qa) or abs(_wrap(np.array(qa - true_q))) < 1e-6:
                    continue  # no solution, or the correct alternative
                trial = {**vals, f"S{j + 1}": np.sin(qa), f"C{j + 1}": np.cos(qa)}
                for ci, (cargs, cf) in enumerate(cand_funcs):
                    if abs(float(cf(*[trial[a.name] for a in cargs]))) > 1e-4:
                        covers[ci].add((si, ai))

        chosen: list[sp.Expr] = []
        remaining = set().union(*covers) if covers else set()
        while remaining and len(chosen) < per_step:
            best = max(range(len(cands)), key=lambda c: (len(covers[c] & remaining), -c))
            chosen.append(cands[best])
            remaining -= covers[best]
        if chosen:
            out[j] = chosen
    return out


def _make_plan(
    robot: DHRobot,
    verify: bool = True,
    polynomial: bool = False,
    max_time: float | None = None,
) -> _Plan:
    """
    Solve a robot symbolically and organize the solution for code generation

    :param robot: DH robot model with numeric parameters
    :param verify: check each solution against the forward kinematics
    :param polynomial: give the revolute joints as the roots of polynomials
    :param max_time: give up searching for a symbolic solution after this many
        seconds (Unix, main thread only)
    :raises ValueError: if no solution is found, or the model has symbolic
        parameters
    """
    if robot.n not in _SPECIFIED:
        raise ValueError("Can only solve for 2, 3, 6 DOF")
    try:
        with _time_limit(max_time):
            Q, pool = _ikine_sym(robot, robot.n, False, True, polynomial)
    except _Timeout:
        Q = None
    if Q is None:
        raise ValueError("no symbolic solution for this robot")

    symrobot, values = _symbolize(robot)
    n = robot.n
    revolute = [link.isrevolute for link in symrobot.links]
    pose = [s.name for s in _POSE_SYMBOLS]

    def numeric(e: sp.Expr) -> sp.Expr:
        sub = {s: sp.Float(values[s.name]) for s in e.free_symbols if s.name in values}
        e = e.xreplace(sub)
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
    fk_assigns, fk_exprs = sp.cse(fk, symbols=sp.numbered_symbols("y"))

    order = _solve_order(Q)
    checks = _pruning_checks(
        Q, pool, order, _link_matrices(symrobot), revolute, values
    )

    steps = []
    for j in order:
        poly = isinstance(Q[j][0], PolyRoots)
        exprs = [numeric(e) for e in (Q[j][0].coeffs if poly else Q[j])]
        assigns, exprs = sp.cse(exprs, symbols=sp.numbered_symbols(f"x{j + 1}_"))
        steps.append(
            _Step(
                j + 1,
                revolute[j],
                assigns,
                exprs,
                [numeric(e) for e in checks.get(j, [])],
                poly,
            )
        )

    # a polynomial of degree d has at most d roots, and q = pi is only a
    # solution if the degree is less than expected
    capacity = int(np.prod([len(s.alts) for s in steps]))
    method = "closed form, derived symbolically (symbolic_ik)"
    if polynomial:
        method += ", joints as roots of polynomials"
    return _Plan(
        n, pose, steps, fk_assigns, fk_exprs, capacity, verify, _dh_info(robot), method
    )


class _Emitter:
    """
    Write a plan as source code

    The structure is the same for every language: a nested loop over the
    alternative solutions for each joint, in the order they were solved.  A
    subclass provides the syntax.

    Singularities are detected by ``atan2c``, a checked ``atan2``, which sets a
    flag and returns NaN if both its arguments are nearly zero.  A NaN joint
    value, which also results from the square root of a negative number, means
    that branch is dropped.  At the end the flag distinguishes a singularity
    from a pose that can't be reached.
    """

    printer_class: type

    def __init__(self, plan: _Plan, name: str):
        self.plan = plan
        self.name = name
        self.lines: list[str] = []
        self.depth = 0
        self.printer = self.printer_class(
            {"user_functions": {"safe_sqrt": "safe_sqrt", "atan2c": "atan2c"}}
        )

    # --- helpers ---

    def expr(self, e: sp.Expr) -> str:
        """Source for an expression, with checked square root and atan2."""
        e = e.replace(
            lambda x: x.is_Pow and x.exp == sp.S.Half,
            lambda x: sp.Function("safe_sqrt")(x.base),
        )
        e = e.replace(
            lambda x: isinstance(x, sp.atan2),
            lambda x: sp.Function("atan2c")(sp.Symbol("ctx"), *x.args),
        )
        return self.printer.doprint(e)

    def put(self, text: str = "") -> None:
        self.lines.append(("    " * self.depth + text) if text else "")

    def qout(self, j: int) -> str:
        """Value of joint j (1-based) for output: revolute angles are wrapped."""
        revolute = {st.joint: st.revolute for st in self.plan.steps}[j]
        return self.atan2(f"S{j}", f"C{j}") if revolute else f"q{j}"

    def qs(self) -> str:
        return ", ".join(self.qout(j) for j in range(1, self.plan.n + 1))

    # --- the structure, common to all languages ---

    language = ""  # set by each subclass, for the comment syntax of the banner

    def banner(self) -> None:
        """Replace the placeholder at the top with the provenance of the file."""
        i = self.lines.index("@@BANNER@@")
        text = _provenance(self.plan.info, self.plan.method)
        self.lines[i : i + 1] = _comment(text, self.language)

    def generate(self) -> str:
        self.header()
        self.banner()
        self.step(0)
        self.footer()
        return "\n".join(self.lines) + "\n"

    def step(self, index: int) -> None:
        plan = self.plan
        if index == len(plan.steps):
            if plan.verify:
                for sym, e in plan.fk_assigns:
                    self.decl(str(sym), self.expr(e))
                self.error_start()
                for e, p in zip(plan.fk_exprs, plan.pose):
                    self.error_update(self.expr(e), p)
            self.push(plan.verify)
            return

        st = plan.steps[index]
        for sym, e in st.assigns:
            self.decl(str(sym), self.expr(e))
        k, q = f"k{st.joint}", f"q{st.joint}"
        if st.poly:
            self.poly_open(st.joint, [self.expr(a) for a in st.alts])
        else:
            self.loop_open(k, len(st.alts))
            self.select(q, k, [self.expr(a) for a in st.alts])
            self.skip_if_nan(q)
        if st.revolute:
            self.decl(f"S{st.joint}", self.sin(q))
            self.decl(f"C{st.joint}", self.cos(q))
        for c in st.checks:
            self.check(self.expr(c))
        self.step(index + 1)
        self.loop_close()

    def poly_helper(self, code: str, blank_lines: int) -> None:
        """
        Replace the placeholder line with the root finder, but only if some joint
        is solved by a polynomial, otherwise remove it and the blank lines after it.
        """
        i = self.lines.index("@@POLY@@")
        if any(step.poly for step in self.plan.steps):
            self.lines[i : i + 1] = code.splitlines()
        else:
            del self.lines[i : i + 1 + blank_lines]

    # --- syntax: subclasses implement these ---

    def header(self) -> None: ...
    def footer(self) -> None: ...
    def decl(self, name: str, value: str) -> None: ...
    def loop_open(self, k: str, count: int) -> None: ...
    def poly_open(self, joint: int, coeffs: list[str]) -> None: ...
    def loop_close(self) -> None: ...
    def select(self, q: str, k: str, alts: list[str]) -> None: ...
    def skip_if_nan(self, q: str) -> None: ...
    def sin(self, q: str) -> str:
        raise NotImplementedError

    def cos(self, q: str) -> str:
        raise NotImplementedError

    def atan2(self, y: str, x: str) -> str:
        raise NotImplementedError

    def error_start(self) -> None: ...
    def error_update(self, e: str, pose: str) -> None: ...
    def check(self, e: str) -> None: ...
    def push(self, guarded: bool) -> None: ...


_PY_POLY = """\
POLY_TOL = 1e-9


def half_angle_roots(ctx: _Context, coeffs: list[float]) -> list[float]:
    \"\"\"
    Joint angles q for which sum(coeffs[i] * t**i) = 0, where t = tan(q / 2)

    Only real roots are returned.  If the highest coefficients are zero the
    polynomial has lost degree, and q = pi is also a solution.  If all the
    coefficients are zero the joint is undetermined, which is a singularity.
    \"\"\"
    scale = max(abs(c) for c in coeffs)
    if math.isnan(scale):
        return []
    if scale < POLY_TOL:
        ctx.singular = True
        return []
    n = len(coeffs) - 1
    while n > 0 and abs(coeffs[n]) <= 1e-12 * scale:
        n -= 1
    angles = [math.pi] if n < len(coeffs) - 1 else []
    if n == 0:
        return angles

    # Aberth-Ehrlich iteration for all the roots of the monic polynomial
    # (O. Aberth, Math. Comp. 27(122):339-344, 1973; L. W. Ehrlich, Commun. ACM 10(2):107-108, 1967)
    a = [c / coeffs[n] for c in coeffs[: n + 1]]
    radius = abs(a[0]) ** (1.0 / n) if a[0] != 0 else 1.0
    z = [radius * cmath.exp(1j * (2 * math.pi * k / n + 0.4)) for k in range(n)]
    for _ in range(100):
        converged = True
        for i in range(n):
            p = dp = 0j
            for c in reversed(a):
                dp = dp * z[i] + p
                p = p * z[i] + c
            if p == 0:
                continue
            ratio = p / dp if dp != 0 else p
            s = sum(1 / (z[i] - z[j]) for j in range(n) if j != i and z[i] != z[j])
            denominator = 1 - ratio * s
            w = ratio / denominator if denominator != 0 else ratio
            z[i] -= w
            if abs(w) > 1e-14 * (1 + abs(z[i])):
                converged = False
        if converged:
            break

    for root in z:
        if abs(root.imag) <= 1e-6 * (1 + abs(root.real)):
            t = root.real
            for _ in range(3):  # polish
                f = df = 0.0
                for c in reversed(a):
                    df = df * t + f
                    f = f * t + c
                if df == 0:
                    break
                t -= f / df
            q = 2 * math.atan(t)
            if all(abs(q - other) > 1e-9 for other in angles):
                angles.append(q)
    return angles"""

_C_POLY = """\
#include <complex.h>

#define IKINE_MAX_ANGLES 32
#define IKINE_POLY_TOL 1e-9

/*
 * Joint angles q for which sum(coeffs[i] * t^i) = 0, where t = tan(q / 2), written
 * to angles[].  Only real roots are returned.  If the highest coefficients are
 * zero the polynomial has lost degree, and q = pi is also a solution.  If all the
 * coefficients are zero the joint is undetermined, which is a singularity.
 * Returns the number of angles.
 */
static int half_angle_roots(ik_ctx *ctx, const double *coeffs, int ncoeffs, double *angles) {
    double scale = 0.0;
    for (int i = 0; i < ncoeffs; ++i) {
        if (isnan(coeffs[i])) return 0;
        if (fabs(coeffs[i]) > scale) scale = fabs(coeffs[i]);
    }
    if (scale < IKINE_POLY_TOL) {
        ctx->singular = 1;
        return 0;
    }
    int n = ncoeffs - 1, count = 0;
    while (n > 0 && fabs(coeffs[n]) <= 1e-12 * scale) --n;
    if (n < ncoeffs - 1) angles[count++] = M_PI;
    if (n == 0) return count;

    /* Aberth-Ehrlich iteration for all the roots of the monic polynomial
     * (O. Aberth, Math. Comp. 27(122):339-344, 1973; L. W. Ehrlich, Commun. ACM 10(2):107-108, 1967) */
    double a[IKINE_MAX_ANGLES];
    double complex z[IKINE_MAX_ANGLES];
    for (int i = 0; i <= n; ++i) a[i] = coeffs[i] / coeffs[n];
    double radius = a[0] != 0.0 ? pow(fabs(a[0]), 1.0 / n) : 1.0;
    for (int k = 0; k < n; ++k) z[k] = radius * cexp(I * (2.0 * M_PI * k / n + 0.4));
    for (int iter = 0; iter < 100; ++iter) {
        int converged = 1;
        for (int i = 0; i < n; ++i) {
            double complex p = 0.0, dp = 0.0, s = 0.0;
            for (int k = n; k >= 0; --k) {
                dp = dp * z[i] + p;
                p = p * z[i] + a[k];
            }
            if (p == 0.0) continue;
            double complex ratio = dp != 0.0 ? p / dp : p;
            for (int j = 0; j < n; ++j) {
                if (j != i && z[i] != z[j]) s += 1.0 / (z[i] - z[j]);
            }
            double complex denominator = 1.0 - ratio * s;
            double complex w = denominator != 0.0 ? ratio / denominator : ratio;
            z[i] -= w;
            if (cabs(w) > 1e-14 * (1.0 + cabs(z[i]))) converged = 0;
        }
        if (converged) break;
    }

    for (int i = 0; i < n; ++i) {
        if (fabs(cimag(z[i])) > 1e-6 * (1.0 + fabs(creal(z[i])))) continue;
        double t = creal(z[i]);
        for (int iter = 0; iter < 3; ++iter) { /* polish */
            double f = 0.0, df = 0.0;
            for (int k = n; k >= 0; --k) {
                df = df * t + f;
                f = f * t + a[k];
            }
            if (df == 0.0) break;
            t -= f / df;
        }
        double q = 2.0 * atan(t);
        int duplicate = 0;
        for (int k = 0; k < count; ++k) {
            if (fabs(q - angles[k]) <= 1e-9) duplicate = 1;
        }
        if (!duplicate && count < IKINE_MAX_ANGLES) angles[count++] = q;
    }
    return count;
}"""

_CXX_POLY = """\
#include <complex>

constexpr double IKINE_POLY_TOL = 1e-9;

// Joint angles q for which sum(coeffs[i] * t^i) = 0, where t = tan(q / 2).  Only
// real roots are returned.  If the highest coefficients are zero the polynomial
// has lost degree, and q = pi is also a solution.  If all the coefficients are
// zero the joint is undetermined, which is a singularity.
inline std::vector<double> half_angle_roots(IkContext& ctx, const std::vector<double>& coeffs) {
    std::vector<double> angles;
    double scale = 0.0;
    for (double c : coeffs) {
        if (std::isnan(c)) return angles;
        scale = std::fmax(scale, std::fabs(c));
    }
    if (scale < IKINE_POLY_TOL) {
        ctx.singular = true;
        return angles;
    }
    int n = static_cast<int>(coeffs.size()) - 1;
    while (n > 0 && std::fabs(coeffs[n]) <= 1e-12 * scale) --n;
    if (n < static_cast<int>(coeffs.size()) - 1) angles.push_back(M_PI);
    if (n == 0) return angles;

    // Aberth-Ehrlich iteration for all the roots of the monic polynomial
    // (O. Aberth, Math. Comp. 27(122):339-344, 1973; L. W. Ehrlich, Commun. ACM 10(2):107-108, 1967)
    using cd = std::complex<double>;
    std::vector<double> a(n + 1);
    for (int i = 0; i <= n; ++i) a[i] = coeffs[i] / coeffs[n];
    std::vector<cd> z(n);
    const double radius = a[0] != 0.0 ? std::pow(std::fabs(a[0]), 1.0 / n) : 1.0;
    for (int k = 0; k < n; ++k) z[k] = std::polar(radius, 2.0 * M_PI * k / n + 0.4);
    for (int iter = 0; iter < 100; ++iter) {
        bool converged = true;
        for (int i = 0; i < n; ++i) {
            cd p = 0.0, dp = 0.0, s = 0.0;
            for (int k = n; k >= 0; --k) {
                dp = dp * z[i] + p;
                p = p * z[i] + a[k];
            }
            if (p == 0.0) continue;
            const cd ratio = dp != 0.0 ? p / dp : p;
            for (int j = 0; j < n; ++j) {
                if (j != i && z[i] != z[j]) s += 1.0 / (z[i] - z[j]);
            }
            const cd denominator = 1.0 - ratio * s;
            const cd w = denominator != 0.0 ? ratio / denominator : ratio;
            z[i] -= w;
            if (std::abs(w) > 1e-14 * (1.0 + std::abs(z[i]))) converged = false;
        }
        if (converged) break;
    }

    for (int i = 0; i < n; ++i) {
        if (std::fabs(z[i].imag()) > 1e-6 * (1.0 + std::fabs(z[i].real()))) continue;
        double t = z[i].real();
        for (int iter = 0; iter < 3; ++iter) {  // polish
            double f = 0.0, df = 0.0;
            for (int k = n; k >= 0; --k) {
                df = df * t + f;
                f = f * t + a[k];
            }
            if (df == 0.0) break;
            t -= f / df;
        }
        const double q = 2.0 * std::atan(t);
        bool duplicate = false;
        for (double other : angles) duplicate = duplicate || std::fabs(q - other) <= 1e-9;
        if (!duplicate) angles.push_back(q);
    }
    return angles;
}"""


class _PythonEmitter(_Emitter):
    printer_class = PythonCodePrinter
    language = "python"

    def header(self) -> None:
        n = self.plan.n
        p = self.plan.pose
        for line in f'''@@BANNER@@

import cmath
import enum
import math
from typing import NamedTuple

SINGULARITY_TOL = {SINGULARITY_TOL!r}


class IKStatus(enum.Enum):
    OK = 0  # solutions found
    SINGULAR = 1  # one or more branches were dropped due to a singularity
    UNREACHABLE = 2  # no solution, the pose is outside the workspace


class IKResult(NamedTuple):
    solutions: list[tuple[float, ...]]
    status: IKStatus


class _Context:
    singular = False


def safe_sqrt(x: float) -> float:
    # round-off can make a zero argument slightly negative
    return math.nan if x < -1e-9 else math.sqrt(max(x, 0.0))


def atan2c(ctx: _Context, y: float, x: float) -> float:
    if math.isnan(x) or math.isnan(y):
        return math.nan
    if math.hypot(y, x) < SINGULARITY_TOL:
        ctx.singular = True  # the joint is undetermined
        return math.nan
    return math.atan2(y, x)


@@POLY@@


def {self.name}(T, tol: float = 1e-6) -> IKResult:
    """
    Joint configurations (q1..q{n}) for which the end-effector pose is T

    T is a 4x4 array-like.  The status distinguishes a pose that is outside the
    workspace from one where a singularity caused solutions to be dropped.
    """
    ctx = _Context()
    {", ".join(p[0:4])} = T[0][0], T[0][1], T[0][2], T[0][3]
    {", ".join(p[4:8])} = T[1][0], T[1][1], T[1][2], T[1][3]
    {", ".join(p[8:12])} = T[2][0], T[2][1], T[2][2], T[2][3]
    solutions: list[tuple[float, ...]] = []'''.splitlines():
            self.lines.append(line)
        self.poly_helper(_PY_POLY, 2)
        self.depth = 1

    def footer(self) -> None:
        for line in [
            "if ctx.singular:",
            "    status = IKStatus.SINGULAR",
            "elif solutions:",
            "    status = IKStatus.OK",
            "else:",
            "    status = IKStatus.UNREACHABLE",
            "return IKResult(solutions, status)",
        ]:
            self.put(line)

    def decl(self, name: str, value: str) -> None:
        self.put(f"{name} = {value}")

    def loop_open(self, k: str, count: int) -> None:
        self.put(f"for {k} in range({count}):")
        self.depth += 1

    def poly_open(self, joint: int, coeffs: list[str]) -> None:
        self.put(f"for q{joint} in half_angle_roots(ctx, [{', '.join(coeffs)}]):")
        self.depth += 1

    def loop_close(self) -> None:
        self.depth -= 1

    def select(self, q: str, k: str, alts: list[str]) -> None:
        s = alts[-1]
        for i in range(len(alts) - 2, -1, -1):
            s = f"({alts[i]}) if {k} == {i} else ({s})"
        self.put(f"{q} = {s}")

    def skip_if_nan(self, q: str) -> None:
        self.put(f"if {q} != {q}:  # NaN, this branch has no solution")
        self.put("    continue")

    def sin(self, q: str) -> str:
        return f"math.sin({q})"

    def cos(self, q: str) -> str:
        return f"math.cos({q})"

    def atan2(self, y: str, x: str) -> str:
        return f"math.atan2({y}, {x})"

    def error_start(self) -> None:
        self.put("err = 0.0")

    def error_update(self, e: str, pose: str) -> None:
        self.put(f"err = max(err, abs({e} - {pose}))")

    def check(self, e: str) -> None:
        self.put(f"if abs({e}) > tol:  # a wrong alternative")
        self.put("    continue")

    def push(self, guarded: bool) -> None:
        if guarded:
            self.put("if err < tol:")
            self.put(f"    solutions.append(({self.qs()},))")
        else:
            self.put(f"solutions.append(({self.qs()},))")


class _CEmitter(_Emitter):
    printer_class = C99CodePrinter
    language = "c"
    math_prefix = ""  # C++ uses std::

    def pose_lines(self) -> list[str]:
        p = self.plan.pose
        return [
            f"const double {p[4 * r + c]} = T[{4 * r + c}];"
            for r in range(3)
            for c in range(4)
        ]

    def header(self) -> None:
        n, cap = self.plan.n, self.plan.capacity
        for line in f"""@@BANNER@@
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IKINE_DOF {n}
#define IKINE_MAX_SOLUTIONS {cap}
#define IKINE_SINGULARITY_TOL {SINGULARITY_TOL!r}

typedef enum {{
    IK_OK = 0,          /* solutions found */
    IK_SINGULAR = 1,    /* one or more branches were dropped due to a singularity */
    IK_UNREACHABLE = 2  /* no solution, the pose is outside the workspace */
}} ik_status;

typedef struct {{
    int singular;
}} ik_ctx;

static double safe_sqrt(double x) {{
    /* round-off can make a zero argument slightly negative */
    return x < -1e-9 ? NAN : sqrt(x > 0.0 ? x : 0.0);
}}

static double atan2c(ik_ctx *ctx, double y, double x) {{
    if (isnan(x) || isnan(y)) return NAN;
    if (hypot(y, x) < IKINE_SINGULARITY_TOL) {{
        ctx->singular = 1;  /* the joint is undetermined */
        return NAN;
    }}
    return atan2(y, x);
}}

@@POLY@@

/*
 * T is the 4x4 end-effector pose, row-major.  Solutions are written to
 * solutions[0..*count-1], each IKINE_DOF joint values.
 */
ik_status {self.name}(const double T[16], double tol,
                  double solutions[IKINE_MAX_SOLUTIONS][IKINE_DOF], int *count) {{
    ik_ctx ctx_ = {{0}};
    ik_ctx *ctx = &ctx_;""".splitlines():
            self.lines.append(line)
        self.poly_helper(_C_POLY, 1)
        self.depth = 1
        for line in self.pose_lines():
            self.put(line)
        self.put("*count = 0;")

    def footer(self) -> None:
        self.put("return ctx->singular ? IK_SINGULAR : (*count > 0 ? IK_OK : IK_UNREACHABLE);")
        self.depth = 0
        self.put("}")

    def decl(self, name: str, value: str) -> None:
        self.put(f"const double {name} = {value};")

    def loop_open(self, k: str, count: int) -> None:
        self.put(f"for (int {k} = 0; {k} < {count}; ++{k}) {{")
        self.depth += 1

    def poly_open(self, joint: int, coeffs: list[str]) -> None:
        n = len(coeffs)
        self.put(f"const double coeffs{joint}[{n}] = {{{', '.join(coeffs)}}};")
        self.put(f"double angles{joint}[IKINE_MAX_ANGLES];")
        self.put(
            f"const int n{joint} = half_angle_roots(ctx, coeffs{joint}, {n}, angles{joint});"
        )
        self.put(f"for (int k{joint} = 0; k{joint} < n{joint}; ++k{joint}) {{")
        self.depth += 1
        self.put(f"const double q{joint} = angles{joint}[k{joint}];")

    def loop_close(self) -> None:
        self.depth -= 1
        self.put("}")

    def select(self, q: str, k: str, alts: list[str]) -> None:
        s = alts[-1]
        for i in range(len(alts) - 2, -1, -1):
            s = f"({k} == {i}) ? ({alts[i]}) : ({s})"
        self.put(f"const double {q} = {s};")

    def skip_if_nan(self, q: str) -> None:
        self.put(f"if ({self.math_prefix}isnan({q})) continue;")

    def sin(self, q: str) -> str:
        return f"{self.math_prefix}sin({q})"

    def cos(self, q: str) -> str:
        return f"{self.math_prefix}cos({q})"

    def atan2(self, y: str, x: str) -> str:
        return f"{self.math_prefix}atan2({y}, {x})"

    def error_start(self) -> None:
        self.put("double err = 0.0;")

    def error_update(self, e: str, pose: str) -> None:
        m = self.math_prefix
        self.put(f"err = {m}fmax(err, {m}fabs({e} - {pose}));")

    def check(self, e: str) -> None:
        self.put(f"if ({self.math_prefix}fabs({e}) > tol) continue;  /* a wrong alternative */")

    def push(self, guarded: bool) -> None:
        self.put(
            "if (err < tol && *count < IKINE_MAX_SOLUTIONS) {"
            if guarded
            else "if (*count < IKINE_MAX_SOLUTIONS) {"
        )
        for j in range(1, self.plan.n + 1):
            self.put(f"    solutions[*count][{j - 1}] = {self.qout(j)};")
        self.put("    ++*count;")
        self.put("}")


class _CxxEmitter(_CEmitter):
    printer_class = CXX17CodePrinter
    language = "c++"
    math_prefix = "std::"

    def header(self) -> None:
        n = self.plan.n
        for line in f"""@@BANNER@@
#include <array>
#include <cmath>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

constexpr double IKINE_SINGULARITY_TOL = {SINGULARITY_TOL!r};

enum class IkStatus {{
    Ok = 0,          // solutions found
    Singular = 1,    // one or more branches were dropped due to a singularity
    Unreachable = 2  // no solution, the pose is outside the workspace
}};

struct IkResult {{
    std::vector<std::array<double, {n}>> solutions;
    IkStatus status = IkStatus::Unreachable;
}};

struct IkContext {{
    bool singular = false;
}};

inline double safe_sqrt(double x) {{
    // round-off can make a zero argument slightly negative
    return x < -1e-9 ? std::nan("") : std::sqrt(x > 0.0 ? x : 0.0);
}}

inline double atan2c(IkContext& ctx, double y, double x) {{
    if (std::isnan(x) || std::isnan(y)) return std::nan("");
    if (std::hypot(y, x) < IKINE_SINGULARITY_TOL) {{
        ctx.singular = true;  // the joint is undetermined
        return std::nan("");
    }}
    return std::atan2(y, x);
}}

@@POLY@@

// T is the 4x4 end-effector pose, row-major.
inline IkResult {self.name}(const double* T, double tol = 1e-6) {{
    IkContext ctx;
    IkResult result;""".splitlines():
            self.lines.append(line)
        self.poly_helper(_CXX_POLY, 1)
        self.depth = 1
        for line in self.pose_lines():
            self.put(line)

    def poly_open(self, joint: int, coeffs: list[str]) -> None:
        self.put(
            f"const std::vector<double> angles{joint} = "
            f"half_angle_roots(ctx, {{{', '.join(coeffs)}}});"
        )
        self.put(f"for (const double q{joint} : angles{joint}) {{")
        self.depth += 1

    def footer(self) -> None:
        self.put("if (ctx.singular) result.status = IkStatus::Singular;")
        self.put("else if (!result.solutions.empty()) result.status = IkStatus::Ok;")
        self.put("return result;")
        self.depth = 0
        self.put("}")

    def push(self, guarded: bool) -> None:
        self.put("if (err < tol) {" if guarded else "{")
        self.put(f"    result.solutions.push_back({{{self.qs()}}});")
        self.put("}")


class _RustEmitter(_Emitter):
    printer_class = RustCodePrinter
    language = "rust"

    def header(self) -> None:
        n = self.plan.n
        p = self.plan.pose
        for line in f"""@@BANNER@@
use std::cell::Cell;
use std::f64::consts::PI;

const SINGULARITY_TOL: f64 = {SINGULARITY_TOL!r};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IkStatus {{
    Ok,          // solutions found
    Singular,    // one or more branches were dropped due to a singularity
    Unreachable, // no solution, the pose is outside the workspace
}}

pub struct IkResult {{
    pub solutions: Vec<[f64; {n}]>,
    pub status: IkStatus,
}}

struct Ctx {{
    singular: Cell<bool>,
}}

impl Ctx {{
    fn atan2c(&self, y: f64, x: f64) -> f64 {{
        if x.is_nan() || y.is_nan() {{
            return f64::NAN;
        }}
        if y.hypot(x) < SINGULARITY_TOL {{
            self.singular.set(true); // the joint is undetermined
            return f64::NAN;
        }}
        y.atan2(x)
    }}
}}

trait SafeSqrt {{
    fn safe_sqrt(self) -> f64;
}}

impl SafeSqrt for f64 {{
    // round-off can make a zero argument slightly negative
    fn safe_sqrt(self) -> f64 {{
        if self < -1e-9 {{ f64::NAN }} else {{ self.max(0.0).sqrt() }}
    }}
}}

/// `t` is the 4x4 end-effector pose, row-major.
#[allow(non_snake_case, unused_variables)]
pub fn {self.name}(t: &[f64; 16], tol: f64) -> IkResult {{
    let ctx = Ctx {{ singular: Cell::new(false) }};
    let ({", ".join(p[0:4])}) = (t[0], t[1], t[2], t[3]);
    let ({", ".join(p[4:8])}) = (t[4], t[5], t[6], t[7]);
    let ({", ".join(p[8:12])}) = (t[8], t[9], t[10], t[11]);
    let mut solutions: Vec<[f64; {n}]> = Vec::new();""".splitlines():
            self.lines.append(line)
        self.depth = 1

    def footer(self) -> None:
        for line in [
            "let status = if ctx.singular.get() {",
            "    IkStatus::Singular",
            "} else if !solutions.is_empty() {",
            "    IkStatus::Ok",
            "} else {",
            "    IkStatus::Unreachable",
            "};",
            "IkResult { solutions, status }",
        ]:
            self.put(line)
        self.depth = 0
        self.put("}")

    def expr(self, e: sp.Expr) -> str:
        """Source for an expression, in which every number is a float."""
        text = super().expr(e)
        # Rust does not multiply an integer by a float, so ``2*x`` is ``2.0*x``.
        # The exponent of ``powi`` is an integer, and so is any in a number like 1e-9.
        text = re.sub(r"\.powi\((-?\d+)\)", r".powi(@\1)", text)
        text = re.sub(r"(?<![\w.@])(?<!e-)(?<!e\+)(\d+)(?![\w.])", r"\1.0", text)
        return text.replace(".powi(@", ".powi(")

    def decl(self, name: str, value: str) -> None:
        self.put(f"let {name} = {value};")

    def loop_open(self, k: str, count: int) -> None:
        self.put(f"for {k} in 0..{count} {{")
        self.depth += 1

    def poly_open(self, joint: int, coeffs: list[str]) -> None:
        raise NotImplementedError("polynomial solutions are not yet supported for Rust")

    def loop_close(self) -> None:
        self.depth -= 1
        self.put("}")

    def select(self, q: str, k: str, alts: list[str]) -> None:
        s = "{ " + alts[-1] + " }"
        for i in range(len(alts) - 2, -1, -1):
            s = f"if {k} == {i} {{ {alts[i]} }} else {s}"
        self.put(f"let {q}: f64 = {s};")

    def skip_if_nan(self, q: str) -> None:
        self.put(f"if {q}.is_nan() {{ continue; }}")

    def sin(self, q: str) -> str:
        return f"{q}.sin()"

    def cos(self, q: str) -> str:
        return f"{q}.cos()"

    def atan2(self, y: str, x: str) -> str:
        return f"{y}.atan2({x})"

    def error_start(self) -> None:
        self.put("let mut err: f64 = 0.0;")

    def error_update(self, e: str, pose: str) -> None:
        self.put(f"err = err.max(({e} - {pose}).abs());")

    def check(self, e: str) -> None:
        self.put(f"if ({e}).abs() > tol {{ continue; }} // a wrong alternative")

    def push(self, guarded: bool) -> None:
        self.put("if err < tol {" if guarded else "{")
        self.put(f"    solutions.push([{self.qs()}]);")
        self.put("}")


_EMITTERS = {
    "python": _PythonEmitter,
    "c": _CEmitter,
    "c++": _CxxEmitter,
    "rust": _RustEmitter,
}


def _emit(plan: _Plan, language: str, name: str) -> str:
    try:
        emitter = _EMITTERS[language]
    except KeyError:
        raise ValueError(f"language must be one of {sorted(_EMITTERS)}") from None
    return emitter(plan, name).generate()


def generate_code(
    robot: DHRobot,
    language: Literal["python", "c", "c++", "rust"] = "python",
    name: str = "ikine",
    verify: bool = True,
    polynomial: bool = False,
    max_time: float | None = None,
) -> str:
    """
    Generate source code for the inverse kinematics of a robot

    :param robot: DH robot model with numeric parameters
    :param language: target language
    :param name: name of the generated function
    :param verify: check each solution against the forward kinematics
    :param polynomial: give every revolute joint as the roots of a polynomial in
        tan(q/2) found numerically, rather than in closed form.  For testing.
    :param max_time: give up searching for a symbolic solution after this many
        seconds (Unix, main thread only)
    :returns: source code
    :raises ValueError: if no solution is found, the model has symbolic
        parameters, or the language is not supported

    The function takes the end-effector pose, as a 4x4 matrix, and returns all
    the joint configurations that reach it, together with a status:

    The revolute joint angles are wrapped to :math:`(-\\pi, \\pi]`.

    - ``OK``: solutions were found
    - ``SINGULAR``: one or more branches were dropped because a joint was
      undetermined, so the list may be empty or incomplete
    - ``UNREACHABLE``: no solutions, the pose is outside the workspace

    The link parameters are constants in the code, which uses only the
    language's standard library.  The joints are solved in the order found by
    :func:`ikine_sym`, with a nested loop over the alternative solutions for each
    joint.  Wrong alternatives are dropped as soon as they are chosen, using
    equations that are zero for a correct choice.  If ``verify`` is true each
    configuration is also checked against the forward kinematics, and only those
    that reach the pose to within ``tol`` are returned.

    The Python, C, C++ and Rust code is tested, the last three if there is a compiler.
    """
    return _emit(_make_plan(robot, verify, polynomial, max_time), language, name)


def generate_python(robot: DHRobot, name: str = "ikine") -> str:
    """Generate Python source, see :func:`generate_code`."""
    return generate_code(robot, "python", name)


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


def _wrap(q: np.ndarray) -> np.ndarray:
    """Wrap angles to the interval (-pi, pi]."""
    return np.arctan2(np.sin(q), np.cos(q))
