r"""
Prototype: inverse kinematics of the general 6R manipulator

This handles the arms that :mod:`symbolic_ik` cannot, those with neither a
spherical wrist nor three parallel axes, such as the Kinova Jaco and Mico.  It is
the method of Raghavan and Roth,

    M. Raghavan and B. Roth, "Inverse kinematics of the general 6R manipulator
    and related linkages," J. Mechanical Design 115(3):502-508, 1993,

in the form given by Manocha and Canny,

    D. Manocha and J. F. Canny, "Efficient inverse kinematics for general 6R
    manipulators," IEEE Trans. Robotics and Automation 10(5):648-657, 1994,

and in Tsai, "Robot Analysis", Wiley, 1999, appendix C.

Provenance: the method is that of those references.  The details of the
construction, that is the loop closure split and the fourteen equations, the
order of the monomials, the elimination of the right hand unknowns with a null
space, the 12x12 matrix polynomial, and the three possible splits, were taken from
the documentation and source of the open source package ssik (Personal Robotics
Lab, BSD-3 licence, https://github.com/personalrobotics/ssik, version 8.1.0, in
particular ``ssik/solvers/ikgeo/_raghavan_roth.py``), which was read for guidance
before this was written.  So was its observation that the choice of split decides
the conditioning for the Jaco, which we reproduced.  No code was copied.  The
recovery of the other joints, the filtering of roots, the choice of split, the
classification of the structure and the validation are our own, and everything
was tested against forward kinematics and numerical inverse kinematics.

The loop closure :math:`T = A_1 \cdots A_6` is split so that three link matrices
are a product on one side, and the other three are moved around :math:`T`, for
example

.. math::

    A_k A_{k+1} A_{k+2} = A_{k-1}^{-1} \cdots A_0^{-1} \, T \, A_5^{-1} \cdots A_{k+3}^{-1}

with joints numbered from 0.  The last matrix on the right postmultiplies, so
the two columns used, the :math:`z` axis :math:`l` and the translation :math:`p`,
do not depend on joint :math:`k+3`.  Fourteen equations come from them, those of
:math:`l`, :math:`p`, :math:`l \times p`, :math:`p \cdot p`, :math:`l \cdot p` and
:math:`(p \cdot p) l - 2 (l \cdot p) p`.  They are bilinear in
:math:`(\sin q_k, \cos q_k, 1)` times the monomials of the two other joints of
the product, and constants times the monomials of the two joints on the right.

The unknown monomials of the right joints are eliminated using the null space of
the 14x8 matrix that multiplies them, leaving six equations.  With
:math:`x = \tan(q_k/2)` and the half angle of the two left joints, and the
equations multiplied by one of those half angles, this is a 12x12 matrix
polynomial that is quadratic in :math:`x`.  Its eigenvalues are the 16 candidates
for :math:`x`, and the other joints follow by back substitution.

Which joint is :math:`k` makes a great difference to the conditioning, and for
some arms the wrong choice gives no solutions at all, so it is chosen
numerically.  Every solution is checked against the forward kinematics.

:func:`classify` finds the structure of an arm, and :func:`make_solver` uses it to
choose between this method and the closed form solutions of :mod:`symbolic_ik`.
Each is only good for its own kind of arm, so there is no fallback from one to the
other: a spherical wrist or three parallel axes goes to the closed form, anything
else to this method, and the choice is then checked on random poses.

Limits
------

- **The method degenerates for some geometries.**  A spherical wrist is the
  example found: the conditioning is fine but the solutions are not valid, and
  :meth:`General6R.validate` recovers the original configuration for under half of
  random poses of a Puma 560.  That is why such arms are not given to this method.
- **Completeness is not proven.**  Each solution is checked, but nothing checks
  that all of them are found.  A general 6R arm has up to 16 solutions.  On 300
  random poses each, the Jaco, Mico and UR5 always recovered the original
  configuration, but a few poses (2 for the Jaco, 5 for the Mico) gave an odd
  number of solutions (3, 7 or 9) when they normally come in pairs.  A root may
  have been lost, or two merged by the test for duplicates.  Comparison with
  numerical inverse kinematics from random starts is only a lower bound.
- **The eigenvalue solver matters.**  SciPy's QZ algorithm is used if it is
  installed.  NumPy alone, with a companion matrix, needs the leading matrix to be
  invertible.  In a trial before this class was written it recovered the original
  configuration for 298 of 300 poses of the Jaco and of the Mico, against 300 of
  300 with SciPy, and 300 of 300 for the UR5.  The NumPy-only path has not been
  tested in this class.
- **Conditioning depends on the pose.**  The split is chosen from 20 random
  poses, which does not guarantee it is the best for every pose.  The condition
  number of the leading matrix for the chosen split was between 40 and 2.4e4 on
  the poses tried for the Jaco and Mico, and accuracy falls as it rises.
- **Near singularities the answer is unreliable.**  A root whose null space is
  not one dimensional is dropped, and if that leaves no solutions the status is
  ``SINGULAR``.  Solutions near a singularity may be missing.
- **It is slower than the closed form**, about 1 ms per pose in NumPy on an M1,
  against a few microseconds for the generated closed form code in C.  The
  symbolic derivation takes a few seconds for each split, and is cached only in
  the running process.
- **Only standard DH robots with six revolute joints** are handled.
- **There is no code generation for this method yet.**  It is a Python class, and
  needs NumPy.

Run the tests with::

    pytest src/roboticstoolbox/robot/symbolic_ik_rr.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from roboticstoolbox.robot.DHRobot import DHRobot
from roboticstoolbox.robot.symbolic_ik import IKSolutions, IKStatus, generate_code

try:  # preferred for the eigenvalues, but NumPy alone is almost always enough
    import scipy.linalg
except ImportError:  # pragma: no cover
    scipy = None


# ------------------------------------------------------------------------- #
#                      classification of the structure                      #
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Structure:
    """
    Features of a serial arm that decide how its inverse kinematics is solved

    :param spherical: first joint (1-based) of each set of three consecutive
        revolute joints whose axes meet at a point, such as a spherical wrist
    :param parallel: first joint (1-based) of each set of three consecutive
        revolute joints with parallel axes, such as the UR arms
    """

    spherical: tuple[int, ...]
    parallel: tuple[int, ...]

    @property
    def general(self) -> bool:
        """True if there is neither, so the arm needs the general 6R method."""
        return not (self.spherical or self.parallel)


def classify(robot: DHRobot, tol: float = 1e-8) -> Structure:
    r"""
    Find the spherical and parallel joint triples of a standard DH robot

    :param robot: robot with standard DH links
    :param tol: tolerance for a length or sine of an angle being zero
    :raises NotImplementedError: for modified DH links

    In standard DH the axis of joint :math:`i` is :math:`z_{i-1}`, :math:`a_i` is
    the length of the common normal between :math:`z_{i-1}` and :math:`z_i`,
    :math:`d_{i}` the offset along :math:`z_{i-1}`, and :math:`\alpha_i` the twist
    from :math:`z_{i-1}` to :math:`z_i`.  For joints :math:`i, i+1, i+2`:

    - the axes meet at a point if :math:`a_i = a_{i+1} = d_{i+1} = 0` and neither
      pair of axes is parallel (the first two meet at the origin of frame
      :math:`i`, the second two at the origin of frame :math:`i+1`, and these are
      the same point if :math:`d_{i+1}=0`)
    - the axes are parallel if :math:`\alpha_i` and :math:`\alpha_{i+1}` are 0 or
      :math:`\pi`.

    The wrist test of :meth:`DHRobot.isspherical` is the first with
    :math:`i = n-2`, but uses exact comparison of floating point numbers.
    """
    links = robot.links
    if any(link.mdh for link in links):
        raise NotImplementedError("modified DH links are not supported")

    def zero(x) -> bool:
        return abs(float(x)) < tol

    def parallel_pair(link) -> bool:  # the twist is 0 or pi
        return zero(np.sin(float(link.alpha)))

    spherical, parallel = [], []
    for i in range(len(links) - 2):
        A, B, C = links[i : i + 3]  # joints i+1, i+2, i+3 (1-based)
        if not all(link.isrevolute for link in (A, B, C)):
            continue
        if zero(A.a) and zero(B.a) and zero(B.d):
            if not parallel_pair(A) and not parallel_pair(B):
                spherical.append(i + 1)
        if parallel_pair(A) and parallel_pair(B):
            parallel.append(i + 1)
    return Structure(tuple(spherical), tuple(parallel))


# ------------------------------------------------------------------------- #
#                         the Raghavan-Roth solver                          #
# ------------------------------------------------------------------------- #

# joints (0-based) in each split, by k: the two joints whose monomials are on
# the left, the two whose monomials are on the right, and the dropped joint
_SPLITS = {
    2: ((3, 4), (0, 1), 5),
    1: ((2, 3), (0, 5), 4),
    0: ((1, 2), (4, 5), 3),
}

# the monomials x_u^i x_v^j of the 9-vector, as (i, j)
_MON9 = [(2, 2), (2, 1), (2, 0), (1, 2), (1, 1), (1, 0), (0, 2), (0, 1), (0, 0)]
# and of the 12-vector, three more from multiplying by x_u
_MON12 = _MON9 + [(3, 2), (3, 1), (3, 0)]


def _clean(x: float) -> float:
    """Remove round-off from the sine or cosine of a multiple of a right angle."""
    for target in (0.0, 1.0, -1.0):
        if abs(x - target) < 1e-12:
            return target
    return x


def _weierstrass() -> np.ndarray:
    r"""
    Matrix that converts the sin/cos monomials of two joints to half-angle ones

    Row :math:`m` is the expansion of the :math:`m`'th of the nine monomials
    :math:`(s_u s_v, s_u c_v, c_u s_v, c_u c_v, s_u, c_u, s_v, c_v, 1)`, multiplied
    by :math:`(1+x_u^2)(1+x_v^2)`, in the monomials :math:`x_u^i x_v^j`, using
    :math:`s = 2x/(1+x^2)` and :math:`c = (1-x^2)/(1+x^2)`.
    """
    xu, xv = sp.symbols("xu xv")
    part = {
        "su": 2 * xu, "cu": 1 - xu**2, "du": 1 + xu**2,
        "sv": 2 * xv, "cv": 1 - xv**2, "dv": 1 + xv**2,
    }  # fmt: skip
    rows = [
        ("su", "sv"), ("su", "cv"), ("cu", "sv"), ("cu", "cv"),
        ("su", "dv"), ("cu", "dv"), ("du", "sv"), ("du", "cv"), ("du", "dv"),
    ]  # fmt: skip
    W = np.zeros((9, 9))
    for row, (a, b) in enumerate(rows):
        for (i, j), coeff in sp.Poly(sp.expand(part[a] * part[b]), xu, xv).terms():
            W[row, _MON9.index((i, j))] = float(coeff)
    return W


_W = _weierstrass()


def _embed(E: np.ndarray) -> np.ndarray:
    """
    The 12x12 block of the system, from a 6x9 set of equations

    The top six rows are the equations, and the bottom six are the same
    equations multiplied by :math:`x_u`, which maps each monomial of the 9-vector
    to one of the 12-vector.
    """
    M = np.zeros((12, 12))
    M[:6, :9] = E
    for j, (i, k) in enumerate(_MON9):
        M[6:, _MON12.index((i + 1, k))] += E[:, j]
    return M


def _dh(theta: float, alpha: float, a: float, d: float) -> np.ndarray:
    """Standard DH link transform."""
    ca, sa = _clean(np.cos(alpha)), _clean(np.sin(alpha))
    s, c = np.sin(theta), np.cos(theta)
    return np.array(
        [[c, -s * ca, s * sa, a * c], [s, c * ca, -c * sa, a * s], [0, sa, ca, d], [0, 0, 0, 1]]
    )


def _derive(alpha, a, d, k: int):
    """
    Derive the matrices of the 14 equations for a robot, symbolically

    :param alpha: twist angles of the six links
    :param a: link lengths
    :param d: link offsets
    :param k: which split, see the module documentation
    :returns: four functions of the 12 elements of the pose, which return the
        matrices ``P_sin``, ``P_cos``, ``P_one`` (14x9) and ``Q`` (14x8) so that
        ``(sin(q_k) P_sin + cos(q_k) P_cos + P_one) @ v_left = Q @ v_right``
    """
    left, right, drop = _SPLITS[k]
    s = sp.symbols("s0:6", real=True)
    c = sp.symbols("c0:6", real=True)
    t = sp.symbols("T0:12", real=True)
    T = sp.Matrix([list(t[0:4]), list(t[4:8]), list(t[8:12]), [0, 0, 0, 1]])

    def link(i: int, inverse: bool = False) -> sp.Matrix:
        ca, sa = _clean(np.cos(alpha[i])), _clean(np.sin(alpha[i]))
        if inverse:
            return sp.Matrix(
                [[c[i], s[i], 0, -a[i]],
                 [-s[i] * ca, c[i] * ca, sa, -d[i] * sa],
                 [s[i] * sa, -c[i] * sa, ca, -d[i] * ca],
                 [0, 0, 0, 1]]
            )  # fmt: skip
        return sp.Matrix(
            [[c[i], -s[i] * ca, s[i] * sa, a[i] * c[i]],
             [s[i], c[i] * ca, -c[i] * sa, a[i] * s[i]],
             [0, sa, ca, d[i]],
             [0, 0, 0, 1]]
        )  # fmt: skip

    if k == 2:
        lhs, rhs = link(2) * link(3) * link(4), link(1, True) * link(0, True) * T * link(5, True)
    elif k == 1:
        lhs, rhs = link(1) * link(2) * link(3), link(0, True) * T * link(5, True) * link(4, True)
    else:
        lhs, rhs = link(0) * link(1) * link(2), T * link(5, True) * link(4, True) * link(3, True)

    lL, lR, pL, pR = lhs[:3, 2], rhs[:3, 2], lhs[:3, 3], rhs[:3, 3]
    equations = [lL[i] - lR[i] for i in range(3)] + [pL[i] - pR[i] for i in range(3)]
    cL, cR = lL.cross(pL), lR.cross(pR)
    equations += [cL[i] - cR[i] for i in range(3)]
    ppL, ppR, lpL, lpR = pL.dot(pL), pR.dot(pR), lL.dot(pL), lR.dot(pR)
    equations += [ppL - ppR, lpL - lpR]
    vL, vR = ppL * lL - 2 * lpL * pL, ppR * lR - 2 * lpR * pR
    equations += [vL[i] - vR[i] for i in range(3)]

    active = [i for i in range(6) if i != drop]
    gens = [s[i] for i in active] + [c[i] for i in active]
    trig = [s[i] ** 2 + c[i] ** 2 - 1 for i in active]
    u, v = left
    p, q = right
    left9 = [s[u] * s[v], s[u] * c[v], c[u] * s[v], c[u] * c[v], s[u], c[u], s[v], c[v], sp.Integer(1)]
    right8 = [s[p] * s[q], s[p] * c[q], c[p] * s[q], c[p] * c[q], s[p], c[p], s[q], c[q]]

    Psin, Pcos, Pone, Q = sp.zeros(14, 9), sp.zeros(14, 9), sp.zeros(14, 9), sp.zeros(14, 8)
    for row, eq in enumerate(equations):
        _, remainder = sp.reduced(sp.expand(eq), trig, *gens)
        poly = sp.Poly(sp.expand(remainder), *gens)
        for j, mon in enumerate(left9):
            Psin[row, j] = poly.coeff_monomial(sp.expand(s[k] * mon))
            Pcos[row, j] = poly.coeff_monomial(sp.expand(c[k] * mon))
            Pone[row, j] = poly.coeff_monomial(mon)
        for j, mon in enumerate(right8):
            Q[row, j] = -poly.coeff_monomial(mon)
    return [sp.lambdify(t, m, "numpy") for m in (Psin, Pcos, Pone, Q)]


_DERIVED: dict[tuple, list] = {}  # in-process cache, a derivation takes seconds


def _eigenvalues(M0: np.ndarray, M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """Eigenvalues ``x`` of the matrix polynomial ``(M2 x^2 + M1 x + M0) v = 0``."""
    n = len(M0)
    Z, I = np.zeros((n, n)), np.eye(n)
    if scipy is not None:  # QZ copes with a singular leading matrix
        return scipy.linalg.eig(
            np.block([[Z, I], [-M0, -M1]]), np.block([[I, Z], [Z, M2]]), right=False
        )
    companion = np.block([[Z, I], [-np.linalg.solve(M2, M0), -np.linalg.solve(M2, M1)]])
    return np.linalg.eigvals(companion)


class General6R:
    r"""
    Inverse kinematics of a general 6R arm

    :param robot: robot with six revolute joints and standard DH links
    :param k: which split of the loop closure to use, 0, 1 or 2, see the module
        documentation.  By default the best of the three is chosen numerically.
    :param seed: seed for the random poses used to choose ``k``
    :raises ValueError: if the robot does not have six revolute joints

    The first time a robot is used the equations are derived symbolically, which
    takes a few seconds per split.

    ``solve(T)`` returns all the solutions for a pose as :class:`IKSolutions`.

    .. note::

        The method degenerates for some geometries, for example a spherical
        wrist.  Check with :meth:`validate`, or use :func:`classify`.
    """

    def __init__(self, robot: DHRobot, k: int | None = None, seed: int = 0):
        if robot.n != 6 or not all(link.isrevolute for link in robot.links):
            raise ValueError("a robot with six revolute joints is required")
        if any(link.mdh for link in robot.links):
            raise NotImplementedError("modified DH links are not supported")
        self.robot = robot
        self.alpha = [float(link.alpha) for link in robot.links]
        self.a = [float(link.a) for link in robot.links]
        self.d = [float(link.d) for link in robot.links]
        self.offset = np.array([float(link.offset) for link in robot.links])
        self.sign = np.array([-1.0 if link.isflip else 1.0 for link in robot.links])
        self.revolute = np.ones(6, dtype=bool)
        self.condition: dict[int, float] = {}
        self.k = self._choose(seed) if k is None else k
        self._functions = self._derived(self.k)

    # --- set up ---

    def _derived(self, k: int) -> list:
        key = (tuple(self.alpha), tuple(self.a), tuple(self.d), k)
        if key not in _DERIVED:
            _DERIVED[key] = _derive(self.alpha, self.a, self.d, k)
        return _DERIVED[key]

    def _random_poses(self, count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed)
        poses = []
        for _ in range(count):
            q = rng.uniform(-np.pi, np.pi, 6)
            poses.append((q, self._fkine(self.sign * q + self.offset)))
        return poses

    def _choose(self, seed: int, count: int = 20) -> int:
        """The split that recovers the most poses, then has the best conditioning."""
        poses = self._random_poses(count, seed)
        scores = {}
        for k in (0, 1, 2):
            functions = self._derived(k)
            conds, hits = [], 0
            for q, T in poses:
                result = self._solve(functions, k, T)
                conds.append(result[1])
                hits += any(self._same(s, q) for s in result[0])
            self.condition[k] = float(np.median(conds))
            scores[k] = (-hits, self.condition[k])
        return min(scores, key=lambda k: scores[k])

    # --- forward kinematics and comparisons ---

    def _fkine(self, theta: np.ndarray) -> np.ndarray:
        T = np.eye(4)
        for i in range(6):
            T = T @ _dh(theta[i], self.alpha[i], self.a[i], self.d[i])
        return T

    @staticmethod
    def _same(q1: np.ndarray, q2: np.ndarray, tol: float = 1e-5) -> bool:
        d = q1 - q2
        return bool(np.all(np.abs(np.arctan2(np.sin(d), np.cos(d))) < tol))

    # --- solving ---

    def _solve(self, functions: list, k: int, T: np.ndarray, tol: float = 1e-6):
        """
        :returns: ``(solutions, cond, ambiguous)``: the joint vectors ``q`` that
            reach the pose, the condition number of the leading matrix, and the
            number of roots whose null space was not one dimensional
        """
        left, right, drop = _SPLITS[k]
        Tn = T[:3, :].flatten()
        Psin, Pcos, Pone, Q = [np.array(f(*Tn), dtype=float) for f in functions]

        # eliminate the right monomials with the left null space of Q
        U, _, _ = np.linalg.svd(Q, full_matrices=True)
        N = U[:, -6:]
        Es, Ec, Eo = N.T @ Psin, N.T @ Pcos, N.T @ Pone

        # half angles: x = tan(q_k / 2) and, for the left joints, the 9 monomials
        M2, M1, M0 = _embed((Eo - Ec) @ _W), _embed(2 * Es @ _W), _embed((Eo + Ec) @ _W)
        cond = float(np.linalg.cond(M2))
        eigenvalues = _eigenvalues(M0, M1, M2)
        roots = [
            e.real for e in eigenvalues
            if np.isfinite(e) and abs(e.imag) < 1e-6 * (1 + abs(e.real)) and abs(e) < 1e8
        ]  # fmt: skip

        found: list[np.ndarray] = []
        ambiguous = 0
        for x in roots:
            _, sv, Vt = np.linalg.svd(M2 * x**2 + M0 + M1 * x)
            if sv[-2] < 1e-9 * sv[0]:  # not a one dimensional null space
                ambiguous += 1
                continue
            v = Vt[-1]
            if abs(v[8]) < 1e-12:
                continue
            v = v / v[8]
            theta = np.zeros(6)
            theta[k] = 2 * np.arctan(x)
            theta[left[0]], theta[left[1]] = 2 * np.arctan(v[5]), 2 * np.arctan(v[7])

            su, cu = np.sin(theta[left[0]]), np.cos(theta[left[0]])
            sv_, cv = np.sin(theta[left[1]]), np.cos(theta[left[1]])
            v_left = np.array([su * sv_, su * cv, cu * sv_, cu * cv, su, cu, sv_, cv, 1.0])
            rhs = (np.sin(theta[k]) * Psin + np.cos(theta[k]) * Pcos + Pone) @ v_left
            v_right, *_ = np.linalg.lstsq(Q, rhs, rcond=None)
            theta[right[0]] = np.arctan2(v_right[4], v_right[5])
            theta[right[1]] = np.arctan2(v_right[6], v_right[7])

            # the dropped joint from  A_drop = (A_0..A_drop-1)^-1 T (A_drop+1..A_5)^-1
            pre, post = np.eye(4), np.eye(4)
            for i in range(drop):
                pre = pre @ _dh(theta[i], self.alpha[i], self.a[i], self.d[i])
            for i in range(drop + 1, 6):
                post = post @ _dh(theta[i], self.alpha[i], self.a[i], self.d[i])
            A = np.linalg.inv(pre) @ T @ np.linalg.inv(post)
            theta[drop] = np.arctan2(A[1, 0], A[0, 0])

            if np.abs(self._fkine(theta) - T).max() < tol:
                q = self.sign * (theta - self.offset)
                q = np.arctan2(np.sin(q), np.cos(q))
                if not any(self._same(q, other) for other in found):
                    found.append(q)
        return found, cond, ambiguous

    def solve(self, T: np.ndarray) -> IKSolutions:
        """
        All the joint configurations that reach a pose

        :param T: end-effector pose, a 4x4 array
        :returns: the solutions, wrapped to (-pi, pi].  The status is ``OK`` if
            there are any.  If there are none it is ``SINGULAR`` if a candidate
            had to be dropped because its null space was ambiguous, otherwise
            ``UNREACHABLE``.
        """
        T = np.asarray(T, dtype=float)
        found, _, ambiguous = self._solve(self._functions, self.k, T)
        if found:
            status = IKStatus.OK
        elif ambiguous:
            status = IKStatus.SINGULAR
        else:
            status = IKStatus.UNREACHABLE
        return IKSolutions(np.array(found).reshape(-1, 6), status, self.revolute)

    def validate(self, trials: int = 100, seed: int = 1) -> float:
        """
        Fraction of random poses for which the original configuration is found

        :param trials: number of random configurations to try
        :param seed: seed for the random configurations
        :returns: a value between 0 and 1.  A method that works gives 1, or
            very nearly, and one that does not apply to this robot gives 0.
        """
        hits = 0
        for q, T in self._random_poses(trials, seed):
            hits += any(self._same(s, q) for s in self.solve(T).q)
        return hits / trials


# ------------------------------------------------------------------------- #
#                          choosing a method                                #
# ------------------------------------------------------------------------- #


class ClosedForm:
    r"""
    Inverse kinematics from the closed form solution of :mod:`symbolic_ik`

    :param robot: robot with standard DH links and six joints
    :param max_time: give up looking for the symbolic solution after this many
        seconds
    :raises ValueError: if no closed form solution is found

    The solution is generated as Python code, which is then run.  The same code
    can be had in other languages from :func:`symbolic_ik.generate_code`.
    """

    def __init__(self, robot: DHRobot, max_time: float | None = 120.0):
        self.robot = robot
        self.revolute = np.array([link.isrevolute for link in robot.links])
        self.source = generate_code(robot, "python", "ikine", max_time=max_time)
        namespace: dict = {}
        exec(self.source, namespace)  # noqa: S102
        self._ikine = namespace["ikine"]

    def solve(self, T: np.ndarray) -> IKSolutions:
        """All the joint configurations that reach a pose, see :class:`IKSolutions`."""
        result = self._ikine(np.asarray(T, dtype=float))
        return IKSolutions.from_generated(result, self.revolute)

    def validate(self, trials: int = 100, seed: int = 1) -> float:
        """Fraction of random poses for which the original configuration is found."""
        rng = np.random.default_rng(seed)
        hits = 0
        for _ in range(trials):
            q = np.where(
                self.revolute,
                rng.uniform(-np.pi, np.pi, self.robot.n),
                rng.uniform(0.2, 0.6, self.robot.n),
            )
            sols = self.solve(self.robot.fkine(q).A)
            if len(sols):
                d = sols.q - q
                d[:, self.revolute] = np.arctan2(
                    np.sin(d[:, self.revolute]), np.cos(d[:, self.revolute])
                )
                hits += bool(np.any(np.all(np.abs(d) < 1e-5, axis=1)))
        return hits / trials


def make_solver(
    robot: DHRobot,
    trials: int = 100,
    min_success: float = 0.99,
    max_time: float | None = 120.0,
) -> ClosedForm | General6R:
    """
    Choose and build the inverse kinematics method for a robot

    :param robot: robot with standard DH links and six joints
    :param trials: number of random poses used to check the method
    :param min_success: smallest acceptable fraction of those poses for which the
        original configuration is recovered
    :param max_time: give up looking for a closed form solution after this long
    :returns: a :class:`ClosedForm` solver for an arm with a spherical wrist or
        three parallel axes, otherwise a :class:`General6R` solver
    :raises ValueError: if the method does not recover enough of the poses

    Each method is only good for its own kind of arm, for example the general
    method fails for a spherical wrist, so there is no fallback from one to the
    other.  The method is checked on random poses before it is returned.
    """
    if classify(robot).general:
        solver = General6R(robot)
        method = "general 6R"
    else:
        solver = ClosedForm(robot, max_time)
        method = "closed form"
    rate = solver.validate(trials)
    if rate < min_success:
        raise ValueError(
            f"the {method} method recovered the configuration for only "
            f"{rate:.0%} of random poses for this robot"
        )
    return solver


# ------------------------------------------------------------------------- #
#                                  tests                                    #
# ------------------------------------------------------------------------- #


def test_classify():
    import roboticstoolbox as rtb

    models = rtb.models.DH
    # spherical wrist: joints 4, 5, 6
    for name in ("Puma560", "IRB140", "KR5", "Stanford"):
        s = classify(getattr(models, name)())
        assert 4 in s.spherical, name
        assert not s.general
    # the UR arms: parallel axes 2, 3, 4
    for name in ("UR3", "UR5", "UR10"):
        s = classify(getattr(models, name)())
        assert s.parallel == (2,) and not s.spherical, name
    # neither
    for name in ("Jaco", "Mico"):
        assert classify(getattr(models, name)()).general, name


def test_classify_tolerance():
    """Round-off does not matter, unlike the exact comparison of isspherical."""
    import roboticstoolbox as rtb
    from roboticstoolbox import DHRobot, RevoluteDH

    robot = rtb.models.DH.Puma560()
    assert robot.isspherical()
    noisy = DHRobot(
        [
            RevoluteDH(
                d=link.d + 1e-13, a=link.a + (1e-13 if link.a == 0 else 0),
                alpha=link.alpha * (1 + 1e-12), offset=link.offset,
            )
            for link in robot.links
        ]
    )  # fmt: skip
    assert not noisy.isspherical()  # exact comparison fails
    assert classify(noisy, tol=1e-8).spherical  # a tolerance does not


def test_choice_of_split():
    """The Jaco has pathological conditioning for two of the splits."""
    import roboticstoolbox as rtb

    solver = General6R(rtb.models.DH.Jaco())
    assert solver.k == 1
    assert solver.condition[1] < 1e4 < solver.condition[0]
    assert solver.condition[1] < 1e4 < solver.condition[2]


def test_general_6r():
    import roboticstoolbox as rtb

    for name in ("Jaco", "Mico", "UR5"):
        robot = getattr(rtb.models.DH, name)()
        solver = General6R(robot)
        assert solver.validate(100) >= 0.99, name

        # solutions reach the pose
        q0 = np.array([0.3, -0.8, 1.1, 0.4, -0.6, 0.9])
        T = robot.fkine(q0).A
        sols = solver.solve(T)
        assert sols.status is IKStatus.OK and len(sols) >= 2
        for q in sols.q:
            assert np.abs(robot.fkine(q).A - T).max() < 1e-6
        np.testing.assert_allclose(sols.nearest(q0), q0, atol=1e-5)


def test_unreachable():
    import roboticstoolbox as rtb
    from spatialmath import SE3

    solver = General6R(rtb.models.DH.Jaco())
    sols = solver.solve((SE3(5, 5, 5) * SE3.RPY([0.3, 0.4, 0.5])).A)
    assert sols.status is IKStatus.UNREACHABLE and len(sols) == 0


def test_not_applicable():
    """A spherical wrist is not the general case, and validation says so."""
    import roboticstoolbox as rtb

    robot = rtb.models.DH.Puma560()
    assert not classify(robot).general
    assert General6R(robot, k=1).validate(30) < 0.5


def test_bad_robot():
    import pytest
    import roboticstoolbox as rtb

    with pytest.raises(ValueError):
        General6R(rtb.models.DH.Planar2())
    with pytest.raises(ValueError):
        General6R(rtb.models.DH.Stanford())  # a prismatic joint


def test_make_solver():
    import pytest
    import roboticstoolbox as rtb

    models = rtb.models.DH
    for name, kind in (
        ("Puma560", ClosedForm),
        ("Stanford", ClosedForm),
        ("UR5", ClosedForm),
        ("Jaco", General6R),
    ):
        robot = getattr(models, name)()
        solver = make_solver(robot, trials=30)
        assert isinstance(solver, kind), name

        q0 = np.where(
            [link.isrevolute for link in robot.links],
            [0.3, -0.8, 1.1, 0.4, -0.6, 0.9],
            [0.3, 0.5, 0.4, 0.2, 0.6, 0.5],
        )
        sols = solver.solve(robot.fkine(q0).A)
        assert sols.status is IKStatus.OK
        np.testing.assert_allclose(sols.nearest(q0), q0, atol=1e-5)

    with pytest.raises(ValueError, match="recovered"):
        make_solver(models.Jaco(), trials=10, min_success=1.01)


if __name__ == "__main__":  # pragma nocover
    import pytest

    pytest.main([__file__, "-v", "-s"])
