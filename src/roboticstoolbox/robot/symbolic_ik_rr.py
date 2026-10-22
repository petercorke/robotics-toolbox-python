r"""
Prototype: inverse kinematics of the general 6R manipulator

This handles the arms that ``symbolic_ik`` cannot, those with neither a
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

with joints numbered from 0.  For :math:`k=2` this is Eq. 3 of Raghavan and Roth,
which moves the three joints that are not in the product to the right.  Writing
each link matrix as :math:`A_i = A_{i,v} A_{i,s}`, the first being the joint
variable alone and the second the constant link parameters, lets the variable of
joint :math:`k-1` be isolated, so that it appears only as a rotation about
:math:`z`.  The last matrix on the right postmultiplies, so the two columns used,
the :math:`z` axis :math:`l` and the translation :math:`p`, do not depend on joint
:math:`k+3`.  Fourteen equations come from them, those of
:math:`l`, :math:`p`, :math:`l \times p`, :math:`p \cdot p`, :math:`l \cdot p` and
:math:`(p \cdot p) l - 2 (l \cdot p) p`.  They are bilinear in
:math:`(\sin q_k, \cos q_k, 1)` times the monomials of the two other joints of
the product, and constants times the monomials of the two joints on the right.

The unknown monomials of the right joints are eliminated using the null space of
the 14x8 matrix that multiplies them, leaving six equations.  With
:math:`x = \tan(q_k/2)` and the half angle of the two left joints, and the
equations multiplied by one of those half angles, this is a 12x12 matrix
polynomial that is quadratic in :math:`x`, Eq. 14 of the paper.  The determinant is
the *characteristic polynomial* of degree 16, so a general 6R arm has up to 16
solutions, of which in general only some are real.  A quadratic matrix polynomial
of size 12 has 24 eigenvalues, though, not 16, and the 8 that are not roots of
the characteristic polynomial are candidates that do not correspond to a
solution.  They are discarded by the check of each candidate against the forward
kinematics.  The other joints follow from the real roots by back substitution.

Which joint is :math:`k` makes a great difference to the conditioning, and for
some arms the wrong choice gives no solutions at all, so it is chosen
numerically.  The paper obtains the three choices as the cyclic rearrangements of
its Eq. 4 (the section "Alternative Equations": joints 2, 3 and 6 have nothing
special about them), here ``_SPLITS``.  Every solution is checked against the
forward kinematics.

``classify`` finds the structure of an arm, and ``make_solver`` uses it to
choose between this method and the closed form solutions of ``symbolic_ik``.
Each is only good for its own kind of arm, so there is no fallback from one to the
other: a spherical wrist or three parallel axes goes to the closed form, anything
else to this method, and the choice is then checked on random poses.

Limits
------

- **The method degenerates for some geometries.**  A spherical wrist is the
  example found: the conditioning is fine but the solutions are not valid, and
  :meth:`General6RIK.validate` recovers the original configuration for under half of
  random poses of a Puma 560.  That is why such arms are not given to this method.
- **Completeness is not proven.**  Each solution is checked, but nothing checks
  that all of them are found.  A general 6R arm has up to 16 solutions.  On 300
  random poses each, the Jaco, Mico and UR5 always recovered the original
  configuration, but one pose of the Jaco (and one of the Mico without SciPy) gave
  an odd number of solutions (7), when they normally come in pairs.  A root may
  have been lost, or two merged by the test for duplicates.  Before the
  candidates were polished, with Gauss-Newton steps on the joint angles, there
  were 2 for the Jaco and 5 for the Mico, most of them solutions that were just
  over the tolerance for being a solution.  Comparison with numerical inverse
  kinematics from random starts is only a lower bound.
- **The eigenvalue solver matters less than it did.**  SciPy's QZ algorithm is used
  if it is installed.  NumPy alone, with a companion matrix, needs the leading
  matrix to be invertible, and before polishing it recovered the original
  configuration for 298 of 300 poses of the Jaco and of the Mico, against 300 of
  300 with SciPy.  With polishing both recover 300 of 300, for the Jaco, Mico and
  UR5, but poorly conditioned poses are where NumPy alone is the weaker.
- **Conditioning depends on the pose.**  The split is chosen from 20 random
  poses, which does not guarantee it is the best for every pose.  The condition
  number of the leading matrix for the chosen split was between 40 and 2.4e4 on
  the poses tried for the Jaco and Mico, and accuracy falls as it rises.
- **Near singularities the answer is unreliable.**  A root whose null space is
  not one dimensional is dropped, and if that leaves no solutions the status is
  ``SINGULAR``.  Solutions near a singularity may be missing.
- **It is slower than the closed form.**  The Python class takes about 1.7 ms per
  pose in NumPy on an M1, and the generated C about 165 microseconds for the Jaco,
  against a few microseconds for the generated closed form code in C.  Polishing
  the candidates doubled the time of the Python.  The symbolic derivation takes a
  few seconds for each split, and is cached only in the running process.
- **Only standard DH robots with six revolute joints** are handled.  The paper
  also covers prismatic joints, by using the link offset :math:`d_i` and
  :math:`d_i^2` as the variables in place of :math:`\sin q_i` and
  :math:`\cos q_i`.  One prismatic joint (5R,P) keeps the degree at 16, but two
  or more lower the degree of the characteristic polynomial, which needs a
  different, smaller elimination (the paper's Appendix 1 is the case of 4R,2P
  for which it is 8).  Neither is implemented.
- **Special geometry loses constraint equations.**  The paper explains the
  failure for a spherical wrist: the number of power products falls, the
  characteristic polynomial drops to degree 4, and some joints become
  double-valued for each root.  That is the reason arms with a spherical wrist
  are given to the closed form method and not to this one.
- **Code is generated in Python, C and C++**, by :meth:`General6RIK.generate`, but
  not Rust.  The Python needs NumPy, and uses SciPy if it is installed.  The C and
  C++ need no library, they contain a small linear algebra kernel, see below.  It
  has been tested against NumPy, but it is new code and has had much less use than
  LAPACK.

The kernel
----------

The C and C++ code includes a small dense linear algebra kernel in real
arithmetic, in plain C99 that is also valid C++: LU with partial pivoting,
Householder QR, a one-sided Jacobi SVD, and the eigenvalues of a general real
matrix by balancing, Householder reduction to Hessenberg form and the Francis
double shift QR iteration.  The matrices are tiny, at most 24x24, so a library
would add a dependency and no speed.  The algorithms are the standard published
ones, written for this project and not copied from any implementation, and
references are in the comments of the source: Golub and Van Loan, "Matrix
Computations"; M. R. Hestenes, J. SIAM 6(1), 1958; B. N. Parlett and C. Reinsch,
Numer. Math. 13, 1969; J. G. F. Francis, Computer J. 4, 1961-62; and J. H.
Wilkinson and C. Reinsch, "Handbook for Automatic Computation vol. II: Linear
Algebra", 1971, whose EISPACK ``hqr`` is the model for the QR iteration.

Run the tests with::

    pytest src/roboticstoolbox/robot/symbolic_ik_rr.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import sympy as sp
from sympy.printing.c import C99CodePrinter
from sympy.printing.pycode import PythonCodePrinter

from spatialmath import SE3
from spatialmath.base import tr2delta

from roboticstoolbox.robot.DHRobot import DHRobot
from roboticstoolbox.robot.IK import IKSolution, check_q0, normalise_q, within_limits
from roboticstoolbox.tools.types import ArrayLike
from roboticstoolbox.robot.symbolic_ik import (
    IKSolutions,
    IKStatus,
    _comment,
    _dh_info,
    _emit,
    _make_plan,
    _provenance,
)

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

    The wrist test of :meth:`~roboticstoolbox.robot.DHRobot.DHRobot.isspherical` is the first with
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


# A small dense linear algebra kernel, for the C and C++ code, in real arithmetic.
# The matrices are tiny (at most 24x24) so a linear algebra library would only add
# a dependency, and the matrices are stored row-major.  It is plain C99 that is also
# valid C++.
#
# The algorithms are the standard published ones, written for this project:
#   - LU decomposition with partial pivoting, and Householder QR, see Golub and Van
#     Loan, "Matrix Computations".
#   - one-sided Jacobi SVD, M. R. Hestenes, J. SIAM 6(1), 1958, see also J. Demmel
#     and K. Veselic, SIAM J. Matrix Anal. Appl. 13(4), 1992.
#   - eigenvalues of a general real matrix: balancing, B. N. Parlett and C.
#     Reinsch, Numer. Math. 13, 1969, reduction to Hessenberg form by Householder
#     reflections, and the Francis double shift QR algorithm, J. G. F. Francis,
#     Computer J. 4, 1961-62; see Wilkinson and Reinsch, "Handbook for Automatic
#     Computation vol. II: Linear Algebra" (1971), whose EISPACK translation of the
#     last is the model, and Golub and Van Loan section 7.5.
_C_KERNEL = """\
#define IKRR_MAXN 24

/* Solve A X = B, with A n-by-n and B n-by-m, by LU decomposition with partial
 * pivoting.  A is destroyed and B is overwritten by X.  Returns 0 if A is singular. */
static int ikrr_lu_solve(double *A, int n, double *B, int m) {
    for (int k = 0; k < n; ++k) {
        int p = k;
        double big = fabs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            if (fabs(A[i * n + k]) > big) { big = fabs(A[i * n + k]); p = i; }
        }
        if (big == 0.0) return 0;
        if (p != k) {
            for (int j = 0; j < n; ++j) { double t = A[k * n + j]; A[k * n + j] = A[p * n + j]; A[p * n + j] = t; }
            for (int j = 0; j < m; ++j) { double t = B[k * m + j]; B[k * m + j] = B[p * m + j]; B[p * m + j] = t; }
        }
        for (int i = k + 1; i < n; ++i) {
            double f = A[i * n + k] / A[k * n + k];
            for (int j = k; j < n; ++j) A[i * n + j] -= f * A[k * n + j];
            for (int j = 0; j < m; ++j) B[i * m + j] -= f * B[k * m + j];
        }
    }
    for (int j = 0; j < m; ++j) {
        for (int i = n - 1; i >= 0; --i) {
            double s = B[i * m + j];
            for (int k = i + 1; k < n; ++k) s -= A[i * n + k] * B[k * m + j];
            B[i * m + j] = s / A[i * n + i];
        }
    }
    return 1;
}

/* Householder QR of the m-by-n matrix A, m >= n.  On return the upper triangle of
 * A is R, and Qf is the m-by-m orthogonal matrix such that A = Qf R.  The last
 * m - n columns of Qf span the left null space of A. */
static void ikrr_qr(double *A, int m, int n, double *Qf) {
    double v[IKRR_MAXN];
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) Qf[i * m + j] = (i == j) ? 1.0 : 0.0;
    }
    for (int k = 0; k < n; ++k) {
        double norm = 0.0;
        for (int i = k; i < m; ++i) norm += A[i * n + k] * A[i * n + k];
        norm = sqrt(norm);
        if (norm == 0.0) continue;
        double alpha = (A[k * n + k] > 0.0) ? -norm : norm;
        for (int i = k; i < m; ++i) v[i] = A[i * n + k];
        v[k] -= alpha;
        double vv = 0.0;
        for (int i = k; i < m; ++i) vv += v[i] * v[i];
        if (vv == 0.0) continue;
        double beta = 2.0 / vv;
        for (int j = k; j < n; ++j) {
            double dot = 0.0;
            for (int i = k; i < m; ++i) dot += v[i] * A[i * n + j];
            dot *= beta;
            for (int i = k; i < m; ++i) A[i * n + j] -= dot * v[i];
        }
        for (int i = 0; i < m; ++i) {
            double dot = 0.0;
            for (int j = k; j < m; ++j) dot += Qf[i * m + j] * v[j];
            dot *= beta;
            for (int j = k; j < m; ++j) Qf[i * m + j] -= dot * v[j];
        }
        for (int i = k + 1; i < m; ++i) A[i * n + k] = 0.0;
    }
}

/* x minimising |A x - b| for the m-by-n matrix A already factored by ikrr_qr into
 * R (R is in the upper triangle of A) and Qf. */
static void ikrr_qr_solve(const double *R, const double *Qf, int m, int n, const double *b, double *x) {
    double y[IKRR_MAXN];
    for (int j = 0; j < n; ++j) {
        y[j] = 0.0;
        for (int i = 0; i < m; ++i) y[j] += Qf[i * m + j] * b[i];
    }
    for (int i = n - 1; i >= 0; --i) {
        double s = y[i];
        for (int k = i + 1; k < n; ++k) s -= R[i * n + k] * x[k];
        x[i] = s / R[i * n + i];
    }
}

/* Singular value decomposition of the square n-by-n matrix A by one-sided Jacobi
 * rotations.  A is not changed.  The singular values are written to s in
 * decreasing order, and the corresponding right singular vectors are the columns
 * of the n-by-n matrix V. */
static void ikrr_svd(const double *A, int n, double *s, double *V) {
    double U[IKRR_MAXN * IKRR_MAXN], norms[IKRR_MAXN];
    int order[IKRR_MAXN];
    for (int i = 0; i < n * n; ++i) U[i] = A[i];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) V[i * n + j] = (i == j) ? 1.0 : 0.0;
    }
    for (int sweep = 0; sweep < 60; ++sweep) {
        int rotated = 0;
        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int i = 0; i < n; ++i) {
                    alpha += U[i * n + p] * U[i * n + p];
                    beta += U[i * n + q] * U[i * n + q];
                    gamma += U[i * n + p] * U[i * n + q];
                }
                if (gamma == 0.0 || fabs(gamma) <= 1e-15 * sqrt(alpha * beta)) continue;
                rotated = 1;
                double zeta = (beta - alpha) / (2.0 * gamma);
                double t = ((zeta >= 0.0) ? 1.0 : -1.0) / (fabs(zeta) + sqrt(1.0 + zeta * zeta));
                double c = 1.0 / sqrt(1.0 + t * t), sn = c * t;
                for (int i = 0; i < n; ++i) {
                    double up = U[i * n + p], uq = U[i * n + q];
                    U[i * n + p] = c * up - sn * uq;
                    U[i * n + q] = sn * up + c * uq;
                    double vp = V[i * n + p], vq = V[i * n + q];
                    V[i * n + p] = c * vp - sn * vq;
                    V[i * n + q] = sn * vp + c * vq;
                }
            }
        }
        if (!rotated) break;
    }
    for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += U[i * n + j] * U[i * n + j];
        norms[j] = sqrt(sum);
        order[j] = j;
    }
    for (int i = 1; i < n; ++i) {  /* insertion sort, decreasing */
        int o = order[i], j = i - 1;
        while (j >= 0 && norms[order[j]] < norms[o]) { order[j + 1] = order[j]; --j; }
        order[j + 1] = o;
    }
    double Vs[IKRR_MAXN * IKRR_MAXN];
    for (int j = 0; j < n; ++j) {
        s[j] = norms[order[j]];
        for (int i = 0; i < n; ++i) Vs[i * n + j] = V[i * n + order[j]];
    }
    for (int i = 0; i < n * n; ++i) V[i] = Vs[i];
}

/* Scale rows and columns of A by powers of two so that the norm of each row is
 * about that of the column, which improves the accuracy of the eigenvalues.  This
 * is a similarity transform, so they are unchanged. */
static void ikrr_balance(double *A, int n) {
    int changed = 1;
    while (changed) {
        changed = 0;
        for (int i = 0; i < n; ++i) {
            double c = 0.0, r = 0.0;
            for (int j = 0; j < n; ++j) {
                if (j != i) { c += fabs(A[j * n + i]); r += fabs(A[i * n + j]); }
            }
            if (c == 0.0 || r == 0.0) continue;
            double f = 1.0;  /* power of two with c f^2 close to r */
            while (c * f * f < 0.5 * r) f *= 2.0;
            while (c * f * f > 2.0 * r) f *= 0.5;
            if (c * f + r / f < 0.95 * (c + r)) {
                for (int j = 0; j < n; ++j) { A[i * n + j] /= f; A[j * n + i] *= f; }
                changed = 1;
            }
        }
    }
}

/* Reduce A to upper Hessenberg form by Householder reflections. */
static void ikrr_hessenberg(double *A, int n) {
    double v[IKRR_MAXN];
    for (int k = 0; k < n - 2; ++k) {
        double norm = 0.0;
        for (int i = k + 1; i < n; ++i) norm += A[i * n + k] * A[i * n + k];
        norm = sqrt(norm);
        if (norm == 0.0) continue;
        double alpha = (A[(k + 1) * n + k] > 0.0) ? -norm : norm;
        for (int i = k + 1; i < n; ++i) v[i] = A[i * n + k];
        v[k + 1] -= alpha;
        double vv = 0.0;
        for (int i = k + 1; i < n; ++i) vv += v[i] * v[i];
        if (vv == 0.0) continue;
        double beta = 2.0 / vv;
        for (int j = k; j < n; ++j) {
            double dot = 0.0;
            for (int i = k + 1; i < n; ++i) dot += v[i] * A[i * n + j];
            dot *= beta;
            for (int i = k + 1; i < n; ++i) A[i * n + j] -= dot * v[i];
        }
        for (int i = 0; i < n; ++i) {
            double dot = 0.0;
            for (int j = k + 1; j < n; ++j) dot += A[i * n + j] * v[j];
            dot *= beta;
            for (int j = k + 1; j < n; ++j) A[i * n + j] -= dot * v[j];
        }
        for (int i = k + 2; i < n; ++i) A[i * n + k] = 0.0;
    }
}

/* One implicit double shift QR (Francis) step on the active block l..h of the
 * Hessenberg matrix H, the shifts being the roots of x^2 - s x + t. */
static void ikrr_francis(double *H, int n, int l, int h, double s, double t) {
    double x = H[l * n + l] * H[l * n + l] + H[l * n + l + 1] * H[(l + 1) * n + l] - s * H[l * n + l] + t;
    double y = H[(l + 1) * n + l] * (H[l * n + l] + H[(l + 1) * n + l + 1] - s);
    double z = H[(l + 1) * n + l] * H[(l + 2) * n + l + 1];
    for (int k = l; k <= h - 1; ++k) {
        int three = (k < h - 1);
        if (k > l) {
            x = H[k * n + k - 1];
            y = H[(k + 1) * n + k - 1];
            z = three ? H[(k + 2) * n + k - 1] : 0.0;
        }
        double norm = sqrt(x * x + y * y + z * z);
        if (norm == 0.0) continue;
        double alpha = (x > 0.0) ? -norm : norm;
        double v0 = x - alpha, v1 = y, v2 = z;
        double beta = 2.0 / (v0 * v0 + v1 * v1 + v2 * v2);
        for (int j = (k > l ? k - 1 : l); j <= h; ++j) {  /* P H */
            double dot = v0 * H[k * n + j] + v1 * H[(k + 1) * n + j] + (three ? v2 * H[(k + 2) * n + j] : 0.0);
            dot *= beta;
            H[k * n + j] -= dot * v0;
            H[(k + 1) * n + j] -= dot * v1;
            if (three) H[(k + 2) * n + j] -= dot * v2;
        }
        int last = (k + 3 < h) ? k + 3 : h;
        for (int i = l; i <= last; ++i) {  /* H P */
            double dot = H[i * n + k] * v0 + H[i * n + k + 1] * v1 + (three ? H[i * n + k + 2] * v2 : 0.0);
            dot *= beta;
            H[i * n + k] -= dot * v0;
            H[i * n + k + 1] -= dot * v1;
            if (three) H[i * n + k + 2] -= dot * v2;
        }
        if (k > l) {
            H[(k + 1) * n + k - 1] = 0.0;
            if (three) H[(k + 2) * n + k - 1] = 0.0;
        }
    }
}

/* Eigenvalues of the upper Hessenberg matrix H, which is destroyed.  The real and
 * imaginary parts are written to wr and wi.  Returns 0 if it fails to converge. */
static int ikrr_hqr(double *H, int n, double *wr, double *wi) {
    const double eps = 2.220446049250313e-16;
    double anorm = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = (i > 0 ? i - 1 : 0); j < n; ++j) anorm += fabs(H[i * n + j]);
    }
    int h = n - 1, iterations = 0;
    while (h >= 0) {
        int l = h;
        while (l > 0) {  /* the start of the unreduced block ending at h */
            double scale = fabs(H[(l - 1) * n + l - 1]) + fabs(H[l * n + l]);
            if (scale == 0.0) scale = anorm;
            if (fabs(H[l * n + l - 1]) <= eps * scale) { H[l * n + l - 1] = 0.0; break; }
            --l;
        }
        if (l == h) {  /* one eigenvalue */
            wr[h] = H[h * n + h]; wi[h] = 0.0;
            --h; iterations = 0;
        } else if (l == h - 1) {  /* a 2-by-2 block */
            double a = H[(h - 1) * n + h - 1], b = H[(h - 1) * n + h];
            double c = H[h * n + h - 1], d = H[h * n + h];
            double p = 0.5 * (a - d), q = p * p + b * c;
            if (q >= 0.0) {
                double z = p + ((p >= 0.0) ? sqrt(q) : -sqrt(q));
                wr[h - 1] = d + z;
                wr[h] = (z != 0.0) ? d - b * c / z : d + z;
                wi[h - 1] = wi[h] = 0.0;
            } else {
                wr[h - 1] = wr[h] = d + p;
                wi[h - 1] = sqrt(-q); wi[h] = -wi[h - 1];
            }
            h -= 2; iterations = 0;
        } else {
            if (iterations >= 60) return 0;
            double a = H[(h - 1) * n + h - 1], b = H[(h - 1) * n + h];
            double c = H[h * n + h - 1], d = H[h * n + h];
            double s = a + d, t = a * d - b * c;
            if (iterations == 10 || iterations == 20) {  /* break a cycle */
                double shift = d + 0.75 * (fabs(c) + fabs(H[(h - 1) * n + h - 2]));
                s = 2.0 * shift; t = shift * shift;
            }
            ikrr_francis(H, n, l, h, s, t);
            ++iterations;
        }
    }
    return 1;
}

/* Eigenvalues of the general real n-by-n matrix A, which is destroyed.  Returns 0
 * if the iteration fails to converge. */
static int ikrr_eig(double *A, int n, double *wr, double *wi) {
    ikrr_balance(A, n);
    ikrr_hessenberg(A, n);
    return ikrr_hqr(A, n, wr, wi);
}
"""


_GENERATED_C = """\
@@BANNER@@
#include <math.h>
#include <string.h>

@@TYPES@@

#define IKRR_MAX_SOLUTIONS 32

static const double IKRR_ALPHA_COS[6] = @@COSALPHA@@;
static const double IKRR_ALPHA_SIN[6] = @@SINALPHA@@;
static const double IKRR_A[6] = @@A@@;      /* link lengths */
static const double IKRR_D[6] = @@D@@;      /* link offsets */
static const double IKRR_OFFSET[6] = @@OFFSET@@;  /* joint angle offsets, theta = sign * q + offset */
static const double IKRR_SIGN[6] = @@SIGN@@;
#define IKRR_K @@K@@        /* the joint whose half angle is the variable of the matrix polynomial */
#define IKRR_LEFT0 @@LEFT0@@   /* the two joints whose monomials are on the left */
#define IKRR_LEFT1 @@LEFT1@@
#define IKRR_RIGHT0 @@RIGHT0@@  /* and on the right */
#define IKRR_RIGHT1 @@RIGHT1@@
#define IKRR_DROP @@DROP@@      /* the joint that does not appear in the equations */
/* sin/cos monomials of the left joints to half angle monomials, 9-by-9 */
static const double IKRR_W[81] = @@W@@;
/* where the monomial j of the nine goes in the 12-vector after multiplying by x_u */
static const int IKRR_SHIFT[9] = @@SHIFT@@;

@@KERNEL@@

/* The matrices of the fourteen equations for the pose T, 4x4 row-major. */
static void ikrr_matrices(const double *T, double *Psin, double *Pcos, double *Pone, double *Q) {
@@TDECL@@
    memset(Psin, 0, 14 * 9 * sizeof(double));
    memset(Pcos, 0, 14 * 9 * sizeof(double));
    memset(Pone, 0, 14 * 9 * sizeof(double));
    memset(Q, 0, 14 * 8 * sizeof(double));
@@MATRICES@@
}

/* C = A B, with A m-by-k and B k-by-n. */
static void ikrr_matmul(const double *A, int m, int k, const double *B, int n, double *C) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) sum += A[i * k + l] * B[l * n + j];
            C[i * n + j] = sum;
        }
    }
}

/* The 12x12 block from six equations, and the same multiplied by x_u. */
static void ikrr_embed(const double *E, double *M) {
    memset(M, 0, 144 * sizeof(double));
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 9; ++c) {
            M[r * 12 + c] = E[r * 9 + c];
            M[(6 + r) * 12 + IKRR_SHIFT[c]] += E[r * 9 + c];
        }
    }
}

/* C = A B for 4x4 row-major matrices. */
static void ikrr_mul4(const double *A, const double *B, double *C) {
    ikrr_matmul(A, 4, 4, B, 4, C);
}

/* The link transform of joint i, and its derivative with respect to theta. */
static void ikrr_dh(double theta, int i, double *A) {
    double ca = IKRR_ALPHA_COS[i], sa = IKRR_ALPHA_SIN[i];
    double s = sin(theta), c = cos(theta);
    A[0] = c;  A[1] = -s * ca; A[2] = s * sa;   A[3] = IKRR_A[i] * c;
    A[4] = s;  A[5] = c * ca;  A[6] = -c * sa;  A[7] = IKRR_A[i] * s;
    A[8] = 0;  A[9] = sa;      A[10] = ca;      A[11] = IKRR_D[i];
    A[12] = 0; A[13] = 0;      A[14] = 0;       A[15] = 1;
}

static void ikrr_dh_derivative(double theta, int i, double *A) {
    double ca = IKRR_ALPHA_COS[i], sa = IKRR_ALPHA_SIN[i];
    double s = sin(theta), c = cos(theta);
    A[0] = -s; A[1] = -c * ca; A[2] = c * sa;  A[3] = -IKRR_A[i] * s;
    A[4] = c;  A[5] = -s * ca; A[6] = s * sa;  A[7] = IKRR_A[i] * c;
    for (int j = 8; j < 16; ++j) A[j] = 0.0;
}

static void ikrr_fkine(const double *theta, double *T) {
    double L[16], P[16];
    for (int j = 0; j < 16; ++j) T[j] = (j % 5 == 0) ? 1.0 : 0.0;
    for (int i = 0; i < 6; ++i) {
        ikrr_dh(theta[i], i, L);
        ikrr_mul4(T, L, P);
        memcpy(T, P, sizeof(P));
    }
}

/* The largest difference between the top three rows of two 4x4 matrices. */
static double ikrr_error(const double *A, const double *B) {
    double e = 0.0;
    for (int j = 0; j < 12; ++j) {
        double d = fabs(A[j] - B[j]);
        if (!(d <= e)) e = d;  /* also catches NaN */
    }
    return e;
}

/* Gauss-Newton steps on the joint angles, to polish a candidate. */
static void ikrr_refine(double *theta, const double *T) {
    for (int iteration = 0; iteration < 3; ++iteration) {
        double links[6][16], prefix[7][16], residual[12], J[72], Qf[144], dtheta[6];
        for (int j = 0; j < 16; ++j) prefix[0][j] = (j % 5 == 0) ? 1.0 : 0.0;
        for (int i = 0; i < 6; ++i) {
            ikrr_dh(theta[i], i, links[i]);
            ikrr_mul4(prefix[i], links[i], prefix[i + 1]);
        }
        double biggest = 0.0;
        for (int j = 0; j < 12; ++j) {
            residual[j] = prefix[6][j] - T[j];
            if (fabs(residual[j]) > biggest) biggest = fabs(residual[j]);
        }
        if (biggest < 1e-13) return;
        double suffix[16], dA[16], tmp[16], dT[16], next[16];
        for (int j = 0; j < 16; ++j) suffix[j] = (j % 5 == 0) ? 1.0 : 0.0;
        for (int i = 5; i >= 0; --i) {
            ikrr_dh_derivative(theta[i], i, dA);
            ikrr_mul4(prefix[i], dA, tmp);
            ikrr_mul4(tmp, suffix, dT);
            for (int j = 0; j < 12; ++j) J[j * 6 + i] = dT[j];
            ikrr_mul4(links[i], suffix, next);
            memcpy(suffix, next, sizeof(next));
        }
        ikrr_qr(J, 12, 6, Qf);
        ikrr_qr_solve(J, Qf, 12, 6, residual, dtheta);
        for (int i = 0; i < 6; ++i) theta[i] -= dtheta[i];
    }
}

/* The inverse of a rigid transform. */
static void ikrr_inverse(const double *A, double *B) {
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) B[i * 4 + j] = A[j * 4 + i];
        B[i * 4 + 3] = -(B[i * 4 + 0] * A[3] + B[i * 4 + 1] * A[7] + B[i * 4 + 2] * A[11]);
    }
    B[12] = 0; B[13] = 0; B[14] = 0; B[15] = 1;
}

static int ikrr_same(const double *q1, const double *q2) {
    for (int i = 0; i < 6; ++i) {
        double d = q1[i] - q2[i];
        if (!(fabs(atan2(sin(d), cos(d))) < 1e-5)) return 0;
    }
    return 1;
}

/* All the joint configurations that reach the pose T, 4x4 row-major, written to
 * solutions.  Returns 0 if there are any, 1 if there are none because a joint was
 * undetermined, a singularity, and 2 if the pose cannot be reached. */
static int ikrr_solve(const double *T, double tol, double (*solutions)[6], int *count, int max_solutions) {
    double Psin[14 * 9], Pcos[14 * 9], Pone[14 * 9], Q[14 * 8], Qf[14 * 14];
    double Es[6 * 9], Ec[6 * 9], Eo[6 * 9], Em[6 * 9], E[6 * 9], M0[144], M1[144], M2[144];
    ikrr_matrices(T, Psin, Pcos, Pone, Q);

    /* eliminate the monomials of the right hand joints with the null space of Q */
    double R[14 * 8];
    memcpy(R, Q, sizeof(R));
    ikrr_qr(R, 14, 8, Qf);
    for (int r = 0; r < 6; ++r) {
        for (int c = 0; c < 9; ++c) {
            double sn = 0.0, cs = 0.0, on = 0.0;
            for (int i = 0; i < 14; ++i) {
                double n = Qf[i * 14 + 8 + r];
                sn += n * Psin[i * 9 + c];
                cs += n * Pcos[i * 9 + c];
                on += n * Pone[i * 9 + c];
            }
            Es[r * 9 + c] = sn; Ec[r * 9 + c] = cs; Eo[r * 9 + c] = on;
        }
    }
    for (int j = 0; j < 54; ++j) Em[j] = Eo[j] - Ec[j];
    ikrr_matmul(Em, 6, 9, IKRR_W, 9, E);
    ikrr_embed(E, M2);
    for (int j = 0; j < 54; ++j) Em[j] = 2.0 * Es[j];
    ikrr_matmul(Em, 6, 9, IKRR_W, 9, E);
    ikrr_embed(E, M1);
    for (int j = 0; j < 54; ++j) Em[j] = Eo[j] + Ec[j];
    ikrr_matmul(Em, 6, 9, IKRR_W, 9, E);
    ikrr_embed(E, M0);

    /* the eigenvalues x of (M2 x^2 + M1 x + M0) v = 0, from a companion matrix */
    double lead[144], B[12 * 24], C[24 * 24], wr[24], wi[24];
    memcpy(lead, M2, sizeof(lead));
    for (int i = 0; i < 12; ++i) {
        for (int j = 0; j < 12; ++j) { B[i * 24 + j] = M0[i * 12 + j]; B[i * 24 + 12 + j] = M1[i * 12 + j]; }
    }
    *count = 0;
    if (!ikrr_lu_solve(lead, 12, B, 24)) return 1;
    memset(C, 0, sizeof(C));
    for (int i = 0; i < 12; ++i) {
        C[i * 24 + 12 + i] = 1.0;
        for (int j = 0; j < 24; ++j) C[(12 + i) * 24 + j] = -B[i * 24 + j];
    }
    if (!ikrr_eig(C, 24, wr, wi)) return 1;

    int ambiguous = 0;
    for (int e = 0; e < 24 && *count < max_solutions; ++e) {
        double x = wr[e];
        if (!(isfinite(x) && fabs(x) < 1e8 && fabs(wi[e]) < 1e-6 * (1.0 + fabs(x)))) continue;
        double Mx[144], s[12], V[144], v[12];
        for (int j = 0; j < 144; ++j) Mx[j] = M2[j] * x * x + M1[j] * x + M0[j];
        ikrr_svd(Mx, 12, s, V);
        if (s[10] < 1e-9 * s[0]) { ++ambiguous; continue; }  /* not a one dimensional null space */
        if (fabs(V[8 * 12 + 11]) < 1e-12) continue;
        for (int i = 0; i < 12; ++i) v[i] = V[i * 12 + 11] / V[8 * 12 + 11];

        double theta[6] = {0, 0, 0, 0, 0, 0};
        theta[IKRR_K] = 2.0 * atan(x);
        theta[IKRR_LEFT0] = 2.0 * atan(v[5]);
        theta[IKRR_LEFT1] = 2.0 * atan(v[7]);
        double su = sin(theta[IKRR_LEFT0]), cu = cos(theta[IKRR_LEFT0]);
        double sv = sin(theta[IKRR_LEFT1]), cv = cos(theta[IKRR_LEFT1]);
        double left[9] = {su * sv, su * cv, cu * sv, cu * cv, su, cu, sv, cv, 1.0};
        double sk = sin(theta[IKRR_K]), ck = cos(theta[IKRR_K]), rhs[14], right[8];
        for (int r = 0; r < 14; ++r) {
            rhs[r] = 0.0;
            for (int c = 0; c < 9; ++c) rhs[r] += (sk * Psin[r * 9 + c] + ck * Pcos[r * 9 + c] + Pone[r * 9 + c]) * left[c];
        }
        ikrr_qr_solve(R, Qf, 14, 8, rhs, right);
        theta[IKRR_RIGHT0] = atan2(right[4], right[5]);
        theta[IKRR_RIGHT1] = atan2(right[6], right[7]);

        /* the dropped joint, from  A_drop = (A_0..A_drop-1)^-1 T (A_drop+1..A_5)^-1 */
        double pre[16], post[16], L[16], P[16], inv[16], mid[16], D[16];
        for (int j = 0; j < 16; ++j) { pre[j] = (j % 5 == 0) ? 1.0 : 0.0; post[j] = pre[j]; }
        for (int i = 0; i < IKRR_DROP; ++i) { ikrr_dh(theta[i], i, L); ikrr_mul4(pre, L, P); memcpy(pre, P, sizeof(P)); }
        for (int i = IKRR_DROP + 1; i < 6; ++i) { ikrr_dh(theta[i], i, L); ikrr_mul4(post, L, P); memcpy(post, P, sizeof(P)); }
        double T4[16];
        memcpy(T4, T, 12 * sizeof(double));
        T4[12] = 0; T4[13] = 0; T4[14] = 0; T4[15] = 1;
        ikrr_inverse(pre, inv);
        ikrr_mul4(inv, T4, mid);
        ikrr_inverse(post, inv);
        ikrr_mul4(mid, inv, D);
        theta[IKRR_DROP] = atan2(D[4], D[0]);

        /* a candidate that is close is polished, since the roots are only
         * accurate to about the conditioning of the problem times round-off */
        double fk[16];
        ikrr_fkine(theta, fk);
        if (ikrr_error(fk, T) < 1e-3) { ikrr_refine(theta, T); ikrr_fkine(theta, fk); }
        if (!(ikrr_error(fk, T) < tol)) continue;

        double q[6];
        for (int i = 0; i < 6; ++i) {
            double a = IKRR_SIGN[i] * (theta[i] - IKRR_OFFSET[i]);
            q[i] = atan2(sin(a), cos(a));
        }
        int duplicate = 0;
        for (int k = 0; k < *count; ++k) if (ikrr_same(q, solutions[k])) duplicate = 1;
        if (!duplicate) { memcpy(solutions[*count], q, sizeof(q)); ++*count; }
    }
    if (*count > 0) return 0;
    return ambiguous ? 1 : 2;
}

@@FRONT@@
"""

_FRONT_C = """\
/* T is the 4x4 end-effector pose, row-major.  Solutions are written to
 * solutions[0..*count-1], each IKINE_DOF joint values in (-pi, pi]. */
ik_status @@NAME@@(const double T[16], double tol,
                  double solutions[IKINE_MAX_SOLUTIONS][IKINE_DOF], int *count) {
    return (ik_status)ikrr_solve(T, tol, solutions, count, IKINE_MAX_SOLUTIONS);
}"""

_TYPES_C = """\
typedef enum {
    IK_OK = 0,          /* solutions found */
    IK_SINGULAR = 1,    /* a joint was undetermined, so there are none or some are missing */
    IK_UNREACHABLE = 2  /* no solution, the pose is outside the workspace */
} ik_status;

#define IKINE_DOF 6
#define IKINE_MAX_SOLUTIONS 32"""

_FRONT_CXX = """\
// T is the 4x4 end-effector pose, row-major.
inline IkResult @@NAME@@(const double* T, double tol = 1e-6) {
    double solutions[IKRR_MAX_SOLUTIONS][6];
    int count = 0;
    int status = ikrr_solve(T, tol, solutions, &count, IKRR_MAX_SOLUTIONS);
    IkResult result;
    for (int i = 0; i < count; ++i) {
        result.solutions.push_back({solutions[i][0], solutions[i][1], solutions[i][2],
                                    solutions[i][3], solutions[i][4], solutions[i][5]});
    }
    result.status = (status == 0) ? IkStatus::Ok : (status == 1) ? IkStatus::Singular : IkStatus::Unreachable;
    return result;
}"""

_TYPES_CXX = """\
#include <array>
#include <vector>

enum class IkStatus {
    Ok = 0,          // solutions found
    Singular = 1,    // a joint was undetermined, so there are none or some are missing
    Unreachable = 2  // no solution, the pose is outside the workspace
};

struct IkResult {
    std::vector<std::array<double, 6>> solutions;
    IkStatus status = IkStatus::Unreachable;
};"""


_GENERATED = """\
@@BANNER@@

import enum
import math
from typing import NamedTuple

import numpy as np

try:  # QZ copes with a singular leading matrix, NumPy alone usually does not
    import scipy.linalg
except ImportError:
    scipy = None


class IKStatus(enum.Enum):
    OK = 0  # solutions found
    SINGULAR = 1  # candidates were dropped because a joint was undetermined
    UNREACHABLE = 2  # no solution, the pose is outside the workspace


class IKResult(NamedTuple):
    solutions: list[tuple[float, ...]]
    status: IKStatus


ALPHA = @@ALPHA@@  # link twists
A = @@A@@  # link lengths
D = @@D@@  # link offsets
OFFSET = np.array(@@OFFSET@@)  # joint angle offsets, theta = SIGN * q + OFFSET
SIGN = np.array(@@SIGN@@)
K = @@K@@  # the joint whose half angle is the variable of the matrix polynomial
LEFT = @@LEFT@@  # the two joints whose monomials are on the left
RIGHT = @@RIGHT@@  # and on the right
DROP = @@DROP@@  # the joint that does not appear in the equations
W = np.array(@@W@@)  # sin/cos monomials to half angle monomials
MON9 = @@MON9@@  # exponents of x_u, x_v in the nine monomials
MON12 = MON9 + [(3, 2), (3, 1), (3, 0)]


def _matrices(T):
    \"\"\"The matrices of the fourteen equations for the pose T, a 4x4 array.\"\"\"
    T0, T1, T2, T3 = T[0][0], T[0][1], T[0][2], T[0][3]
    T4, T5, T6, T7 = T[1][0], T[1][1], T[1][2], T[1][3]
    T8, T9, T10, T11 = T[2][0], T[2][1], T[2][2], T[2][3]
@@MATRICES@@
    return Psin, Pcos, Pone, Q


def _embed(E):
    \"\"\"The 12x12 block from six equations, and the same multiplied by x_u.\"\"\"
    M = np.zeros((12, 12))
    M[:6, :9] = E
    for j, (i, k) in enumerate(MON9):
        M[6:, MON12.index((i + 1, k))] += E[:, j]
    return M


def _polynomial(T):
    \"\"\"The matrices, and the 12x12 matrices of the polynomial M2 x^2 + M1 x + M0.\"\"\"
    Psin, Pcos, Pone, Q = _matrices(T)
    # eliminate the monomials of the right hand joints with the null space of Q
    U, _, _ = np.linalg.svd(Q, full_matrices=True)
    N = U[:, -6:]
    Es, Ec, Eo = N.T @ Psin, N.T @ Pcos, N.T @ Pone
    M2, M1, M0 = _embed((Eo - Ec) @ W), _embed(2 * Es @ W), _embed((Eo + Ec) @ W)
    return Psin, Pcos, Pone, Q, M0, M1, M2


def condition(T):
    \"\"\"Condition number of the leading matrix for the pose T, large is unreliable.\"\"\"
    return float(np.linalg.cond(_polynomial(np.asarray(T, dtype=float))[-1]))


def _eigenvalues(M0, M1, M2):
    \"\"\"Eigenvalues x of (M2 x^2 + M1 x + M0) v = 0.\"\"\"
    Z, I = np.zeros((12, 12)), np.eye(12)
    if scipy is not None:
        return scipy.linalg.eig(
            np.block([[Z, I], [-M0, -M1]]), np.block([[I, Z], [Z, M2]]), right=False
        )
    companion = np.block([[Z, I], [-np.linalg.solve(M2, M0), -np.linalg.solve(M2, M1)]])
    return np.linalg.eigvals(companion)


def _clean(x):
    for target in (0.0, 1.0, -1.0):
        if abs(x - target) < 1e-12:
            return target
    return x


def _dh(theta, i):
    ca, sa = _clean(math.cos(ALPHA[i])), _clean(math.sin(ALPHA[i]))
    s, c = math.sin(theta), math.cos(theta)
    return np.array(
        [
            [c, -s * ca, s * sa, A[i] * c],
            [s, c * ca, -c * sa, A[i] * s],
            [0, sa, ca, D[i]],
            [0, 0, 0, 1],
        ]
    )


def _dh_derivative(theta, i):
    ca, sa = _clean(math.cos(ALPHA[i])), _clean(math.sin(ALPHA[i]))
    s, c = math.sin(theta), math.cos(theta)
    return np.array(
        [
            [-s, -c * ca, c * sa, -A[i] * s],
            [c, -s * ca, s * sa, A[i] * c],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )


def _fkine(theta):
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(theta[i], i)
    return T


def _refine(theta, T, iterations=3):
    \"\"\"Gauss-Newton steps on the joint angles, to polish a candidate.\"\"\"
    for _ in range(iterations):
        links = [_dh(theta[i], i) for i in range(6)]
        prefix = [np.eye(4)]
        for link in links:
            prefix.append(prefix[-1] @ link)
        residual = (prefix[-1] - T)[:3, :].ravel()
        if np.abs(residual).max() < 1e-13:
            break
        J = np.empty((12, 6))
        suffix = np.eye(4)
        for i in range(5, -1, -1):
            J[:, i] = (prefix[i] @ _dh_derivative(theta[i], i) @ suffix)[:3, :].ravel()
            suffix = links[i] @ suffix
        theta = theta - np.linalg.lstsq(J, residual, rcond=None)[0]
    return theta


def _same(q1, q2, tol=1e-5):
    d = q1 - q2
    return bool(np.all(np.abs(np.arctan2(np.sin(d), np.cos(d))) < tol))


def @@NAME@@(T, tol: float = 1e-6) -> IKResult:
    \"\"\"
    All joint configurations (q1..q6) that reach the end-effector pose T

    T is a 4x4 array.  The status distinguishes a pose that is outside the
    workspace from one where candidates were dropped because a joint was
    undetermined, a singularity.
    \"\"\"
    T = np.asarray(T, dtype=float)
    Psin, Pcos, Pone, Q, M0, M1, M2 = _polynomial(T)

    solutions = []
    ambiguous = 0
    for e in _eigenvalues(M0, M1, M2):
        if not (np.isfinite(e) and abs(e.imag) < 1e-6 * (1 + abs(e.real)) and abs(e) < 1e8):
            continue
        x = e.real
        _, sv, Vt = np.linalg.svd(M2 * x**2 + M1 * x + M0)
        if sv[-2] < 1e-9 * sv[0]:  # not a one dimensional null space
            ambiguous += 1
            continue
        v = Vt[-1]
        if abs(v[8]) < 1e-12:
            continue
        v = v / v[8]
        theta = np.zeros(6)
        theta[K] = 2 * np.arctan(x)
        theta[LEFT[0]], theta[LEFT[1]] = 2 * np.arctan(v[5]), 2 * np.arctan(v[7])

        su, cu = math.sin(theta[LEFT[0]]), math.cos(theta[LEFT[0]])
        sv_, cv = math.sin(theta[LEFT[1]]), math.cos(theta[LEFT[1]])
        v_left = np.array([su * sv_, su * cv, cu * sv_, cu * cv, su, cu, sv_, cv, 1.0])
        rhs = (math.sin(theta[K]) * Psin + math.cos(theta[K]) * Pcos + Pone) @ v_left
        v_right, *_ = np.linalg.lstsq(Q, rhs, rcond=None)
        theta[RIGHT[0]] = math.atan2(v_right[4], v_right[5])
        theta[RIGHT[1]] = math.atan2(v_right[6], v_right[7])

        # the dropped joint from  A_drop = (A_0..A_drop-1)^-1 T (A_drop+1..A_5)^-1
        pre, post = np.eye(4), np.eye(4)
        for i in range(DROP):
            pre = pre @ _dh(theta[i], i)
        for i in range(DROP + 1, 6):
            post = post @ _dh(theta[i], i)
        M = np.linalg.inv(pre) @ T @ np.linalg.inv(post)
        theta[DROP] = math.atan2(M[1, 0], M[0, 0])

        # a candidate that is close is polished, since the roots are only
        # accurate to about the conditioning of the problem times round-off
        if np.abs(_fkine(theta) - T).max() < 1e-3:
            theta = _refine(theta, T)
        if np.abs(_fkine(theta) - T).max() < tol:
            q = SIGN * (theta - OFFSET)
            q = np.arctan2(np.sin(q), np.cos(q))
            if not any(_same(q, other) for other in solutions):
                solutions.append(q)

    if solutions:
        status = IKStatus.OK
    elif ambiguous:
        status = IKStatus.SINGULAR
    else:
        status = IKStatus.UNREACHABLE
    return IKResult([tuple(float(x) for x in q) for q in solutions], status)
"""


def _derive(alpha, a, d, k: int):
    """
    Derive the matrices of the 14 equations for a robot, symbolically

    :param alpha: twist angles of the six links
    :param a: link lengths
    :param d: link offsets
    :param k: which split, see the module documentation
    :returns: ``(functions, matrices)``.  The matrices ``P_sin``, ``P_cos``,
        ``P_one`` (14x9) and ``Q`` (14x8), such that
        ``(sin(q_k) P_sin + cos(q_k) P_cos + P_one) @ v_left = Q @ v_right``,
        are symbolic in the 12 elements of the pose ``T0 ... T11``, and
        ``functions`` are those four matrices as functions of the pose elements.
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
    matrices = (Psin, Pcos, Pone, Q)
    return [sp.lambdify(t, m, "numpy") for m in matrices], matrices


_DERIVED: dict[tuple, tuple] = {}  # in-process cache, a derivation takes seconds


def _poses(T: SE3 | np.ndarray) -> tuple[np.ndarray, bool]:
    # poses as an array (N, 4, 4), and whether it was a single pose
    if isinstance(T, SE3):
        return np.asarray(T.A, dtype=float).reshape(-1, 4, 4), len(T) == 1
    A = np.asarray(T, dtype=float)
    if A.ndim == 2:
        return A.reshape(1, 4, 4), True
    if A.ndim == 3 and A.shape[1:] == (4, 4):
        return A, False
    raise ValueError("T must be an SE3, a 4x4 array or an array (N, 4, 4)")


class AnalyticIK(ABC):
    r"""
    The interface of an analytic inverse kinematics solver for a robot

    :param robot: the robot, with standard DH links and six joints

    Made by :meth:`~roboticstoolbox.robot.DHRobot.DHRobot.analytic_ik`, which chooses the method that suits the
    arm: :class:`ClosedFormIK` for an arm with a spherical wrist or three parallel
    axes, and :class:`General6RIK` for the others.  The equations are derived
    symbolically when the solver is made, which takes seconds, and the solver can
    then be used for any number of poses, or written out as source code that has
    no dependency on the Toolbox, see :meth:`~roboticstoolbox.robot.symbolic_ik_rr.AnalyticIK.generate`.

    Two methods solve a problem, and return an :class:`~roboticstoolbox.robot.IK.IKSolution`
    as the numerical solvers do:

    - :meth:`solve` gives *all* the solutions for one pose, as the rows of ``q``
    - :meth:`ikine` gives *one* solution for each pose, the nearest to ``q0``, for
      a pose or a trajectory, as :meth:`~roboticstoolbox.robot.Robot.Robot.ikine_LM` does

    .. versionadded:: 1.5.0
    """

    #: name of the method, for messages
    method = ""

    def __init__(self, robot: DHRobot):
        self.robot = robot
        self.revolute = np.array([link.isrevolute for link in robot.links])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.robot.name!r})"

    def __str__(self) -> str:
        structure = classify(self.robot)

        def joints(first: tuple[int, ...]) -> str:
            # the joints of each triple, for example "4-6"
            return ", ".join(f"{i}-{i + 2}" for i in first) or "none"

        return (
            f"{type(self).__name__} for {self.robot.name}: {self.method}\n"
            f"  joints meeting at a point: {joints(structure.spherical)}\n"
            f"  joints with parallel axes: {joints(structure.parallel)}"
        )

    def _dh_pose(self, T: np.ndarray) -> np.ndarray:
        # the pose of the last DH frame, with the base and tool transforms removed
        return np.linalg.inv(self.robot.base.A) @ T @ np.linalg.inv(self.robot.tool.A)

    def _alternatives(self, T: np.ndarray, joint_limits: bool) -> tuple[np.ndarray, str]:
        # all the solutions for one 4x4 pose, with revolute angles shifted by whole
        # turns into the joint limits, and a reason if there are none
        sols = self._solve_pose(self._dh_pose(T))
        q = sols.q
        if len(q) == 0:
            reason = {
                IKStatus.UNREACHABLE: "the pose is outside the workspace",
                IKStatus.SINGULAR: "the pose is singular and no solution was found",
            }.get(sols.status, "no solution")
            return q, reason

        qlim = self.robot.qlim
        q = np.array([normalise_q(row, qlim, self.revolute) for row in q])
        if joint_limits:
            inside = np.array([within_limits(row, qlim) for row in q])
            if not inside.any():
                return q[:0], f"all {len(q)} solutions violate the joint limits"
            q = q[inside]
        return q, ""

    def solve(self, T: SE3 | np.ndarray, joint_limits: bool = True) -> IKSolution:
        """
        All the solutions for one pose

        :param T: end-effector pose, an SE3 or a 4x4 array
        :param joint_limits: discard solutions that violate the joint limits.
            Revolute angles are first shifted by whole turns to fit the limits,
            as the numerical solvers do.
        :returns: the solutions as the rows of ``q``, shape (m, n), with ``success``
            true if there is at least one.  Otherwise ``q`` is empty and ``reason``
            says why.
        :raises ValueError: if ``T`` is more than one pose, use :meth:`ikine`

        There can be several solutions for a pose, up to 8 for an arm with a
        spherical wrist and 16 for the general case, and all are returned.

        .. runblock:: pycon

            >>> import roboticstoolbox as rtb
            >>> puma = rtb.models.DH.Puma560()
            >>> ik = puma.analytic_ik()
            >>> sol = ik.solve(puma.fkine(puma.qn))
            >>> len(sol)
        """
        poses, _ = _poses(T)
        if len(poses) != 1:
            raise ValueError("solve takes a single pose, use ikine for a trajectory")
        q, reason = self._alternatives(poses[0], joint_limits)
        return IKSolution(q, success=len(q) > 0, reason=reason)

    def ikine(
        self,
        T: SE3 | np.ndarray,
        q0: ArrayLike | None = None,
        joint_limits: bool = True,
        weights: ArrayLike | None = None,
    ) -> IKSolution:
        """
        One solution for each pose, the nearest to a configuration

        :param T: end-effector pose, an SE3 or a 4x4 array, or a trajectory of
            poses, an SE3 with N poses or an array (N, 4, 4)
        :param q0: configuration to be near, default all zeros.  For a trajectory
            it is used for the first pose, and each later pose is solved nearest
            to the solution of the one before, so the path stays on one branch
            where it can.
        :param joint_limits: reject solutions that violate the joint limits
        :param weights: weight for each joint in the distance from ``q0``, default
            1.  A weight of zero ignores that joint.
        :returns: ``q`` of shape (n,) for a pose, or (N, n) for a trajectory.  If
            any pose has no solution ``success`` is false, ``reason`` says why and
            the rows for those poses are NaN.
        :raises ValueError: if ``q0`` has the wrong length

        The distance is the weighted norm of the joint differences, with revolute
        differences wrapped to :math:`(-\\pi, \\pi]`, and the angles returned are
        the equivalent ones nearest to ``q0``, if they are within the limits.
        ``iterations`` and ``searches`` are zero, as there is no iteration, and
        ``residual`` is the largest error of the forward kinematics of the
        solutions, as the norm of the 6-vector of the error, see
        :func:`~spatialmath.base.tr2delta`.
        """
        poses, single = _poses(T)
        n = self.robot.n
        q_prev = np.zeros(n) if q0 is None else check_q0(q0, n)
        assert q_prev is not None  # nosec B101: narrows the type for mypy
        qlim = self.robot.qlim

        Q = np.full((len(poses), n), np.nan)
        reason = ""
        residual = 0.0
        for k, Tk in enumerate(poses):
            rows, why = self._alternatives(Tk, joint_limits)
            if len(rows) == 0:
                if not reason:
                    reason = why if single else f"pose {k}: {why}"
                continue
            best = IKSolutions(rows, IKStatus.OK, self.revolute).nearest(q_prev, weights)
            assert best is not None  # nosec B101: narrows the type for mypy
            # the equivalent angles nearest to the previous configuration, if allowed
            turns = 2 * np.pi * np.round((q_prev - best) / (2 * np.pi))
            moved = np.where(self.revolute, best + turns, best)
            if joint_limits:
                ok = (moved >= qlim[0]) & (moved <= qlim[1])
                moved = np.where(ok, moved, best)
            Q[k] = q_prev = moved
            error = tr2delta(self.robot.fkine(moved).A, Tk)
            residual = max(residual, float(np.linalg.norm(error)))

        success = not np.isnan(Q).any()
        return IKSolution(
            Q[0] if single else Q, success, residual=residual, reason=reason
        )

    @abstractmethod
    def _solve_pose(self, T: np.ndarray) -> IKSolutions:
        """
        All the joint configurations that reach a pose

        :param T: end-effector pose, a 4x4 array
        :returns: the solutions and a status, see :class:`IKSolutions`
        """

    @abstractmethod
    def generate(self, language: str = "python", name: str = "ikine") -> str:
        """
        Source code for the solution for this robot

        :param language: the language of the code
        :param name: name of the generated function
        :returns: source code, which starts with a header saying where it came from
        :raises NotImplementedError: for a language that is not supported
        """

    def validate(self, trials: int = 100, seed: int = 1) -> float:
        """
        Fraction of random poses for which the original configuration is found

        :param trials: number of random configurations to try
        :param seed: seed for the random configurations
        :returns: a value between 0 and 1.  A method that works gives 1, or very
            nearly, and one that does not apply to this robot gives a low value.
        """
        rng = np.random.default_rng(seed)
        n = self.robot.n
        hits = 0
        for _ in range(trials):
            q = np.where(
                self.revolute, rng.uniform(-np.pi, np.pi, n), rng.uniform(0.2, 0.6, n)
            )
            sols = self._solve_pose(self._dh_pose(self.robot.fkine(q).A))
            if len(sols):
                d = sols.q - q
                d[:, self.revolute] = np.arctan2(
                    np.sin(d[:, self.revolute]), np.cos(d[:, self.revolute])
                )
                hits += bool(np.any(np.all(np.abs(d) < 1e-5, axis=1)))
        return hits / trials


class General6RIK(AnalyticIK):
    r"""
    Inverse kinematics of a general 6R arm

    :param robot: robot with six revolute joints and standard DH links
    :param k: which split of the loop closure to use, 0, 1 or 2, see the module
        documentation.  By default the best of the three is chosen numerically.
    :param seed: seed for the random poses used to choose ``k``
    :raises ValueError: if the robot does not have six revolute joints

    The first time a robot is used the equations are derived symbolically, which
    takes a few seconds per split.  The solver is Python code, which is generated
    and then run, see :meth:`~roboticstoolbox.robot.symbolic_ik_rr.AnalyticIK.generate`.

    ``solve(T)`` returns all the solutions for a pose, and ``ikine(T, q0)`` one for each pose,
    as an :class:`~roboticstoolbox.robot.IK.IKSolution`.

    .. note::

        The method degenerates for some geometries, for example a spherical
        wrist.  Check with ``validate``, or use ``classify``.
    """

    method = "general 6R"

    def __init__(self, robot: DHRobot, k: int | None = None, seed: int = 0):
        if robot.n != 6 or not all(link.isrevolute for link in robot.links):
            raise ValueError("a robot with six revolute joints is required")
        if any(link.mdh for link in robot.links):
            raise NotImplementedError("modified DH links are not supported")
        super().__init__(robot)
        self.alpha = [float(link.alpha) for link in robot.links]
        self.a = [float(link.a) for link in robot.links]
        self.d = [float(link.d) for link in robot.links]
        self.offset = np.array([float(link.offset) for link in robot.links])
        self.sign = np.array([-1.0 if link.isflip else 1.0 for link in robot.links])
        self.condition: dict[int, float] = {}
        self._modules: dict[int, dict] = {}
        self.k = self._choose(seed) if k is None else k
        self._ikine = self._module(self.k)["ikine"]

    # --- set up ---

    def _derived(self, k: int) -> tuple:
        key = (tuple(self.alpha), tuple(self.a), tuple(self.d), k)
        if key not in _DERIVED:
            _DERIVED[key] = _derive(self.alpha, self.a, self.d, k)
        return _DERIVED[key]

    def _module(self, k: int) -> dict:
        """The generated code for split ``k``, run, as its namespace."""
        if k not in self._modules:
            namespace: dict = {}
            exec(self._source(k, "ikine"), namespace)  # noqa: S102
            self._modules[k] = namespace
        return self._modules[k]

    def _random_poses(self, count: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed)
        qs = rng.uniform(-np.pi, np.pi, (count, 6))
        return [(q, self.robot.fkine(q).A) for q in qs]

    def _choose(self, seed: int, count: int = 20) -> int:
        """The split that recovers the most poses, then has the best conditioning."""
        poses = self._random_poses(count, seed)
        scores = {}
        for k in (0, 1, 2):
            module = self._module(k)
            hits = sum(
                any(self._same(np.array(s), q) for s in module["ikine"](T).solutions)
                for q, T in poses
            )
            self.condition[k] = float(np.median([module["condition"](T) for _, T in poses]))
            scores[k] = (-hits, self.condition[k])
        return min(scores, key=lambda k: scores[k])

    @staticmethod
    def _same(q1: np.ndarray, q2: np.ndarray, tol: float = 1e-5) -> bool:
        d = q1 - q2
        return bool(np.all(np.abs(np.arctan2(np.sin(d), np.cos(d))) < tol))

    # --- solving ---

    def _solve_pose(self, T: np.ndarray) -> IKSolutions:
        """
        All the joint configurations that reach a pose

        :param T: end-effector pose, a 4x4 array
        :returns: the solutions, wrapped to (-pi, pi].  The status is ``OK`` if
            there are any.  If there are none it is ``SINGULAR`` if a candidate
            had to be dropped because its null space was ambiguous, otherwise
            ``UNREACHABLE``.
        """
        result = self._ikine(np.asarray(T, dtype=float))
        return IKSolutions.from_generated(result, self.revolute)

    def _source(self, k: int, name: str) -> str:
        """Python source for split ``k``."""
        _, symbolic = self._derived(k)
        entries = [e for m in symbolic for e in m]
        replacements, reduced = sp.cse(entries, symbols=sp.numbered_symbols("x"))
        printer = PythonCodePrinter()
        lines = [f"    {sym} = {printer.doprint(e)}" for sym, e in replacements]
        pos = 0
        for matrix_name, m in zip(("Psin", "Pcos", "Pone", "Q"), symbolic):
            rows = []
            for _ in range(m.rows):
                row = reduced[pos : pos + m.cols]
                pos += m.cols
                rows.append("[" + ", ".join(printer.doprint(e) for e in row) + "]")
            lines.append(f"    {matrix_name} = np.array([" + ",\n        ".join(rows) + "])")
        left, right, drop = _SPLITS[k]
        text = _GENERATED
        for key, value in {
            "ALPHA": repr(self.alpha), "A": repr(self.a), "D": repr(self.d),
            "OFFSET": repr([float(x) for x in self.offset]),
            "SIGN": repr([float(x) for x in self.sign]), "K": repr(k),
            "LEFT": repr(left), "RIGHT": repr(right), "DROP": repr(drop),
            "W": repr(_W.tolist()), "MON9": repr(_MON9),
            "MATRICES": "\n".join(lines), "NAME": name,
            "BANNER": "\n".join(
                _comment(
                    _provenance(
                        _dh_info(self.robot),
                        "general 6R, Raghavan-Roth (symbolic_ik_rr)",
                    ),
                    "python",
                )
            ),
        }.items():  # fmt: skip
            text = text.replace(f"@@{key}@@", value)
        return text

    def generate(self, language: str = "python", name: str = "ikine") -> str:
        """
        Source code for the solution for this robot

        :param language: ``"python"``, ``"c"`` or ``"c++"``
        :param name: name of the generated function
        :returns: Python source code, which needs NumPy, and uses SciPy if it is
            installed
        :raises NotImplementedError: for any other language.  The C and C++ code
            contains its own small linear algebra kernel, and needs no library.

        The link parameters and the symbolic matrices of the fourteen equations
        are constants in the code.  The function ``name(T)`` takes the pose, a
        4x4 array, and returns an ``IKResult`` with the solutions, as a list of
        tuples, and the status.  The code is what :meth:`solve` runs.
        """
        if language == "python":
            return self._source(self.k, name)
        if language in ("c", "c++"):
            return self._source_c(self.k, name, language)
        raise NotImplementedError("Python, C and C++ code can be generated")

    def _source_c(self, k: int, name: str, language: str) -> str:
        """C or C++ source for split ``k``."""
        _, symbolic = self._derived(k)
        entries = [e for m in symbolic for e in m]
        replacements, reduced = sp.cse(entries, symbols=sp.numbered_symbols("x"))
        printer = C99CodePrinter()
        used = set().union(*(e.free_symbols for e in [*reduced, *(r[1] for r in replacements)]))
        lines = [
            f"    const double T{i} = T[{i}];" for i in range(12) if sp.Symbol(f"T{i}", real=True) in used
        ]
        tdecl = "\n".join(lines)
        body = [f"    const double {sym} = {printer.doprint(e)};" for sym, e in replacements]
        pos = 0
        for matrix_name, m in zip(("Psin", "Pcos", "Pone", "Q"), symbolic):
            for r in range(m.rows):
                for c in range(m.cols):
                    e = reduced[pos]
                    pos += 1
                    if e != 0:
                        body.append(f"    {matrix_name}[{r * m.cols + c}] = {printer.doprint(e)};")
        left, right, drop = _SPLITS[k]
        def clist(values) -> str:
            return "{" + ", ".join(repr(float(v)) for v in values) + "}"
        def cleaned(f) -> list[float]:
            return [_clean(f(a)) for a in self.alpha]
        shift = [_MON12.index((i + 1, j)) for i, j in _MON9]
        text = _GENERATED_C
        for key, value in {
            "BANNER": "\n".join(
                _comment(
                    _provenance(
                        _dh_info(self.robot), "general 6R, Raghavan-Roth (symbolic_ik_rr)"
                    ),
                    language,
                )
            ),
            "TYPES": _TYPES_C if language == "c" else _TYPES_CXX,
            "COSALPHA": clist(cleaned(np.cos)), "SINALPHA": clist(cleaned(np.sin)),
            "A": clist(self.a), "D": clist(self.d), "OFFSET": clist(self.offset),
            "SIGN": clist(self.sign), "K": repr(k),
            "LEFT0": repr(left[0]), "LEFT1": repr(left[1]),
            "RIGHT0": repr(right[0]), "RIGHT1": repr(right[1]), "DROP": repr(drop),
            "W": clist(_W.ravel()), "SHIFT": "{" + ", ".join(map(str, shift)) + "}",
            "TDECL": tdecl, "MATRICES": "\n".join(body),
        }.items():  # fmt: skip
            text = text.replace(f"@@{key}@@", value)
        text = text.replace("@@KERNEL@@", _C_KERNEL)
        front = _FRONT_C if language == "c" else _FRONT_CXX
        return text.replace("@@FRONT@@", front).replace("@@NAME@@", name)


class ClosedFormIK(AnalyticIK):
    r"""
    Inverse kinematics from the closed form solution of ``symbolic_ik``

    :param robot: robot with standard DH links and six joints
    :param max_time: give up looking for the symbolic solution after this many
        seconds
    :param verify: check each solution against the forward kinematics
    :raises ValueError: if no closed form solution is found

    The symbolic solution is found once, when the solver is made.  The solution
    is then generated as Python code, which is run to solve, and the same plan
    can be written in other languages by :meth:`~roboticstoolbox.robot.symbolic_ik_rr.AnalyticIK.generate`.
    """

    method = "closed form"

    def __init__(
        self, robot: DHRobot, max_time: float | None = 120.0, verify: bool = True
    ):
        super().__init__(robot)
        self._plan = _make_plan(robot, verify, max_time=max_time)
        self.source = _emit(self._plan, "python", "ikine")
        namespace: dict = {}
        exec(self.source, namespace)  # noqa: S102
        self._ikine = namespace["ikine"]

    def _solve_pose(self, T: np.ndarray) -> IKSolutions:
        """All the joint configurations that reach a pose, see :class:`IKSolutions`."""
        result = self._ikine(np.asarray(T, dtype=float))
        return IKSolutions.from_generated(result, self.revolute)

    def generate(self, language: str = "python", name: str = "ikine") -> str:
        """
        Source code for the closed form solution

        :param language: ``"python"``, ``"c"``, ``"c++"`` or ``"rust"``
        :param name: name of the generated function
        :returns: source code, which needs no library
        :raises ValueError: for any other language
        """
        return _emit(self._plan, language, name)


def make_solver(
    robot: DHRobot,
    trials: int = 100,
    min_success: float = 0.99,
    max_time: float | None = 120.0,
) -> AnalyticIK:
    """
    Choose and build the inverse kinematics method for a robot

    :param robot: robot with standard DH links and six joints
    :param trials: number of random poses used to check the method
    :param min_success: smallest acceptable fraction of those poses for which the
        original configuration is recovered
    :param max_time: give up looking for a closed form solution after this long
    :returns: a :class:`ClosedFormIK` solver for an arm with a spherical wrist or
        three parallel axes, otherwise a :class:`General6RIK` solver
    :raises NotImplementedError: if the arm needs the general method but has a
        prismatic joint
    :raises ValueError: if the method does not recover enough of the poses

    Each method is only good for its own kind of arm, for example the general
    method fails for a spherical wrist, so there is no fallback from one to the
    other.  The method is checked on random poses before it is returned.
    """
    solver: AnalyticIK
    if classify(robot).general:
        prismatic = [i + 1 for i, link in enumerate(robot.links) if link.isprismatic]
        if prismatic:
            joints = ", ".join(str(j) for j in prismatic)
            raise NotImplementedError(
                "this arm has neither a spherical wrist nor three parallel axes, "
                "so it needs the general 6R method, which handles revolute joints "
                f"only, and the arm has prismatic joint(s) {joints}"
            )
        solver = General6RIK(robot)
    else:
        solver = ClosedFormIK(robot, max_time)
    rate = solver.validate(trials)
    if rate < min_success:
        raise ValueError(
            f"the {solver.method} method recovered the configuration for only "
            f"{rate:.0%} of random poses for this robot"
        )
    return solver
