"""
Safety-net tests for the ETS / fknm / frne refactor (Phase 0).

Two tiers of test here, and they check different things:

  Tier 1 ("*Fallback" classes, ``test_matches_c_path`` methods): runs an ETS
  function twice, once via the C extension (the normal, unpatched call) and
  once via the pure-Python fallback (forced by patching the C-calling
  function to raise), and checks they agree. This is only meaningful when
  the C extension is actually built -- if it isn't, the "C path" call
  silently dispatches to Python too (see fknm.py/frne.py), and the
  comparison degrades to Python-vs-itself without any indication. These are
  gated with ``@unittest.skipUnless(_C_AVAILABLE, ...)`` so that case shows
  as an explicit skip, never a false pass.

  Tier 2 ("*Reference" classes): compares whichever path is actually active
  against hardcoded ground-truth values, computed once against the C
  extension (which Tier 1 already shows agrees with Python). These run
  regardless of ``_C_AVAILABLE`` -- they're what actually guarantees a
  pure-Python build (no compiled extension at all, e.g. the pyodide/wasm
  wheel) is numerically correct on its own terms, not merely "consistent
  with itself".

Robot used: Franka Panda (7-DOF) for ETS functions; Puma560 (6-DOF DH) for rne.
"""

import math
import os
import sys
import timeit
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import numpy.testing as nt
import sympy

from roboticstoolbox.ets.fknm import _C_AVAILABLE as _FKNM_C_AVAILABLE
from roboticstoolbox.robot.frne import _C_AVAILABLE as _FRNE_C_AVAILABLE
from roboticstoolbox.models.DH.TwoLink import TwoLink

import roboticstoolbox as rtb
# roboticstoolbox/ets/ETS.py defines a class also called ETS, and
# roboticstoolbox/ets/__init__.py does `from ...ETS import ETS`, which
# rebinds the "ETS" attribute on the roboticstoolbox.ets package to the
# class, shadowing the submodule of the same name. `import a.b.c as x` is
# defined as `import a.b.c; x = a.b.c` — that second step is still
# attribute access, so it hits the same shadowing and also gives the
# class, not the module. sys.modules[...] is a plain dict keyed by the
# literal dotted string, with no getattr involved, so it's the only
# reliably-correct way to get the real module object here.
#
# This only matters for patch() at all because Python 3.10's
# unittest.mock resolves dotted string patch targets via plain getattr
# (falling back to import only on AttributeError), so
# patch("...ETS.ETS_fkine", ...) resolves "ETS" to the shadowing class
# and fails with AttributeError; 3.11+ uses pkgutil.resolve_name and
# isn't fooled. patch.object() against the real module (via sys.modules)
# works on every version.
_ETS_module = sys.modules["roboticstoolbox.ets.ETS"]
from spatialmath import SE3

_NO_FKNM_C = "compiled _fknm_c extension not built -- can't cross-validate against C"
_NO_FRNE_C = "compiled _frne_c extension not built -- can't cross-validate against C"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Non-trivial joint configuration for Panda
PANDA_Q = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])

# Non-trivial joint configuration for Puma560
PUMA_Q = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def _panda_ets():
    return rtb.models.Panda().ets()


def _puma():
    return rtb.models.DH.Puma560()


def _twolink():
    # Standard-DH, non-identity base (SE3.Rx(pi/2)) -- unlike Puma560
    # (identity base), this exercises the self.base rotation path in
    # rne()/rne_python()/Robot.rne().
    return rtb.models.DH.TwoLink()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _no_c_fkine():
    """Force ETS.eval() onto the Python path via the facade's Python implementation."""
    from roboticstoolbox.ets.fknm import _python_fkine

    def _py(fknm, q, base, tool, include_base, _data=None):
        return _python_fkine(_data, q, base, tool, include_base)

    with patch.object(_ETS_module, "ETS_fkine", new=_py):
        yield


@contextmanager
def _no_c_jacob0():
    """Force ETS.jacob0() onto the Python path via the facade's Python implementation."""
    from roboticstoolbox.ets.fknm import _python_jacob0

    def _py(fknm, q, tool, _data=None, _n=None):
        return _python_jacob0(_data, _n, q, tool)

    with patch.object(_ETS_module, "ETS_jacob0", new=_py):
        yield


@contextmanager
def _no_c_jacobe():
    """Force ETS.jacobe() onto the Python path via the facade's Python implementation."""
    from roboticstoolbox.ets.fknm import _python_jacobe

    def _py(fknm, q, tool, _data=None, _n=None):
        return _python_jacobe(_data, _n, q, tool)

    with patch.object(_ETS_module, "ETS_jacobe", new=_py):
        yield


@contextmanager
def _no_c_hessian0():
    """Force ETS.hessian0() onto the Python path via the facade's Python implementation."""
    from roboticstoolbox.ets.fknm import _python_jacob0, _python_hessian
    from spatialmath.base import getvector, verifymatrix

    def _py(fknm, q, J0, tool, _data=None, _n=None):
        if J0 is None:
            if q is None:
                raise ValueError("Either J0 or q must be provided")
            q = getvector(q, None)
            J0 = _python_jacob0(_data, _n, q, tool)
        else:
            verifymatrix(J0, (6, _n))
        return _python_hessian(J0)

    with patch.object(_ETS_module, "ETS_hessian0", new=_py):
        yield


@contextmanager
def _no_c_hessiane():
    """Force ETS.hessiane() onto the Python path via the facade's Python implementation."""
    from roboticstoolbox.ets.fknm import _python_jacobe, _python_hessian
    from spatialmath.base import getvector, verifymatrix

    def _py(fknm, q, Je, tool, _data=None, _n=None):
        if Je is None:
            if q is None:
                raise ValueError("Either Je or q must be provided")
            q = getvector(q, None)
            Je = _python_jacobe(_data, _n, q, tool)
        else:
            verifymatrix(Je, (6, _n))
        return _python_hessian(Je)

    with patch.object(_ETS_module, "ETS_hessiane", new=_py):
        yield


# ---------------------------------------------------------------------------
# eval() / fkine() fallback
# ---------------------------------------------------------------------------

@unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
class TestEvalFallback(unittest.TestCase):
    """eval() Python fallback produces numerically identical results to C path."""

    def setUp(self):
        self.ets = _panda_ets()
        self.q = PANDA_Q
        self.c_result = self.ets.eval(self.q)

    def test_single_config(self):
        with _no_c_fkine():
            py = self.ets.eval(self.q)
        nt.assert_array_almost_equal(py, self.c_result)

    def test_trajectory(self):
        qt = np.vstack([PANDA_Q * s for s in np.linspace(0.5, 1.5, 6)])
        c = self.ets.eval(qt)
        with _no_c_fkine():
            py = self.ets.eval(qt)
        nt.assert_array_almost_equal(py, c)

    def test_with_base_ndarray(self):
        base = SE3.Rx(0.5).A
        c = self.ets.eval(self.q, base=base)
        with _no_c_fkine():
            py = self.ets.eval(self.q, base=base)
        nt.assert_array_almost_equal(py, c)

    def test_with_base_SE3(self):
        base = SE3.Rx(0.5)
        c = self.ets.eval(self.q, base=base)
        with _no_c_fkine():
            py = self.ets.eval(self.q, base=base)
        nt.assert_array_almost_equal(py, c)

    def test_with_tool_ndarray(self):
        tool = SE3.Tz(0.1).A
        c = self.ets.eval(self.q, tool=tool)
        with _no_c_fkine():
            py = self.ets.eval(self.q, tool=tool)
        nt.assert_array_almost_equal(py, c)

    def test_with_tool_SE3(self):
        tool = SE3.Tz(0.1)
        c = self.ets.eval(self.q, tool=tool)
        with _no_c_fkine():
            py = self.ets.eval(self.q, tool=tool)
        nt.assert_array_almost_equal(py, c)

    def test_with_base_and_tool(self):
        base = SE3.Rx(0.3).A
        tool = SE3.Tz(0.1).A
        c = self.ets.eval(self.q, base=base, tool=tool)
        with _no_c_fkine():
            py = self.ets.eval(self.q, base=base, tool=tool)
        nt.assert_array_almost_equal(py, c)

    def test_include_base_false(self):
        c = self.ets.eval(self.q, include_base=False)
        with _no_c_fkine():
            py = self.ets.eval(self.q, include_base=False)
        nt.assert_array_almost_equal(py, c)


class TestFkineFallback(unittest.TestCase):
    """fkine() wraps eval(); verify it returns SE3 and agrees with C path."""

    def setUp(self):
        self.ets = _panda_ets()
        self.q = PANDA_Q

    def test_returns_SE3(self):
        with _no_c_fkine():
            result = self.ets.fkine(self.q)
        self.assertIsInstance(result, SE3)

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_matches_c_path(self):
        c = self.ets.fkine(self.q)
        with _no_c_fkine():
            py = self.ets.fkine(self.q)
        nt.assert_array_almost_equal(py.A, c.A)

    def test_trajectory_returns_se3_batch(self):
        qt = np.vstack([PANDA_Q * s for s in np.linspace(0.5, 1.5, 4)])
        with _no_c_fkine():
            result = self.ets.fkine(qt)
        self.assertEqual(len(result), 4)


# ---------------------------------------------------------------------------
# jacob0() fallback
# ---------------------------------------------------------------------------

class TestJacob0Fallback(unittest.TestCase):
    """jacob0() Python fallback agrees with C path."""

    def setUp(self):
        self.ets = _panda_ets()
        self.q = PANDA_Q
        self.c_result = self.ets.jacob0(self.q)

    def test_shape(self):
        with _no_c_jacob0():
            py = self.ets.jacob0(self.q)
        self.assertEqual(py.shape, (6, self.ets.n))

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_matches_c_path(self):
        with _no_c_jacob0():
            py = self.ets.jacob0(self.q)
        nt.assert_array_almost_equal(py, self.c_result)

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_with_tool(self):
        tool = SE3.Tz(0.1).A
        c = self.ets.jacob0(self.q, tool=tool)
        with _no_c_jacob0():
            py = self.ets.jacob0(self.q, tool=tool)
        nt.assert_array_almost_equal(py, c)


# ---------------------------------------------------------------------------
# jacobe() fallback
# ---------------------------------------------------------------------------

class TestJacobeFallback(unittest.TestCase):
    """jacobe() Python fallback agrees with C path."""

    def setUp(self):
        self.ets = _panda_ets()
        self.q = PANDA_Q
        self.c_result = self.ets.jacobe(self.q)

    def test_shape(self):
        with _no_c_jacobe():
            py = self.ets.jacobe(self.q)
        self.assertEqual(py.shape, (6, self.ets.n))

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_matches_c_path(self):
        with _no_c_jacobe():
            py = self.ets.jacobe(self.q)
        nt.assert_array_almost_equal(py, self.c_result)

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_with_tool(self):
        tool = SE3.Tz(0.1).A
        c = self.ets.jacobe(self.q, tool=tool)
        with _no_c_jacobe():
            py = self.ets.jacobe(self.q, tool=tool)
        nt.assert_array_almost_equal(py, c)


# ---------------------------------------------------------------------------
# hessian0() fallback
# ---------------------------------------------------------------------------

class TestHessian0Fallback(unittest.TestCase):
    """hessian0() Python fallback agrees with C path."""

    def setUp(self):
        self.ets = _panda_ets()
        self.q = PANDA_Q
        self.J0 = self.ets.jacob0(self.q)
        self.c_result = self.ets.hessian0(self.q, J0=self.J0)

    def test_shape(self):
        with _no_c_hessian0():
            py = self.ets.hessian0(self.q, J0=self.J0)
        n = self.ets.n
        self.assertEqual(py.shape, (n, 6, n))

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_matches_c_path(self):
        with _no_c_hessian0():
            py = self.ets.hessian0(self.q, J0=self.J0)
        nt.assert_array_almost_equal(py, self.c_result)

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_without_precomputed_J0(self):
        """hessian0 should compute J0 internally when not supplied."""
        c = self.ets.hessian0(self.q)
        with _no_c_hessian0():
            # Python path computes J0 internally via self.jacob0()
            # but jacob0() itself will try C first; we need to patch both
            with _no_c_jacob0():
                py = self.ets.hessian0(self.q)
        nt.assert_array_almost_equal(py, c)


# ---------------------------------------------------------------------------
# hessiane() fallback
# ---------------------------------------------------------------------------

class TestHessianeFallback(unittest.TestCase):
    """hessiane() Python fallback agrees with C path."""

    def setUp(self):
        self.ets = _panda_ets()
        self.q = PANDA_Q
        self.Je = self.ets.jacobe(self.q)
        self.c_result = self.ets.hessiane(self.q, Je=self.Je)

    def test_shape(self):
        with _no_c_hessiane():
            py = self.ets.hessiane(self.q, Je=self.Je)
        n = self.ets.n
        self.assertEqual(py.shape, (n, 6, n))

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_matches_c_path(self):
        with _no_c_hessiane():
            py = self.ets.hessiane(self.q, Je=self.Je)
        nt.assert_array_almost_equal(py, self.c_result)


# ---------------------------------------------------------------------------
# Reference values: fkine/jacob0 against known Panda results (regression
# guard, independent of C).
#
# Unlike test_matches_c_path above, these do NOT compare the C path against
# the Python path -- they compare whichever path is actually active against
# hardcoded ground truth. That makes them meaningful in every environment,
# including the pure-Python (no _fknm_c) case the pyodide wheel ships:
# when C is unavailable, ets.fkine()/jacob0() already dispatch to the
# Python implementation automatically (ETS_init returns None), so this is
# exactly what a C-less install needs to prove it's still correct on its
# own terms, not just "consistent with itself".
#
# Values computed once against the C extension (see TestFkineFallback /
# TestJacob0Fallback, which already show the two paths agree) using
# ets.fkine(PANDA_Q) / ets.jacob0(PANDA_Q) on Franka Panda.
# ---------------------------------------------------------------------------

class TestFkineReference(unittest.TestCase):
    """fkine() on Panda against hardcoded reference values."""

    def setUp(self):
        self.ets = _panda_ets()

    def test_fkine(self):
        nt.assert_array_almost_equal(
            self.ets.fkine(PANDA_Q).A,
            [
                [-0.50827907, -0.57904589, 0.63746234, 0.44707793],
                [0.83014553, -0.52639462, 0.18375824, 0.16175746],
                [0.22915229, 0.62258699, 0.74824773, 0.96828043],
                [0.00000000, 0.00000000, 0.00000000, 1.00000000],
            ],
            decimal=6,
        )


class TestJacob0Reference(unittest.TestCase):
    """jacob0() on Panda against hardcoded reference values."""

    def setUp(self):
        self.ets = _panda_ets()

    def test_jacob0(self):
        nt.assert_array_almost_equal(
            self.ets.jacob0(PANDA_Q),
            [
                [-1.61757460e-01, 1.07976800e-01, -3.41587423e-02,
                 3.35336541e-01, -1.07172949e-02, 1.03491264e-01, 0.0],
                [4.47077932e-01, 6.26036931e-01, 4.16714460e-01,
                 -8.05054464e-02, 7.78094113e-02, -1.17637200e-02, 0.0],
                [-6.03103094e-17, -2.35392404e-01, -8.20662027e-02,
                 -5.14331129e-01, -9.97831132e-03, -2.02887489e-01, 0.0],
                [4.61988821e-17, -9.85449730e-01, 3.37672585e-02,
                 -6.16735653e-02, 6.68449878e-01, -1.35361558e-01,
                 6.37462344e-01],
                [9.61515015e-18, 1.69967143e-01, 1.95778638e-01,
                 9.79165111e-01, 1.84470262e-01, 9.82748279e-01,
                 1.83758244e-01],
                [1.00000000e+00, -7.36706147e-17, 9.80066578e-01,
                 -1.93473657e-01, 7.20517510e-01, -1.26028049e-01,
                 7.48247732e-01],
            ],
            decimal=6,
        )


# ---------------------------------------------------------------------------
# Symbolic inputs
# ---------------------------------------------------------------------------

class TestSymbolicFkine(unittest.TestCase):
    """fkine() handles SymPy symbols in q and in ET parameters."""

    def test_symbolic_q(self):
        q0 = sympy.Symbol("q0")
        ets = rtb.ET.Rz(jindex=0) * rtb.ET.tz(0.5)
        result = ets.fkine([q0])
        self.assertIsInstance(result, SE3)

    def test_symbolic_et_param(self):
        # ET with a symbolic constant parameter mixed with joint ETs.
        # This is the pattern from test_fkine_sym in test_ETS.py.
        x = sympy.Symbol("x")
        q0 = sympy.Symbol("q0")
        ets = rtb.ET.Rx(x) * rtb.ET.Rz(jindex=0) * rtb.ET.tz(1.0)
        result = ets.fkine([q0])
        self.assertIsInstance(result, SE3)
        # Result should contain at least one of our symbols
        flat = list(result.A.flat)
        has_symbol = any(
            hasattr(v, "free_symbols") and (x in v.free_symbols or q0 in v.free_symbols)
            for v in flat
        )
        self.assertTrue(has_symbol, "Expected symbolic content in result matrix")

    def test_symbolic_agrees_with_numeric(self):
        """Symbolic evaluation at a concrete value matches pure numeric result."""
        q0 = sympy.Symbol("q0")
        ets = rtb.ET.Rz(jindex=0) * rtb.ET.tz(0.4) * rtb.ET.Ry(jindex=1)

        q_numeric = [0.7, 1.2]
        q_symbolic = [q0, sympy.Rational(6, 5)]  # 1.2 exactly

        numeric_result = ets.fkine(q_numeric).A

        sym_result_raw = ets.fkine(q_symbolic).A
        sym_result = np.array(
            [[float(v.subs(q0, sympy.Float("0.7"))) for v in row]
             for row in sym_result_raw.tolist()]
        )
        nt.assert_array_almost_equal(sym_result, numeric_result, decimal=6)

    def test_symbolic_jacob0(self):
        q0 = sympy.Symbol("q0")
        ets = rtb.ET.Rz(jindex=0) * rtb.ET.tz(0.5) * rtb.ET.Rz(jindex=1)
        result = ets.jacob0([q0, sympy.Integer(0)])
        self.assertEqual(result.shape, (6, 2))

    def test_symbolic_jacobe(self):
        q0 = sympy.Symbol("q0")
        ets = rtb.ET.Rz(jindex=0) * rtb.ET.tz(0.5) * rtb.ET.Rz(jindex=1)
        result = ets.jacobe([q0, sympy.Integer(0)])
        self.assertEqual(result.shape, (6, 2))


# ---------------------------------------------------------------------------
# rne: C path vs rne_python (pure Python NE)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_FRNE_C_AVAILABLE, _NO_FRNE_C)
class TestRNEFallback(unittest.TestCase):
    """rne() C path (frne/ne.c) agrees with rne_python() on Puma560."""

    def setUp(self):
        self.puma = _puma()
        self.z = np.zeros(6)
        self.o = np.ones(6)

    def _compare(self, q, qd, qdd, **kwargs):
        c = self.puma.rne(q, qd, qdd, **kwargs)
        py = self.puma.rne_python(q, qd, qdd, **kwargs)
        nt.assert_array_almost_equal(c, py, decimal=4)

    def test_static(self):
        """Zero velocity and acceleration — gravity load only."""
        self._compare(self.puma.qn, self.z, self.z)

    def test_accel_only(self):
        self._compare(self.puma.qn, self.z, self.o)

    def test_vel_and_accel(self):
        self._compare(self.puma.qn, self.o, self.o)

    def test_vel_only(self):
        self._compare(self.puma.qn, self.o, self.z)

    def test_no_gravity(self):
        self._compare(self.puma.qn, self.o, self.o, gravity=[0, 0, 0])

    def test_external_wrench(self):
        fext = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
        self._compare(self.puma.qn, self.z, self.z, fext=fext)

    def test_trajectory(self):
        """Batch (trajectory) input: two configs stacked."""
        Q = np.vstack([self.puma.qn, self.puma.qn * 0.5])
        QD = np.vstack([self.z, self.o])
        QDD = np.vstack([self.z, self.o])
        c = self.puma.rne(Q, QD, QDD)
        py = self.puma.rne_python(Q, QD, QDD)
        nt.assert_array_almost_equal(c, py, decimal=4)


# ---------------------------------------------------------------------------
# Reference values: rne against known Puma560 results (regression guard)
# ---------------------------------------------------------------------------

class TestRNEReference(unittest.TestCase):
    """
    rne() on Puma560 against hardcoded reference values.

    These values are the ground truth — if either the C or Python path
    deviates, the corresponding TestRNEFallback test above will also fail,
    but these pin the absolute numbers.
    """

    def setUp(self):
        self.puma = _puma()
        self.z = np.zeros(6)
        self.o = np.ones(6)

    def test_gravity_only(self):
        nt.assert_array_almost_equal(
            self.puma.rne(self.puma.qn, self.z, self.z),
            [-0.0000, 31.6399, 6.0351, 0.0000, 0.0283, 0],
            decimal=4,
        )

    def test_accel_only(self):
        nt.assert_array_almost_equal(
            self.puma.rne(self.puma.qn, self.z, self.o),
            [3.35311, 36.0025, 7.42596, 0.190043, 0.203441, 0.194133],
            decimal=4,
        )

    def test_vel_and_accel(self):
        nt.assert_array_almost_equal(
            self.puma.rne(self.puma.qn, self.o, self.o),
            [32.4952, 60.867, 17.7436, 1.45452, 1.29911, 0.713781],
            decimal=4,
        )

    def test_external_wrench(self):
        nt.assert_array_almost_equal(
            self.puma.rne(self.puma.qn, self.z, self.z, fext=[1, 2, 3, 1, 2, 3]),
            [0.642756, 29.0866, 4.70321, 2.82843, -1.97175, 3],
            decimal=4,
        )




# ---------------------------------------------------------------------------
# rne: C path vs rne_python on a rotated-base robot (TwoLink)
#
# Puma560 (used above) has an identity base, so it never exercised the
# base-rotation handling in either path. TwoLink's base is SE3.Rx(pi/2) --
# this is what regresses the bug where Robot.rne() ignored self.base
# entirely and rne_python()'s rotated-base branch was missing a negation
# (see rne.md / tech-debt.md).
# ---------------------------------------------------------------------------

@unittest.skipUnless(_FRNE_C_AVAILABLE, _NO_FRNE_C)
class TestRNERotatedBaseFallback(unittest.TestCase):
    """rne() C path agrees with rne_python() on TwoLink (rotated base)."""

    def setUp(self):
        self.robot = _twolink()
        self.z = np.zeros(2)

    def _compare(self, q, qd, qdd, **kwargs):
        c = self.robot.rne(q, qd, qdd, **kwargs)
        py = self.robot.rne_python(q, qd, qdd, **kwargs)
        nt.assert_array_almost_equal(c, py, decimal=4)

    def test_gravity_only(self):
        self._compare([0.3, 0.5], self.z, self.z)

    def test_other_pose(self):
        self._compare([1.2, -0.7], self.z, self.z)


class TestRNERotatedBaseReference(unittest.TestCase):
    """rne() on TwoLink (rotated base) against hardcoded reference values.

    Independently verified against the Lagrangian identity (ground truth
    via numerical differentiation of gravitational potential energy,
    convention-independent of either RNE implementation) -- see the
    TwoLink section of rne.md.
    """

    def setUp(self):
        self.robot = _twolink()
        self.z = np.zeros(2)

    def test_gravity_only(self):
        nt.assert_array_almost_equal(
            self.robot.rne([0.3, 0.5], self.z, self.z),
            [-17.457309, -3.413863],
            decimal=4,
        )


# ---------------------------------------------------------------------------
# base_wrench: C path vs rne_python, on both an identity-base (Puma560) and
# rotated-base (TwoLink) robot, with and without a wrench applied to the
# end-effector.
# ---------------------------------------------------------------------------

@unittest.skipUnless(_FRNE_C_AVAILABLE, _NO_FRNE_C)
class TestBaseWrenchFallback(unittest.TestCase):
    """rne(base_wrench=True) C path agrees with rne_python()."""

    # wrench applied to end-effector: [Fx, Fy, Fz, Mx, My, Mz]
    FEXT = [0.5, 0.7, 0.7, 0.1, 0.2, 0.3]

    def _compare(self, robot, q, fext=None):
        n = robot.n
        z = np.zeros(n)
        tau_c, wbase_c = robot.rne(q, z, z, fext=fext, base_wrench=True)
        tau_py, wbase_py = robot.rne_python(q, z, z, fext=fext, base_wrench=True)
        nt.assert_array_almost_equal(tau_c, tau_py, decimal=4)
        nt.assert_array_almost_equal(wbase_c, wbase_py, decimal=4)

    def test_puma_no_fext(self):
        puma = _puma()
        self._compare(puma, puma.qn)

    def test_puma_with_ee_wrench(self):
        puma = _puma()
        self._compare(puma, puma.qn, fext=self.FEXT)

    def test_twolink_no_fext(self):
        self._compare(_twolink(), [0.3, 0.5])

    def test_twolink_with_ee_wrench(self):
        self._compare(_twolink(), [0.3, 0.5], fext=self.FEXT)

    def test_ee_wrench_changes_tau(self):
        """Sanity check the comparison above isn't vacuously trivial: an
        end-effector wrench should actually change the joint torques."""
        puma = _puma()
        z = np.zeros(puma.n)
        tau_plain = puma.rne(puma.qn, z, z)
        tau_wrench = puma.rne(puma.qn, z, z, fext=self.FEXT)
        self.assertGreater(np.abs(tau_wrench - tau_plain).max(), 0.1)


class TestBaseWrenchReference(unittest.TestCase):
    """base_wrench against hardcoded reference values, independent of C."""

    FEXT = [0.5, 0.7, 0.7, 0.1, 0.2, 0.3]

    def test_puma_with_ee_wrench(self):
        puma = _puma()
        z = np.zeros(puma.n)
        tau, wbase = puma.rne(puma.qn, z, z, fext=self.FEXT, base_wrench=True)
        nt.assert_array_almost_equal(
            tau,
            [0.422447, 31.151777, 5.913429, 0.282843, -0.171747, 0.300000],
            decimal=4,
        )
        nt.assert_array_almost_equal(
            wbase,
            [0.700000, 0.700000, 229.544500, -48.487576, -30.681496, 0.422447],
            decimal=4,
        )

    def test_twolink_with_ee_wrench(self):
        robot = _twolink()
        z = np.zeros(robot.n)
        tau, wbase = robot.rne([0.3, 0.5], z, z, fext=self.FEXT, base_wrench=True)
        nt.assert_array_almost_equal(tau, [-15.603289, -2.413863], decimal=4)
        nt.assert_array_almost_equal(
            wbase,
            [-0.153796, -18.753627, 0.700000, 0.635213, -0.945353, -15.603289],
            decimal=4,
        )


# ---------------------------------------------------------------------------
# TwoLink(mdh=True) vs TwoLink(mdh=False): two different DH parameterizations
# of the *same physical robot* -- this is what actually caught the bug where
# TwoLink's mdh=True variant just copied the standard-DH `a`/`alpha` values
# onto RevoluteMDH links, which is not how the DH<->MDH conversion works (it
# shifts `a`/`alpha` to the previous link index). A single hand-checked pose
# (qn = [pi/6, -pi/6]) happened to agree by coincidence -- qn is symmetric
# (q2 = -q1) -- while every other pose diverged by up to ~0.7 in fkine.
# Random poses, not just qn, are the point of this test.
#
# This pairing also gives a genuinely independent cross-check of the DH/MDH
# convention finding (issue 6, rne.md). Originally: rne_python() trusted
# only for standard DH, Robot.rne() only for modified DH. Exercising
# rne_python() on this MDH+rotated-base pair (apparently never hit before)
# found and fixed three real bugs in its MDH branch (see
# test_mdh_rne_python_now_agrees) -- rne_python() is now correct for both
# conventions. Robot.rne() (the separate ETS/Featherstone implementation) is
# joint-last-compliant, and thus correct, for mdh=True DHRobot instances
# (test_robot_rne_on_mdh_variant_matches_standard_dh_rne_python); for
# mdh=False it now asserts rather than silently returning a wrong answer
# (test_robot_rne_rejects_standard_dh) -- see rne.md/tech-debt.md.
# ---------------------------------------------------------------------------

class TestTwoLinkDHMDHEquivalence(unittest.TestCase):
    """TwoLink(mdh=False) and TwoLink(mdh=True) are the same physical robot."""

    def setUp(self):
        self.std = _twolink()
        self.mdh = TwoLink(mdh=True)
        self.poses = [
            np.array([0.3, 0.5]),
            np.array([1.2, -0.7]),
            np.array([0.2, 0.3]),
        ]

    def test_fkine_agrees(self):
        for q in self.poses:
            nt.assert_array_almost_equal(
                self.std.fkine(q).A, self.mdh.fkine(q).A, decimal=10
            )

    def test_fkine_agrees_random_poses(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.uniform(-np.pi, np.pi, 2)
            nt.assert_array_almost_equal(
                self.std.fkine(q).A, self.mdh.fkine(q).A, decimal=10
            )

    def test_rne_agrees_between_parameterizations(self):
        """rne() (the single ne.c implementation, dispatched per-robot via
        its own dhtype: STANDARD vs MODIFIED) gives matching torques for
        both parameterizations of the same physical robot -- evidence that
        ne.c's two dhtype branches are mutually consistent, not that
        there are two separate C implementations to compare."""
        z = np.zeros(2)
        for q in self.poses:
            tau_std_py = self.std.rne_python(q, z, z)
            tau_std = self.std.rne(q, z, z)
            tau_mdh = self.mdh.rne(q, z, z)
            nt.assert_array_almost_equal(tau_std_py, tau_std, decimal=4)
            nt.assert_array_almost_equal(tau_std, tau_mdh, decimal=4)

    def test_robot_rne_on_mdh_variant_matches_standard_dh_rne_python(self):
        """The core rne.md finding, checked against a real matched pair
        rather than the synthetic 1-link case: Robot.rne() (trusted for
        MDH) applied to the MDH parameterization must agree with
        rne_python() (trusted for standard DH) applied to the standard-DH
        parameterization of the same physical robot."""
        from roboticstoolbox.robot.Robot import Robot as RobotBase

        z = np.zeros(2)
        for q in self.poses:
            tau_std_py = self.std.rne_python(q, z, z)
            tau_mdh_base = RobotBase.rne(self.mdh, q, z, z)
            nt.assert_array_almost_equal(tau_std_py, tau_mdh_base, decimal=4)

    def test_mdh_rne_python_now_agrees(self):
        """rne_python() on the MDH parameterization -- exercising this
        (mdh=True + a non-identity base) found three real bugs, none of
        them issue 6:

        1. Base rotation applied to gravity twice: once (correctly)
           before the recursion loop, and again via Tj = base @ Tj for
           the first link -- double-counted, not just wrong.
        2. Missing parentheses in the MDH revolute case's linear
           acceleration formula: `Rt @ cross(wd, pstar) + cross(w,
           cross(w, pstar)) + vd` should distribute Rt over the whole
           sum, per ne.c's MODIFIED-DH branch (rot_trans_vect_mult
           applied to the full bracket) -- the prismatic case right
           below it already had this right.
        3. The backward recursion's moment equation used `pstar` (the
           *next* link's offset) where it should use `r` (this link's
           own CoM offset) for the this-link-force-to-torque term --
           ne.c's equivalent is `R_COG(j) x F`, not `PSTAR(j+1) x F`.

        With all three fixed, rne_python() is now correct for MDH too,
        not just standard DH -- see test_robot_rne_rejects_standard_dh
        for the one implementation (Robot.rne(), the ETS/Featherstone
        one, unrelated code) that still can't handle standard DH (by
        design, guarded, not silently wrong).
        """
        z = np.zeros(2)
        for q in self.poses:
            truth = self.std.rne_python(q, z, z)
            mdh_py = self.mdh.rne_python(q, z, z)
            nt.assert_array_almost_equal(mdh_py, truth, decimal=4)

    def test_robot_rne_rejects_standard_dh(self):
        """Robot.rne() (the ETS/Featherstone implementation, used by
        ERobot/URDF/PoERobot) genuinely cannot handle a standard-DH
        (mdh=False) DHRobot -- the joint isn't the last element of its own
        ETS segment. Rather than silently returning a wrong torque (the
        old behaviour -- see rne.md/tech-debt.md), it now asserts. This
        confirms the guard fires for the one case it must, complementing
        test_robot_rne_on_mdh_variant_matches_standard_dh_rne_python (which
        confirms it does *not* fire, and gives correct results, for the
        mdh=True case)."""
        from roboticstoolbox.robot.Robot import Robot as RobotBase

        z = np.zeros(2)
        with self.assertRaises(AssertionError):
            RobotBase.rne(self.std, self.poses[0], z, z)


# ---------------------------------------------------------------------------
# Actuator dynamics (Jm, G, B, Tc) and non-zero link inertia: C vs
# rne_python() consistency on TwoLink, for both DH conventions. TwoLink is
# zero for all of these by default (per its own docstring: "Motor inertia
# is 0 ... Viscous and Coulomb friction is 0", link inertias also 0 unless
# inertia=True) -- so nothing before this exercised the actuator-dynamics
# terms (ne.c:485-491 / Link.friction()) or a non-zero inertia tensor
# through rne() at all.
# ---------------------------------------------------------------------------

@unittest.skipUnless(_FRNE_C_AVAILABLE, _NO_FRNE_C)
class TestTwoLinkActuatorDynamics(unittest.TestCase):
    """rne() (C) vs rne_python() with non-zero Jm/G/B/Tc and link inertia."""

    def _with_actuator_dynamics(self, robot):
        for i, link in enumerate(robot.links):
            link.Jm = 0.05 * (i + 1)
            link.G = 100.0 * (i + 1)
            link.B = 0.01 * (i + 1)
            link.Tc = [0.3 * (i + 1), -0.2 * (i + 1)]
        return robot

    def setUp(self):
        self.std = self._with_actuator_dynamics(TwoLink(mdh=False, inertia=True))
        self.mdh = self._with_actuator_dynamics(TwoLink(mdh=True, inertia=True))
        # non-zero qd/qdd: needed to actually exercise B/Tc (velocity-
        # dependent) and Jm (acceleration-dependent) -- the gravity-only
        # (qd=qdd=0) tests elsewhere in this file wouldn't touch them
        self.q = np.array([0.3, 0.5])
        self.qd = np.array([0.4, -0.6])
        self.qdd = np.array([0.2, 0.7])

    def test_std_dh_c_matches_python(self):
        tau_c = self.std.rne(self.q, self.qd, self.qdd)
        tau_py = self.std.rne_python(self.q, self.qd, self.qdd)
        nt.assert_array_almost_equal(tau_c, tau_py, decimal=4)

    def test_mdh_c_matches_python(self):
        tau_c = self.mdh.rne(self.q, self.qd, self.qdd)
        tau_py = self.mdh.rne_python(self.q, self.qd, self.qdd)
        nt.assert_array_almost_equal(tau_c, tau_py, decimal=4)

    def test_inertia_and_actuator_dynamics_actually_matter(self):
        """Sanity check the comparisons above aren't vacuous: non-zero
        inertia/Jm/G/B/Tc must actually change the result relative to the
        all-zero-by-default TwoLink()."""
        bare = TwoLink(mdh=False)
        tau_bare = bare.rne(self.q, self.qd, self.qdd)
        tau_with = self.std.rne(self.q, self.qd, self.qdd)
        self.assertGreater(np.abs(tau_with - tau_bare).max(), 0.1)


# ---------------------------------------------------------------------------
# Robot.rne() (ETS/Featherstone) dropped the rotational inertia tensor
# entirely: SpatialInertia(m=link.m, r=link.r) never passed I=link.I. Never
# caught by TestTwoLinkDHMDHEquivalence above because TwoLink's default
# (zero) inertia makes the missing term a no-op, and TwoLink's d=alpha=0
# (planar) also hides it even with inertia=True. Needs nonzero d *and*
# alpha *and* a real inertia tensor together -- see tech-debt.md.
# ---------------------------------------------------------------------------

class TestRobotRneInertiaTensor(unittest.TestCase):
    """Robot.rne() must use the full inertia tensor, not just mass+COM."""

    def _robot(self):
        from roboticstoolbox import DHRobot, RevoluteMDH

        links = [
            RevoluteMDH(
                d=0.5, a=0.3, alpha=0.6,
                m=2.0, r=[0.1, 0.05, 0.02],
                I=[0.01, 0.02, 0.03, 0.001, 0.002, 0.003],
            ),
            RevoluteMDH(
                d=0.2, a=0.25, alpha=0.8,
                m=1.5, r=[0.08, 0.0, 0.01],
                I=[0.005, 0.006, 0.007, 0.0001, 0.0002, 0.0003],
            ),
        ]
        return DHRobot(links, name="rne_inertia_test")

    def test_robot_rne_matches_rne_python_with_full_inertia_tensor(self):
        from roboticstoolbox.robot.Robot import Robot as RobotBase

        robot = self._robot()
        q = np.array([0.3, -0.5])
        qd = np.array([0.4, -0.2])
        qdd = np.array([0.1, 0.3])

        truth = robot.rne_python(q, qd, qdd)
        result = RobotBase.rne(robot, q, qd, qdd)
        nt.assert_array_almost_equal(result, truth, decimal=8)

    def test_robot_rne_matches_rne_python_static_and_random_poses(self):
        from roboticstoolbox.robot.Robot import Robot as RobotBase

        robot = self._robot()
        z = np.zeros(2)

        # static (gravity only) -- agreed even before the fix, since it
        # doesn't exercise qdd, but kept as a baseline
        truth = robot.rne_python(np.array([0.3, -0.5]), z, z)
        result = RobotBase.rne(robot, np.array([0.3, -0.5]), z, z)
        nt.assert_array_almost_equal(result, truth, decimal=8)

        rng = np.random.default_rng(1)
        for _ in range(20):
            q = rng.uniform(-np.pi, np.pi, 2)
            qd = rng.uniform(-2, 2, 2)
            qdd = rng.uniform(-2, 2, 2)
            truth = robot.rne_python(q, qd, qdd)
            result = RobotBase.rne(robot, q, qd, qdd)
            nt.assert_array_almost_equal(result, truth, decimal=6)


def _spong_two_link_tau(
    q1, q2, q1_dot, q2_dot, q1_ddot, q2_ddot,
    m1, m2, l1, l2, lc1, lc2, I1, I2, g=9.81,
):
    """Analytical inverse dynamics for a planar 2R elbow manipulator.

    Equation (7.87) of Spong, Hutchinson, Vidyasagar, "Robot Modeling and
    Control", Wiley 2006 -- an independent, closed-form ground truth. Shares
    no code with rne_python()/rne()/Robot.rne(), unlike every other check in
    this file, which only ever compares those implementations against each
    other and so cannot catch a bug they all share.
    """
    h = -m2 * l1 * lc2 * math.sin(q2)
    d11 = m1 * (lc1**2) + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * math.cos(q2)) + I1 + I2
    d12 = m2 * (lc2**2 + l1 * lc2 * math.cos(q2)) + I2
    d21 = d12
    d22 = m2 * (lc2**2) + I2
    g1 = (m1 * lc1 + m2 * l1) * g * math.cos(q1) + m2 * lc2 * g * math.cos(q1 + q2)
    g2 = m2 * lc2 * g * math.cos(q1 + q2)
    c121 = h
    c211 = h
    c221 = h
    c112 = -h
    tau1 = (
        d11 * q1_ddot + d12 * q2_ddot
        + c121 * q1_dot * q2_dot + c211 * q2_dot * q1_dot + c221 * (q2_dot**2)
        + g1
    )
    tau2 = d21 * q1_ddot + d22 * q2_ddot + c112 * (q1_dot**2) + g2
    return tau1, tau2


def _twolink_ground_truth(q, qd, qdd, I1=0.0, I2=0.0):
    """TwoLink's expected joint torques, from the independent Spong
    closed-form solution above, in TwoLink's own joint-angle/torque sign
    convention.

    TwoLink's convention is the mirror image of Spong's (TwoLink's
    ``base = SE3.Rx(pi/2)`` puts the arm in a different orientation than
    Spong's own diagram) -- empirically calibrated 2026-07-21 against
    rne_python() and confirmed exact to machine precision across several
    q/qd/qdd combinations, not just the gravity-only case:
    ``tau_rtb(q, qd, qdd) == -tau_spong(-q, -qd, -qdd)``.
    """
    t1, t2 = _spong_two_link_tau(
        -q[0], -q[1], -qd[0], -qd[1], -qdd[0], -qdd[1],
        m1=1.0, m2=1.0, l1=1.0, l2=1.0, lc1=0.5, lc2=0.5,
        I1=I1, I2=I2, g=9.8,
    )
    return np.array([-t1, -t2])


class TestTwoLinkAbsoluteGroundTruth(unittest.TestCase):
    """Absolute correctness check, not just relative agreement between our
    own implementations: TwoLink's torques against an independent
    closed-form solution (Spong et al., Eq 7.87) that shares no code with
    rne_python(), rne() (C), or Robot.rne(). Every other test in this file
    only checks these implementations against each other or against
    hand-derived reference numbers computed the same way rne_python() is --
    none of that can catch a bug all of them share.
    """

    def setUp(self):
        self.std = _twolink()  # mdh=False
        self.mdh = TwoLink(mdh=True)
        self.poses = [
            (np.array([0.3, 0.5]), np.zeros(2), np.zeros(2)),
            (np.array([0.3, 0.9]), np.array([0.4, -0.6]), np.array([0.2, 1.1])),
            (np.array([-1.1, 2.0]), np.array([-0.9, 1.3]), np.array([0.5, -0.7])),
        ]

    def test_rne_python_std_dh_matches_spong(self):
        for q, qd, qdd in self.poses:
            truth = _twolink_ground_truth(q, qd, qdd)
            nt.assert_array_almost_equal(
                self.std.rne_python(q, qd, qdd), truth, decimal=6
            )

    def test_rne_python_mdh_matches_spong(self):
        for q, qd, qdd in self.poses:
            truth = _twolink_ground_truth(q, qd, qdd)
            nt.assert_array_almost_equal(
                self.mdh.rne_python(q, qd, qdd), truth, decimal=6
            )

    @unittest.skipUnless(_FRNE_C_AVAILABLE, _NO_FRNE_C)
    def test_rne_c_std_dh_matches_spong(self):
        for q, qd, qdd in self.poses:
            truth = _twolink_ground_truth(q, qd, qdd)
            nt.assert_array_almost_equal(self.std.rne(q, qd, qdd), truth, decimal=6)

    @unittest.skipUnless(_FRNE_C_AVAILABLE, _NO_FRNE_C)
    def test_rne_c_mdh_matches_spong(self):
        for q, qd, qdd in self.poses:
            truth = _twolink_ground_truth(q, qd, qdd)
            nt.assert_array_almost_equal(self.mdh.rne(q, qd, qdd), truth, decimal=6)

    def test_robot_rne_mdh_matches_spong(self):
        """Robot.rne() (ETS/Featherstone) is only valid for mdh=True."""
        from roboticstoolbox.robot.Robot import Robot as RobotBase

        for q, qd, qdd in self.poses:
            truth = _twolink_ground_truth(q, qd, qdd)
            nt.assert_array_almost_equal(
                RobotBase.rne(self.mdh, q, qd, qdd), truth, decimal=6
            )


# ---------------------------------------------------------------------------
# Path verification via timing
# ---------------------------------------------------------------------------

@unittest.skipIf(os.environ.get("CI"), "timing test skipped in CI")
class TestPathTiming(unittest.TestCase):
    """
    Verify that C and Python paths are genuinely different code paths by
    measuring their relative speed.  The C extension must be at least 5×
    faster than the pure-Python fallback on a 7-DOF robot.

    Skipped in CI because timing is sensitive to machine load.  Run locally
    to confirm after any refactor that both paths are still active.
    """

    N = 500  # repetitions per path

    def setUp(self):
        self.ets = _panda_ets()
        self.q = PANDA_Q

    def _time_c(self, fn_name, *args, **kwargs):
        fn = getattr(self.ets, fn_name)
        return timeit.timeit(lambda: fn(*args, **kwargs), number=self.N)

    def _time_python(self, no_c_ctx, fn_name, *args, **kwargs):
        fn = getattr(self.ets, fn_name)
        with no_c_ctx():
            return timeit.timeit(lambda: fn(*args, **kwargs), number=self.N)

    def _assert_c_faster(self, t_c, t_py, factor=5):
        ratio = t_py / t_c
        self.assertGreater(
            ratio,
            factor,
            f"Expected Python path to be >{factor}× slower than C, got {ratio:.1f}×\n"
            f"  C path:      {t_c * 1000 / self.N:.3f} ms/call\n"
            f"  Python path: {t_py * 1000 / self.N:.3f} ms/call",
        )

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_eval_c_faster_than_python(self):
        t_c = self._time_c("eval", self.q)
        t_py = self._time_python(_no_c_fkine, "eval", self.q)
        self._assert_c_faster(t_c, t_py)

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_jacob0_c_faster_than_python(self):
        t_c = self._time_c("jacob0", self.q)
        t_py = self._time_python(_no_c_jacob0, "jacob0", self.q)
        self._assert_c_faster(t_c, t_py)

    @unittest.skipUnless(_FKNM_C_AVAILABLE, _NO_FKNM_C)
    def test_jacobe_c_faster_than_python(self):
        t_c = self._time_c("jacobe", self.q)
        t_py = self._time_python(_no_c_jacobe, "jacobe", self.q)
        self._assert_c_faster(t_c, t_py)


if __name__ == "__main__":
    unittest.main()
