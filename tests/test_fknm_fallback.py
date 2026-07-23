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
