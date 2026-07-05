"""
Safety-net tests for the ETS / fknm / frne refactor (Phase 0).

Each test runs an ETS function twice:
  1. via the C extension (fknm) — the normal path
  2. via the pure-Python fallback — forced by patching the C function to raise

Both paths must produce numerically identical results.  These tests will catch
any regression introduced by the Facade refactor (Phase 1), the BaseETS
redesign (Phase 2), or the nanobind port (Phase 3).

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

import roboticstoolbox as rtb
# roboticstoolbox/robot/ETS.py defines a class also called ETS, and
# roboticstoolbox/robot/__init__.py does `from ...ETS import ETS`, which
# rebinds the "ETS" attribute on the roboticstoolbox.robot package to the
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
_ETS_module = sys.modules["roboticstoolbox.robot.ETS"]
from spatialmath import SE3


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@contextmanager
def _no_c_fkine():
    """Force ETS.eval() onto the Python path via the facade's Python implementation."""
    from roboticstoolbox.robot.fknm import _python_fkine

    def _py(fknm, q, base, tool, include_base, _data=None):
        return _python_fkine(_data, q, base, tool, include_base)

    with patch.object(_ETS_module, "ETS_fkine", new=_py):
        yield


@contextmanager
def _no_c_jacob0():
    """Force ETS.jacob0() onto the Python path via the facade's Python implementation."""
    from roboticstoolbox.robot.fknm import _python_jacob0

    def _py(fknm, q, tool, _data=None, _n=None):
        return _python_jacob0(_data, _n, q, tool)

    with patch.object(_ETS_module, "ETS_jacob0", new=_py):
        yield


@contextmanager
def _no_c_jacobe():
    """Force ETS.jacobe() onto the Python path via the facade's Python implementation."""
    from roboticstoolbox.robot.fknm import _python_jacobe

    def _py(fknm, q, tool, _data=None, _n=None):
        return _python_jacobe(_data, _n, q, tool)

    with patch.object(_ETS_module, "ETS_jacobe", new=_py):
        yield


@contextmanager
def _no_c_hessian0():
    """Force ETS.hessian0() onto the Python path via the facade's Python implementation."""
    from roboticstoolbox.robot.fknm import _python_jacob0, _python_hessian
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
    from roboticstoolbox.robot.fknm import _python_jacobe, _python_hessian
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

    def test_matches_c_path(self):
        with _no_c_jacob0():
            py = self.ets.jacob0(self.q)
        nt.assert_array_almost_equal(py, self.c_result)

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

    def test_matches_c_path(self):
        with _no_c_jacobe():
            py = self.ets.jacobe(self.q)
        nt.assert_array_almost_equal(py, self.c_result)

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

    def test_matches_c_path(self):
        with _no_c_hessian0():
            py = self.ets.hessian0(self.q, J0=self.J0)
        nt.assert_array_almost_equal(py, self.c_result)

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

    def test_matches_c_path(self):
        with _no_c_hessiane():
            py = self.ets.hessiane(self.q, Je=self.Je)
        nt.assert_array_almost_equal(py, self.c_result)


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

    @unittest.skipIf(not __import__("roboticstoolbox.robot.fknm", fromlist=["_C_AVAILABLE"])._C_AVAILABLE,
                     "C extension not built")
    def test_eval_c_faster_than_python(self):
        t_c = self._time_c("eval", self.q)
        t_py = self._time_python(_no_c_fkine, "eval", self.q)
        self._assert_c_faster(t_c, t_py)

    @unittest.skipIf(not __import__("roboticstoolbox.robot.fknm", fromlist=["_C_AVAILABLE"])._C_AVAILABLE,
                     "C extension not built")
    def test_jacob0_c_faster_than_python(self):
        t_c = self._time_c("jacob0", self.q)
        t_py = self._time_python(_no_c_jacob0, "jacob0", self.q)
        self._assert_c_faster(t_c, t_py)

    @unittest.skipIf(not __import__("roboticstoolbox.robot.fknm", fromlist=["_C_AVAILABLE"])._C_AVAILABLE,
                     "C extension not built")
    def test_jacobe_c_faster_than_python(self):
        t_c = self._time_c("jacobe", self.q)
        t_py = self._time_python(_no_c_jacobe, "jacobe", self.q)
        self._assert_c_faster(t_c, t_py)


if __name__ == "__main__":
    unittest.main()
