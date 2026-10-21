"""Tests of analytic inverse kinematics: DHRobot.analytic_ik, ikine_analytic and AnalyticIK."""

import sys
import unittest

import numpy as np
import numpy.testing as nt
from spatialmath import SE3

import roboticstoolbox as rtb
from roboticstoolbox.robot.IK import IKSolution

try:
    import sympy  # noqa: F401

    HAVE_SYMPY = True
except ImportError:  # pragma: no cover
    HAVE_SYMPY = False


@unittest.skipUnless(HAVE_SYMPY, "SymPy is not installed")
class TestAnalyticIK(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.puma = rtb.models.DH.Puma560()
        cls.stanford = rtb.models.DH.Stanford()
        cls.jaco = rtb.models.DH.Jaco()
        # inside the joint limits of the Puma 560
        cls.q = np.array([0.2, 0.9, -0.8, 0.3, -0.5, 0.4])

    def check_fk(self, robot, sol, T, tol=1e-6):
        rows = np.atleast_2d(sol.q)
        Ts = T if isinstance(T, SE3) and len(T) > 1 else [T] * len(rows)
        for q, Tk in zip(rows, Ts):
            nt.assert_allclose(robot.fkine(q).A, np.asarray(Tk.A), atol=tol)

    def test_methods(self):
        from roboticstoolbox.robot.symbolic_ik_rr import (
            AnalyticIK,
            ClosedFormIK,
            General6RIK,
        )

        self.assertIsInstance(self.puma.analytic_ik(), ClosedFormIK)
        self.assertIsInstance(self.stanford.analytic_ik(), ClosedFormIK)
        self.assertIsInstance(self.jaco.analytic_ik(), General6RIK)
        self.assertIsInstance(self.jaco.analytic_ik(), AnalyticIK)

    def test_cached(self):
        self.assertIs(self.puma.analytic_ik(), self.puma.analytic_ik())

    def test_solve_all(self):
        T = self.puma.fkine(self.q)
        ik = self.puma.analytic_ik()

        sol = ik.solve(T)
        self.assertIsInstance(sol, IKSolution)
        self.assertTrue(sol)
        self.assertGreater(len(sol), 1)
        self.check_fk(self.puma, sol, T)
        self.assertTrue(any(np.allclose(row, self.q, atol=1e-6) for row in sol))

        # an array is the same as an SE3
        nt.assert_allclose(ik.solve(T.A).q, sol.q)

        # more solutions without the limits, and every one reaches the pose
        free = ik.solve(T, joint_limits=False)
        self.assertGreaterEqual(len(free), len(sol))
        self.check_fk(self.puma, free, T)
        for row in sol:
            qlim = self.puma.qlim
            self.assertTrue(np.all(row >= qlim[0] - 1e-9) and np.all(row <= qlim[1] + 1e-9))

    def test_solve_needs_one_pose(self):
        T = SE3([self.puma.fkine(self.q)] * 3)
        with self.assertRaises(ValueError):
            self.puma.analytic_ik().solve(T)

    def test_ikine_nearest(self):
        T = self.puma.fkine(self.q)
        sol = self.puma.ikine_analytic(T, q0=self.q + 0.05)
        self.assertTrue(sol)
        self.assertEqual(sol.q.shape, (6,))
        nt.assert_allclose(sol.q, self.q, atol=1e-6)
        self.assertEqual((sol.iterations, sol.searches), (0, 0))
        self.assertLess(sol.residual, 1e-8)

    def test_weights(self):
        T = self.puma.fkine(self.q)
        sol = self.puma.ikine_analytic(T, q0=self.q, weights=[1, 1, 1, 0, 1, 0])
        self.assertTrue(sol)
        nt.assert_allclose(sol.q[[0, 1, 2, 4]], self.q[[0, 1, 2, 4]], atol=1e-6)

    def test_trajectory(self):
        T0 = self.puma.fkine(self.q)
        Ts = rtb.ctraj(T0, T0 * SE3(0.05, 0.02, -0.03), 10)
        sol = self.puma.ikine_analytic(Ts, q0=self.q)
        self.assertTrue(sol)
        self.assertEqual(sol.q.shape, (10, 6))
        self.assertEqual(len(sol), 10)
        self.check_fk(self.puma, sol, Ts)
        # one branch is followed, so no joint jumps
        self.assertLess(np.abs(np.diff(sol.q, axis=0)).max(), 0.3)

        # an array of poses is the same as an SE3
        sol2 = self.puma.ikine_analytic(np.asarray(Ts.A), q0=self.q)
        nt.assert_allclose(sol2.q, sol.q)

    def test_unreachable(self):
        sol = self.puma.ikine_analytic(SE3(5, 5, 5))
        self.assertFalse(sol)
        self.assertTrue(np.isnan(sol.q).all())
        self.assertIn("workspace", sol.reason)

        # in a trajectory the bad pose is named, and the others are solved
        T0 = self.puma.fkine(self.q)
        Ts = SE3([T0, SE3(5, 5, 5), T0])
        sol = self.puma.ikine_analytic(Ts, q0=self.q)
        self.assertFalse(sol)
        self.assertIn("pose 1", sol.reason)
        self.assertTrue(np.isnan(sol.q[1]).all())
        nt.assert_allclose(sol.q[0], self.q, atol=1e-6)
        nt.assert_allclose(sol.q[2], self.q, atol=1e-6)

    def test_joint_limits_reject(self):
        T = self.puma.fkine(self.q)
        # a robot whose limits exclude every solution of this pose
        robot = rtb.models.DH.Puma560()
        robot.qlim = np.array([[0.9] * 6, [1.0] * 6])
        sol = robot.analytic_ik().solve(T)
        self.assertFalse(sol)
        self.assertIn("joint limits", sol.reason)
        self.assertTrue(robot.analytic_ik().solve(T, joint_limits=False))

    def test_bad_q0(self):
        with self.assertRaises(ValueError):
            self.puma.ikine_analytic(self.puma.fkine(self.q), q0=[0, 0, 0])

    def test_base_and_tool(self):
        robot = rtb.models.DH.Puma560()
        robot.base = SE3(0.1, -0.2, 0.3) * SE3.Rz(0.4)
        robot.tool = SE3(0, 0, 0.15)
        T = robot.fkine(self.q)
        sol = robot.ikine_analytic(T, q0=self.q)
        self.assertTrue(sol)
        nt.assert_allclose(sol.q, self.q, atol=1e-6)

    def test_prismatic(self):
        # the Stanford arm has a prismatic joint, and a spherical wrist
        q = np.array([0.3, -0.8, 0.5, 0.4, -0.6, 0.9])
        T = self.stanford.fkine(q)
        sol = self.stanford.ikine_analytic(T, q0=q + 0.02)
        self.assertTrue(sol)
        nt.assert_allclose(sol.q, q, atol=1e-6)

    def test_general(self):
        q = np.array([0.3, -0.8, 1.1, 0.4, -0.6, 0.9])
        T = self.jaco.fkine(q)
        ik = self.jaco.analytic_ik()
        sol = ik.solve(T, joint_limits=False)
        self.assertTrue(sol)
        self.check_fk(self.jaco, sol, T)
        self.assertTrue(any(np.allclose(row, q, atol=1e-5) for row in sol))

        sol = self.jaco.ikine_analytic(T, q0=q, joint_limits=False)
        nt.assert_allclose(sol.q, q, atol=1e-5)

    def test_generate(self):
        for robot in (self.puma, self.jaco):
            ik = robot.analytic_ik()
            for language in ("python", "c", "c++"):
                text = ik.generate(language, "my_ik")
                self.assertIn("my_ik", text)
                self.assertIn("Peter Corke", text)  # the provenance header

    def test_generated_c_is_for_embedded_use(self):
        # no dynamic memory, input or output, or jumps, and only these libraries
        import re

        banned = (
            "malloc", "calloc", "realloc", "free", "alloca",
            "printf", "scanf", "puts", "fopen",
            "goto", "setjmp", "longjmp", "abort", "exit",
        )  # fmt: skip
        for robot in (self.puma, self.stanford, self.jaco):
            source = robot.analytic_ik().generate("c", "my_ik")
            code = re.sub(r"/\*.*?\*/", "", source, flags=re.S)  # not the comments
            code = re.sub(r"//.*", "", code)
            for word in banned:
                self.assertIsNone(re.search(rf"\b{word}\b", code), (robot.name, word))
            libraries = set(re.findall(r"#include\s*<([^>]+)>", code))
            self.assertLessEqual(libraries, {"math.h", "string.h"}, robot.name)

    def test_generated_python_agrees(self):
        T = self.puma.fkine(self.q)
        namespace: dict = {}
        exec(self.puma.analytic_ik().generate("python", "my_ik"), namespace)
        result = namespace["my_ik"](T.A)
        self.assertEqual(result.status.name, "OK")
        self.assertTrue(
            any(np.allclose(np.atleast_1d(q), self.q, atol=1e-6) for q in result.solutions)
        )

    def test_prismatic_arm_without_wrist(self):
        # RPRRPR, the arm of Raghavan and Roth, 1993, Appendix 1
        deg = np.pi / 180
        robot = rtb.DHRobot(
            [
                rtb.RevoluteDH(d=0.21, a=1.46, alpha=135 * deg),
                rtb.PrismaticDH(theta=65 * deg, a=0.56, alpha=78 * deg, qlim=[0, 1]),
                rtb.RevoluteDH(d=0.29, a=0.38, alpha=23 * deg),
                rtb.RevoluteDH(d=-0.54, a=0.56, alpha=46 * deg),
                rtb.PrismaticDH(theta=54 * deg, a=1.08, alpha=35 * deg, qlim=[0, 1]),
                rtb.RevoluteDH(d=0.48, a=0.67, alpha=47 * deg),
            ]
        )
        with self.assertRaisesRegex(NotImplementedError, "prismatic"):
            robot.analytic_ik()

    def test_missing_sympy(self):
        # the message names the extra, and nothing else needs SymPy
        robot = rtb.models.DH.Planar2()  # a robot nobody has made a solver for
        saved = {
            name: module
            for name, module in sys.modules.items()
            if name == "sympy"
            or name.startswith("sympy.")
            or name.startswith("roboticstoolbox.robot.symbolic_ik")
        }
        for name in saved:
            del sys.modules[name]
        sys.modules["sympy"] = None  # makes ``import sympy`` fail
        try:
            with self.assertRaisesRegex(ImportError, r"roboticstoolbox-python\[ik\]"):
                robot.analytic_ik()
        finally:
            del sys.modules["sympy"]
            sys.modules.update(saved)


if __name__ == "__main__":  # pragma nocover
    unittest.main()
