"""
@author: Jesse Haviland
"""

import numpy.testing as nt
import roboticstoolbox as rtb
import numpy as np

# import spatialmath.base as sm
from spatialmath import SE3
from spatialmath.base import tr2jac
import unittest
import sympy
import pytest


class TestIK(unittest.TestCase):
    def test_IK_NR1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_NR2(self):

        q0 = [1, 2, 3, 4, 5, 6, 7]

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol)

        sol = solver.solve(panda, Tep, q0=q0)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_NR3(self):

        q0 = np.array([1, 2, 3, 4, 5, 6, 7])

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol)

        sol = solver.solve(panda, Tep, q0=q0)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_NR4(self):

        q0 = np.array(
            [
                [1.0, 2, 3, 4, 5, 6, 7],
                [1.0, 2, 1, 2, 1, 2, 1],
                [1.0, 2, 3, 1, 2, 3, 1],
                [2.0, 2, 2, 2, 2, 2, 2],
                [0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4],
            ]
        )

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol, slimit=5)

        sol = solver.solve(panda, Tep, q0=q0)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_NR5(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, tol=tol)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(tol, E)

    def test_IK_NR6(self):

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=False, slimit=1)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, False)

    def test_IK_NR7(self):

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -1.3, 0, 1.2, 0, 2.0, 0.1])

        solver = rtb.IK_NR(joint_limits=True, seed=0, pinv=True, slimit=2)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, False)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_IK_NR8(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR(
            joint_limits=True,
            seed=0,
            pinv=True,
            tol=tol,
            kq=0.01,
            km=1.0,
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_IK_LM1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM(
            method="chan", joint_limits=True, seed=0, tol=tol, kq=0.1, km=0.1
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_LM2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM(
            method="sugihara", k=0.0001, joint_limits=True, seed=0, tol=tol
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_LM3(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM(
            method="wampler", k=0.001, joint_limits=True, seed=0, tol=tol
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_IK_GN1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_GN(
            pinv=True, joint_limits=True, seed=0, tol=tol, kq=1.0, km=1.0
        )

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_GN2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_GN(pinv=True, joint_limits=True, seed=0, tol=tol)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_GN3(self):

        tol = 1e-6

        ur5 = rtb.models.UR5().ets()

        Tep = ur5.eval([0, -0.3, 0, -2.2, 0, 2.0])

        solver = rtb.IK_GN(pinv=False, joint_limits=True, seed=0, tol=tol)

        sol = solver.solve(ur5, Tep)

        self.assertEqual(sol.success, True)

        Tq = ur5.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_QP1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP(joint_limits=True, seed=0, tol=tol, kq=2.0, km=100.0)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_QP2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP(joint_limits=True, seed=0, tol=tol)

        sol = solver.solve(panda, Tep)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_IK_QP3(self):

        tol = 1e-6

        q0 = [0, 0, 0, 0, 0, 0, 0.0]

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP(
            joint_limits=True,
            seed=0,
            tol=tol,
            kq=1000.0,
            pi=4.0,
            ps=2.0,
            kj=10000.0,
            slimit=1,
        )

        sol = solver.solve(panda, Tep, q0=q0)

        self.assertEqual(sol.success, False)

    def test_ets_ikine_NR1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR()

        sol = panda.ikine_NR(Tep, pinv=True, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_ets_ikine_NR2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_NR()

        sol = panda.ikine_NR(Tep, pinv=True, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(tol, E)

    def test_ets_ikine_LM1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM()

        sol = panda.ikine_LM(Tep, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_ets_ikine_LM2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_LM()

        sol = panda.ikine_LM(Tep, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(tol, E)

    def test_ets_ikine_GN1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_GN()

        sol = panda.ikine_GN(Tep, pinv=True, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_ets_ikine_GN2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_GN()

        sol = panda.ikine_GN(Tep, pinv=True, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(tol, E)

    def test_ets_ikine_QP1(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP()

        sol = panda.ikine_QP(Tep, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep, Tq)

        self.assertGreater(tol, E)

    def test_ets_ikine_QP2(self):

        tol = 1e-6

        panda = rtb.models.Panda().ets()

        Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        solver = rtb.IK_QP()

        sol = panda.ikine_QP(Tep, tol=tol)

        self.assertEqual(sol.success, True)

        Tq = panda.eval(sol.q)

        _, E = solver.error(Tep.A, Tq)

        self.assertGreater(tol, E)


if __name__ == "__main__":

    unittest.main()
