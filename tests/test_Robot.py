"""
@author: Peter Corke
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rp
import spatialmath as sm
import unittest


class TestRobot(unittest.TestCase):

    def test_init(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()

        with self.assertRaises(TypeError):
            rp.DHRobot([l0, l1], keywords=1)

        with self.assertRaises(TypeError):
            rp.Robot(l0)

        with self.assertRaises(TypeError):
            rp.Robot([l0, 1])

    def test_configurations_str(self):
        r = rp.models.DH.Puma560()
        r.configurations_str()

        r2 = rp.models.ETS.Frankie()
        r2.configurations_str()

    def test_dyntable(self):
        r = rp.models.DH.Puma560()
        r.dynamics()

    def test_linkcolormap(self):
        r = rp.models.DH.Puma560()
        r.linkcolormap()

        r.linkcolormap(['r', 'r', 'r', 'r', 'r', 'r'])

    def test_tool_error(self):
        r = rp.models.DH.Puma560()

        with self.assertRaises(ValueError):
            r.tool = 2

    def test_links(self):

        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH()
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])

        self.assertIs(r0[0], l0)
        self.assertIs(r0[1], l1)
        self.assertIs(r0[2], l2)
        self.assertIs(r0[3], l3)

    def test_ikine_LM(self):
        panda = rp.models.DH.Panda()
        q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        T = panda.fkine(q)
        Tt = sm.SE3([T, T])

        qr = [0.0342, 1.6482, 0.0312, 1.2658, -0.0734, 0.4836, 0.7489]

        sol1 = panda.ikine_LM(T)
        sol2 = panda.ikine_LM(Tt)
        sol3 = panda.ikine_LM(T, q0=[0, 1.4, 0, 1, 0, 0.5, 1])

        self.assertTrue(sol1.success)
        self.assertAlmostEqual(np.linalg.norm(
            T - panda.fkine(sol1.q)), 0, places=4)

        self.assertTrue(sol2.success[0])
        self.assertAlmostEqual(np.linalg.norm(
            T - panda.fkine(sol2.q[0, :])), 0, places=4)
        self.assertTrue(sol2.success[0])
        self.assertAlmostEqual(np.linalg.norm(
            T - panda.fkine(sol2.q[1, :])), 0, places=4)

        self.assertTrue(sol3.success)
        self.assertAlmostEqual(np.linalg.norm(
            T - panda.fkine(sol3.q)), 0, places=4)

        with self.assertRaises(ValueError):
            panda.ikine_LM(T, q0=[1, 2])

    def test_ikine_LM_mask(self):

        # simple RR manipulator, solve with mask

        l0 = rp.RevoluteDH(a=2.0)
        l1 = rp.RevoluteDH(a=1)

        r = rp.DHRobot([l0, l1])  # RR manipulator
        T = sm.SE3(-1, 2, 0)
        sol = r.ikine_LM(T, mask=[1, 1, 0, 0, 0, 0])

        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(
            T.t[:2] - r.fkine(sol.q).t[:2]), 0, places=4)

        # test initial condition search, drop iteration limit so it has to do
        # some searching
        sol = r.ikine_LM(
            T, mask=[1, 1, 0, 0, 0, 0],
            ilimit=8, search=True)

        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(
            T.t[:2] - r.fkine(sol.q).t[:2]), 0, places=4)

    def test_ikine_LM_transpose(self):
        # need to test this on a robot with squarish Jacobian, choose Puma

        r = rp.models.DH.Puma560()
        T = r.fkine(r.qn)

        sol = r.ikine_LM(T, transpose=0.5, ilimit=1000, tol=1e-6)

        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(T - r.fkine(sol.q)), 0, places=4)

    def test_ikine_con(self):
        panda = rp.models.DH.Panda()
        q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        T = panda.fkine(q)
        Tt = sm.SE3([T, T, T])

        # qr = [7.69161412e-04, 9.01409257e-01, -1.46372859e-02,
        #       -6.98000000e-02, 1.38978915e-02, 9.62104811e-01,
        #       7.84926515e-01]

        sol1 = panda.ikine_min(T, qlim=True, q0=np.zeros(7))
        sol2 = panda.ikine_min(Tt, qlim=True)

        self.assertTrue(sol1.success)
        self.assertAlmostEqual(np.linalg.norm(
            T - panda.fkine(sol1.q)), 0, places=4)
        nt.assert_array_almost_equal(
            T.A - panda.fkine(sol1.q).A, np.zeros((4, 4)), decimal=4)

        self.assertTrue(sol2[0].success)
        nt.assert_array_almost_equal(
            T.A - panda.fkine(sol2[0].q).A, np.zeros((4, 4)), decimal=4)
        self.assertTrue(sol2[1].success)
        nt.assert_array_almost_equal(
            T.A - panda.fkine(sol2[1].q).A, np.zeros((4, 4)), decimal=4)

    def test_ikine_unc(self):
        puma = rp.models.DH.Puma560()
        q = puma.qn
        T = puma.fkine(q)
        Tt = sm.SE3([T, T])

        sol1 = puma.ikine_min(Tt)
        sol2 = puma.ikine_min(T)
        sol3 = puma.ikine_min(T)

        self.assertTrue(sol1[0].success)
        nt.assert_array_almost_equal(
            T.A - puma.fkine(sol1[0].q).A, np.zeros((4, 4)), decimal=4)
        self.assertTrue(sol1[1].success)
        nt.assert_array_almost_equal(
            T.A - puma.fkine(sol1[1].q).A, np.zeros((4, 4)), decimal=4)

        self.assertTrue(sol2.success)
        nt.assert_array_almost_equal(
            T.A - puma.fkine(sol2.q).A, np.zeros((4, 4)), decimal=4)

        self.assertTrue(sol3.success)
        nt.assert_array_almost_equal(
            T.A - puma.fkine(sol3.q).A, np.zeros((4, 4)), decimal=4)

    # def test_plot_swift(self):
    #     r = rp.models.Panda()

    #     env = r.plot(r.q, block=False)
    #     env.close()


if __name__ == '__main__':  # pragma nocover
    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])

    # import roboticstoolbox as rtb
    # from spatialmath import SE3

    # l0 = rtb.RevoluteDH(a=2.0)
    # l1 = rtb.RevoluteDH(a=1)
    # # l1 = rp.PrismaticDH(theta=1.0)
    # # l2 = rp.PrismaticDH(theta=1, qlim=[0, 2])

    # # r0 = rtb.DHRobot([l0, l1])  # RP manipulator
    # r1 = rtb.DHRobot([l0, l1])  # RR manipulator
    # print(r1)
    # T = SE3(-1, 2, 0)
    # print(T)
    # sol = r1.ikine_LM(T, mask=[1, 1, 0, 0, 0, 0])
    # print(sol)
    # print(r1.fkine(sol.q))

    # # sol = r1.ikine_LM(T, transpose=0.05, tol=1e-2, ilimit=1000, mask=[1, 1, 0, 0, 0, 0])
    # # print(sol)
    # # print(r1.fkine(sol.q))

    # sol = r1.ikine_LM(
    #     T, mask=[1, 1, 0, 0, 0, 0],
    #     ilimit=8, search=True)
    # print(sol)

    # r = rtb.models.DH.Puma560()

    # T = r.fkine(r.qn)

    # sol = r.ikine_LM(T, transpose=0.5, ilimit=1000, tol=1e-6)
    # print(sol)
