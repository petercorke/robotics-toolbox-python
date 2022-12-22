#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rp
import spatialmath as sm
import unittest
import math


class TestDHRobot(unittest.TestCase):
    def test_DHRobot(self):
        l0 = rp.DHLink()
        rp.DHRobot([l0])

    def test_prismaticjoints(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH()
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])

        ans = [True, False, True, False]

        self.assertEqual(r0.prismaticjoints, ans)

    def test_revolutejoints(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH()
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])

        ans = [False, True, False, True]

        self.assertEqual(r0.revolutejoints, ans)

    def test_isprismatic(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH()
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])

        self.assertEqual(r0.isprismatic(0), True)
        self.assertEqual(r0.isprismatic(1), False)
        self.assertEqual(r0.isprismatic(2), True)
        self.assertEqual(r0.isprismatic(3), False)

    def test_isrevolute(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH()
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])

        ans = [False, True, False, True]

        self.assertEqual(r0.isrevolute(0), False)
        self.assertEqual(r0.isrevolute(1), True)
        self.assertEqual(r0.isrevolute(2), False)
        self.assertEqual(r0.isrevolute(3), True)

    def test_todegrees(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH()
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])
        q = np.array([np.pi, np.pi, np.pi, np.pi / 2.0])

        ans = np.array([np.pi, 180, np.pi, 90])

        nt.assert_array_almost_equal(r0.todegrees(q), ans)

    def test_toradians(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH()
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])
        q = np.array([np.pi, 180, np.pi, 90])
        r0.q = q

        ans = np.array([np.pi, np.pi, np.pi, np.pi / 2.0])

        nt.assert_array_almost_equal(r0.toradians(q), ans)

    def test_d(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH(d=2.0)
        l2 = rp.PrismaticDH()
        l3 = rp.RevoluteDH(d=4.0)

        r0 = rp.DHRobot([l0, l1, l2, l3])
        ans = [0.0, 2.0, 0.0, 4.0]

        self.assertEqual(r0.d, ans)

    def test_a(self):
        l0 = rp.PrismaticDH(a=1.0)
        l1 = rp.RevoluteDH(a=2.0)
        l2 = rp.PrismaticDH(a=3.0)
        l3 = rp.RevoluteDH(a=4.0)

        r0 = rp.DHRobot([l0, l1, l2, l3])
        ans = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(r0.a, ans)

    def test_theta(self):
        l0 = rp.PrismaticDH(theta=1.0)
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH(theta=3.0)
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])
        ans = [1.0, 0.0, 3.0, 0.0]

        self.assertEqual(r0.theta, ans)

    def test_r(self):
        r = np.r_[1, 2, 3]
        l0 = rp.PrismaticDH(r=r)
        l1 = rp.RevoluteDH(r=r)
        l2 = rp.PrismaticDH(r=r)
        l3 = rp.RevoluteDH(r=r)

        r0 = rp.DHRobot([l0, l1, l2, l3])
        r1 = rp.DHRobot([l0])
        ans = np.c_[r, r, r, r]

        nt.assert_array_almost_equal(r0.r, ans)
        nt.assert_array_almost_equal(r1.r, r.flatten())

    def test_offset(self):
        l0 = rp.PrismaticDH(offset=1.0)
        l1 = rp.RevoluteDH(offset=2.0)
        l2 = rp.PrismaticDH(offset=3.0)
        l3 = rp.RevoluteDH(offset=4.0)

        r0 = rp.DHRobot([l0, l1, l2, l3])
        ans = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(r0.offset, ans)

    def test_qlim(self):
        qlim = [-1, 1]
        l0 = rp.PrismaticDH(qlim=qlim)
        l1 = rp.RevoluteDH(qlim=qlim)
        l2 = rp.PrismaticDH(qlim=qlim)
        l3 = rp.RevoluteDH(qlim=qlim)

        r0 = rp.DHRobot([l0, l1, l2, l3])
        r1 = rp.DHRobot([l0])
        ans = np.c_[qlim, qlim, qlim, qlim]

        nt.assert_array_almost_equal(r0.qlim, ans)
        nt.assert_array_almost_equal(r1.qlim, np.c_[qlim])

    def test_fkine(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH(theta=2.0)
        l3 = rp.RevoluteDH()

        q = np.array([1, 2, 3, 4])

        T1 = np.array(
            [
                [-0.14550003, -0.98935825, 0, 0],
                [0.98935825, -0.14550003, 0, 0],
                [0, 0, 1, 4],
                [0, 0, 0, 1],
            ]
        )

        r0 = rp.DHRobot([l0, l1, l2, l3])

        nt.assert_array_almost_equal(r0.fkine(q).A, T1)

    def test_fkine_traj(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH(theta=2.0)
        l3 = rp.RevoluteDH()

        q = np.array([1, 2, 3, 4])
        qq = np.r_[q, q, q, q]

        r0 = rp.DHRobot([l0, l1, l2, l3])

        T1 = r0.fkine(q).A
        TT = r0.fkine(qq)

        nt.assert_array_almost_equal(TT[0].A, T1)
        nt.assert_array_almost_equal(TT[1].A, T1)
        nt.assert_array_almost_equal(TT[2].A, T1)
        nt.assert_array_almost_equal(TT[3].A, T1)

    def test_links(self):
        l0 = rp.PrismaticDH()
        with self.assertRaises(TypeError):
            rp.DHRobot(l0)

    def test_multiple(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH(theta=2.0)
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1])
        r1 = rp.DHRobot([l2, l3])
        r3 = rp.DHRobot([r0, r1])
        r4 = rp.DHRobot([r0, l2, l3])

        q = np.array([1, 2, 3, 4])

        T1 = np.array(
            [
                [-0.14550003, -0.98935825, 0, 0],
                [0.98935825, -0.14550003, 0, 0],
                [0, 0, 1, 4],
                [0, 0, 0, 1],
            ]
        )

        nt.assert_array_almost_equal(r3.fkine(q).A, T1)
        nt.assert_array_almost_equal(r4.fkine(q).A, T1)

    def test_bad_list(self):
        l0 = rp.PrismaticDH()

        with self.assertRaises(TypeError):
            rp.DHRobot([l0, 1])

    def test_add_DHRobot(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH(theta=2.0)
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1])
        r1 = rp.DHRobot([l2, l3])
        r3 = r0 + r1

        q = np.array([1, 2, 3, 4])

        T1 = np.array(
            [
                [-0.14550003, -0.98935825, 0, 0],
                [0.98935825, -0.14550003, 0, 0],
                [0, 0, 1, 4],
                [0, 0, 0, 1],
            ]
        )

        nt.assert_array_almost_equal(r3.fkine(q).A, T1)

    def test_add_links(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        l2 = rp.PrismaticDH(theta=2.0)
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1])
        r1 = rp.DHRobot([l1, l2, l3])
        r3 = r0 + l2 + l3
        r4 = l0 + r1

        q = np.array([1, 2, 3, 4])

        T1 = np.array(
            [
                [-0.14550003, -0.98935825, 0, 0],
                [0.98935825, -0.14550003, 0, 0],
                [0, 0, 1, 4],
                [0, 0, 0, 1],
            ]
        )

        nt.assert_array_almost_equal(r3.fkine(q).A, T1)
        nt.assert_array_almost_equal(r4.fkine(q).A, T1)

    def test_add_error(self):
        l0 = rp.PrismaticDH()
        l1 = rp.RevoluteDH()
        r0 = rp.DHRobot([l0, l1])

        with self.assertRaises(TypeError):
            r0 + 2

    def test_dh_error(self):
        l0 = rp.PrismaticMDH()
        l1 = rp.RevoluteDH()
        r0 = rp.DHRobot([l0])
        r1 = rp.DHRobot([l1])

        with self.assertRaises(ValueError):
            rp.DHRobot([l0, l1])

        with self.assertRaises(ValueError):
            r0 + r1

        with self.assertRaises(ValueError):
            rp.DHRobot([l0, l1])
            r0 + l1

    def test_name(self):
        panda = rp.models.DH.Panda()

        panda.name = "new"
        self.assertEqual(panda.name, "new")

    def test_base(self):
        panda = rp.models.DH.Panda()

        panda.base = sm.SE3.Rx(2)
        nt.assert_array_almost_equal(panda.base.A, sm.SE3.Rx(2).A)

        panda.base = sm.SE3.Ty(2)
        nt.assert_array_almost_equal(panda.base.A, sm.SE3.Ty(2).A)

    def test_A(self):
        panda = rp.models.DH.Panda()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        T1 = np.array(
            [
                [0.5403, -0.8415, 0, 0],
                [0.8415, 0.5403, 0, 0],
                [0, 0, 1, 0.333],
                [0, 0, 0, 1],
            ]
        )

        T2 = np.array(
            [
                [-0.3279, -0.9015, -0.2826, 0.2918],
                [0.9232, -0.3693, 0.1068, 0.06026],
                [-0.2006, -0.2258, 0.9533, 0.3314],
                [0, 0, 0, 1],
            ]
        )

        nt.assert_array_almost_equal(panda.A(0, q).A, T1, decimal=4)
        nt.assert_array_almost_equal(panda.A([1, 4], q).A, T2, decimal=4)

    def test_A_error(self):
        panda = rp.models.DH.Panda()
        q = [1, 2, 3, 4, 5, 6, 7]

        with self.assertRaises(ValueError):
            panda.A(7, q).A

    def test_islimit(self):
        panda = rp.models.DH.Panda()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        ans = np.r_[False, True, True, True, True, True, True]

        nt.assert_array_equal(panda.islimit(q), ans)
        nt.assert_array_equal(panda.islimit(), ans)

    def test_isspherical(self):
        l0 = rp.RevoluteDH()
        l1 = rp.RevoluteDH(alpha=-np.pi / 2)
        l2 = rp.RevoluteDH(alpha=np.pi / 2)
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])
        r1 = rp.DHRobot([l0, l1])
        r2 = rp.DHRobot([l1, l2, l3, l0])

        self.assertTrue(r0.isspherical())
        self.assertFalse(r1.isspherical())
        self.assertFalse(r2.isspherical())

    def test_payload(self):
        panda = rp.models.DH.Panda()
        nt.assert_array_almost_equal(panda.r[:, 6], np.zeros(3))
        # nt.assert_array_almost_equal(panda.links[6].m, 0)

        m = 6
        p = [1, 2, 3]
        panda.payload(m, p)

        nt.assert_array_almost_equal(panda.r[:, 6], p)
        nt.assert_array_almost_equal(panda.links[6].m, m)

    def test_jointdynamics(self):
        puma = rp.models.DH.Puma560()
        jd = puma.jointdynamics(puma.qn)
        print(jd[0])
        # numbers come from MATLAB
        nt.assert_array_almost_equal(jd[0][1], [0.001133478453251, 0.001480000000000])
        nt.assert_array_almost_equal(
            jd[1][1], [0.579706964030143e-3, 0.817000000000000e-3]
        )
        nt.assert_array_almost_equal(jd[2][1], [0.000525146448377, 0.001380000000000])

    def test_twists(self):
        # TODO
        panda = rp.models.DH.Panda()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        panda.twists()
        panda.twists(q)

        puma = rp.models.DH.Puma560()
        q = [1, 2, 3, 4, 5, 6]
        puma.q = q

        puma.twists()
        puma.twists(q)

        l0 = rp.PrismaticMDH()
        r = rp.DHRobot([l0])
        r.twists()

        l0 = rp.PrismaticDH()
        l1 = rp.PrismaticDH()
        r = rp.DHRobot([l0, l1])
        r.twists()

    def test_fkine_panda(self):
        panda = rp.models.DH.Panda()
        q = [1, 2, 3, 4, 5, 6, 7]

        T = np.array(
            [
                [-0.8583, 0.1178, 0.4994, 0.1372],
                [0.1980, 0.9739, 0.1106, 0.3246],
                [-0.4734, 0.1938, -0.8593, 0.4436],
                [0, 0, 0, 1],
            ]
        )

        nt.assert_array_almost_equal(panda.fkine(q).A, T, decimal=4)

    def test_jacobe(self):
        l0 = rp.PrismaticDH(theta=4)
        l1 = rp.RevoluteDH(a=2)
        l2 = rp.PrismaticDH(theta=2)
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])
        q = [1, 2, 3, 4]
        r0.q = q

        Je = np.array(
            [
                [0, -0.5588, 0, 0],
                [0, 1.9203, 0, 0],
                [1.0000, 0, 1.0000, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1.0000, 0, 1.0000],
            ]
        )

        nt.assert_array_almost_equal(r0.jacobe(q), Je, decimal=4)

    def test_jacob0(self):
        l0 = rp.PrismaticDH(theta=4)
        l1 = rp.RevoluteDH(a=2)
        l2 = rp.PrismaticDH(theta=2)
        l3 = rp.RevoluteDH()

        r0 = rp.DHRobot([l0, l1, l2, l3])
        q = [1, 2, 3, 4]
        r0.q = q

        J0 = np.array(
            [
                [0, 0.5588, 0, 0],
                [0, 1.9203, 0, 0],
                [1.0000, 0, 1.0000, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1.0000, 0, 1.0000],
            ]
        )

        nt.assert_array_almost_equal(r0.jacob0(q), J0, decimal=4)

    def test_jacobe_panda(self):
        panda = rp.models.DH.Panda()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        Je = np.array(
            [
                [0.3058, 0.1315, -0.2364, -0.0323, 0.0018, 0.2095, 0],
                [0.0954, 0.0303, -0.0721, 0.1494, -0.0258, 0.0144, 0],
                [-0.1469, 0.3385, 0.0506, 0.0847, -0.0000, -0.0880, 0],
                [-0.4734, 0.8292, -0.0732, 0.8991, -0.2788, -0.0685, 0],
                [0.1938, 0.4271, 0.7224, 0.3461, -0.0191, 0.9976, 0],
                [-0.8593, -0.3605, 0.6876, -0.2679, -0.9602, 0.0000, 1.0000],
            ]
        )

        nt.assert_array_almost_equal(panda.jacobe(q), Je, decimal=4)

    # def test_jacob0v(self):
    #     l0 = rp.PrismaticDH(theta=4)
    #     l1 = rp.RevoluteDH(a=2)
    #     l2 = rp.PrismaticDH(theta=2)
    #     l3 = rp.RevoluteDH()

    #     r0 = rp.DHRobot([l0, l1, l2, l3])
    #     q = [1, 2, 3, 4]
    #     r0.q = q

    #     J = np.array([
    #         [0.8439, 0.5366, 0, 0, 0, 0],
    #         [-0.5366, 0.8439, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 0.8439, 0.5366, 0],
    #         [0, 0, 0, -0.5366, 0.8439, 0],
    #         [0, 0, 0, 0, 0, 1],
    #     ])

    #     nt.assert_array_almost_equal(r0.jacob0v(q), J, decimal=4)
    #     nt.assert_array_almost_equal(r0.jacob0v(), J, decimal=4)

    # def test_jacobev(self):
    #     l0 = rp.PrismaticDH(theta=4)
    #     l1 = rp.RevoluteDH(a=2)
    #     l2 = rp.PrismaticDH(theta=2)
    #     l3 = rp.RevoluteDH()

    #     r0 = rp.DHRobot([l0, l1, l2, l3])
    #     q = [1, 2, 3, 4]
    #     r0.q = q

    #     J = np.array([
    #         [0.8439, -0.5366, 0, 0, 0, 0],
    #         [0.5366, 0.8439, 0, 0, 0, 0],
    #         [0, 0, 1, 0, 0, 0],
    #         [0, 0, 0, 0.8439, -0.5366, 0],
    #         [0, 0, 0, 0.5366, 0.8439, 0],
    #         [0, 0, 0, 0, 0, 1],
    #     ])

    #     nt.assert_array_almost_equal(r0.jacobev(q), J, decimal=4)
    #     nt.assert_array_almost_equal(r0.jacobev(), J, decimal=4)

    def test_nofriction(self):
        l0 = rp.DHLink(Tc=2, B=3)
        l1 = rp.DHLink(Tc=2, B=3)
        l2 = rp.DHLink(Tc=2, B=3)
        l3 = rp.DHLink(Tc=2, B=3)
        L = [l0, l1, l2, l3]

        r0 = rp.DHRobot(L)

        n0 = r0.nofriction()
        n1 = r0.nofriction(viscous=True)
        n2 = r0.nofriction(coulomb=False)

        for i in range(4):
            nt.assert_array_almost_equal(n0.links[i].B, L[i].B)
            nt.assert_array_almost_equal(n0.links[i].Tc, [0, 0])

            nt.assert_array_almost_equal(n1.links[i].B, 0)
            nt.assert_array_almost_equal(n1.links[i].Tc, [0, 0])

            nt.assert_array_almost_equal(n2.links[i].B, L[i].B)
            nt.assert_array_almost_equal(n2.links[i].Tc, L[i].Tc)

    @unittest.skip("payload needs fixing")
    def test_pay(self):
        panda = rp.models.DH.Panda()
        q = [1, 2, 3, 4, 5, 6, 7]

        w = [1, 2, 3, 4, 5, 6]

        wT = np.c_[w, w, w, w].T
        qT = np.c_[q, q, q, q].T

        tau = np.array([6.0241, -4.4972, -7.2160, -4.2400, 7.0215, -4.6884, -6.0000])

        tau0 = np.array([-5.9498, 1.4604, -3.4544, 1.5026, -3.7777, -6.6578, 2.6047])

        tauT = np.c_[tau, tau, tau, tau].T
        tau0T = np.c_[tau0, tau0, tau0, tau0].T

        Je = panda.jacobe(q)
        J0 = panda.jacob0(q)

        JeT = np.zeros((4, 6, 7))
        for i in range(4):
            JeT[i, :, :] = Je

        panda.pay(w)

        nt.assert_array_almost_equal(panda.pay(w), tau, decimal=4)
        nt.assert_array_almost_equal(panda.pay(w, frame=0), tau0, decimal=4)

        nt.assert_array_almost_equal(panda.pay(w, q=q), tau, decimal=4)
        nt.assert_array_almost_equal(panda.pay(wT, q=qT), tauT, decimal=4)
        nt.assert_array_almost_equal(panda.pay(wT, q=qT, frame=0), tau0T, decimal=4)

        nt.assert_array_almost_equal(panda.pay(w, J=Je), tau, decimal=4)
        nt.assert_array_almost_equal(panda.pay(w, J=J0), tau0, decimal=4)

        nt.assert_array_almost_equal(panda.pay(wT, J=JeT), tauT, decimal=4)

        with self.assertRaises(ValueError):
            panda.pay(wT, q)

        with self.assertRaises(TypeError):
            panda.pay(wT)

    def test_friction(self):
        l0 = rp.RevoluteDH(d=2, B=3, G=2, Tc=[2, -1])
        qd = [1, 2, 3, 4]

        r0 = rp.DHRobot([l0, l0.copy(), l0.copy(), l0.copy()])

        tau = np.array([-16, -28, -40, -52])

        nt.assert_array_almost_equal(r0.friction(qd), tau)

    def test_fkine_all(self):
        panda = rp.models.DH.Panda()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        t0 = np.eye(4)
        t1 = np.array(
            [
                [0.5403, -0.8415, 0, 0],
                [0.8415, 0.5403, 0, 0],
                [0, 0, 1, 0.333],
                [0, 0, 0, 1],
            ]
        )
        t2 = np.array(
            [
                [-0.2248, -0.4913, -0.8415, 0],
                [-0.3502, -0.7651, 0.5403, 0],
                [-0.9093, 0.4161, 0, 0.333],
                [0, 0, 0, 1],
            ]
        )
        t3 = np.array(
            [
                [0.1038, 0.8648, 0.4913, 0.1552],
                [0.4229, -0.4855, 0.7651, 0.2418],
                [0.9002, 0.1283, -0.4161, 0.2015],
                [0, 0, 0, 1],
            ]
        )
        t4 = np.array(
            [
                [-0.4397, -0.2425, -0.8648, 0.1638],
                [-0.8555, -0.1801, 0.4855, 0.2767],
                [-0.2735, 0.9533, -0.1283, 0.2758],
                [0, 0, 0, 1],
            ]
        )
        t5 = np.array(
            [
                [-0.9540, -0.1763, -0.2425, 0.107],
                [0.2229, -0.9581, -0.1801, 0.2781],
                [-0.2006, -0.2258, 0.9533, 0.6644],
                [0, 0, 0, 1],
            ]
        )
        t6 = np.array(
            [
                [-0.8482, -0.4994, 0.1763, 0.107],
                [0.2643, -0.1106, 0.9581, 0.2781],
                [-0.4590, 0.8593, 0.2258, 0.6644],
                [0, 0, 0, 1],
            ]
        )
        t7 = np.array(
            [
                [-0.5236, 0.6902, 0.4994, 0.08575],
                [0.8287, 0.5487, 0.1106, 0.3132],
                [-0.1977, 0.4718, -0.8593, 0.5321],
                [0, 0, 0, 1],
            ]
        )

        Tall = panda.fkine_all(q)

        nt.assert_array_almost_equal(Tall[0].A, t0, decimal=4)
        nt.assert_array_almost_equal(Tall[1].A, t1, decimal=4)
        nt.assert_array_almost_equal(Tall[2].A, t2, decimal=4)
        nt.assert_array_almost_equal(Tall[3].A, t3, decimal=4)
        nt.assert_array_almost_equal(Tall[4].A, t4, decimal=4)
        nt.assert_array_almost_equal(Tall[5].A, t5, decimal=4)
        nt.assert_array_almost_equal(Tall[6].A, t6, decimal=4)
        nt.assert_array_almost_equal(Tall[7].A, t7, decimal=4)

    # def test_gravjac(self):
    #     l0 = rp.RevoluteDH(d=2, B=3, G=2, Tc=[2, -1], alpha=0.4, a=0.2,
    #                      r=[0.1, 0.2, 0.05], m=0.5)
    #     l1 = rp.PrismaticDH(theta=0.1, B=3, G=2, Tc=[2, -1], a=0.2,
    #                       r=[0.1, 0.2, 0.05], m=0.5)

    #     r0 = rp.DHRobot([l0, l0, l0, l0])
    #     r1 = rp.DHRobot([l0, l0, l0, l1])
    #     q = [0.3, 0.4, 0.2, 0.1]
    #     qT = np.c_[q, q]

    #     grav = [0.3, 0.5, 0.7]

    #     tauB = [0, 4.6280, 3.1524, 0.9324]
    #     tauB2 = [1.9412, 1.1374, 0.3494, -0.0001]
    #     tauB3 = [0, 3.2819, 2.0195, 1.9693]

    #     res0 = r0.gravjac(qT)
    #     res1 = r0.gravjac(q)
    #     res2 = r0.gravjac(q, grav)
    #     res4 = r1.gravjac(q)

    #     nt.assert_array_almost_equal(res0[:, 0], tauB, decimal=4)
    #     nt.assert_array_almost_equal(res0[:, 1], tauB, decimal=4)
    #     nt.assert_array_almost_equal(res1, tauB, decimal=4)
    #     nt.assert_array_almost_equal(res2, tauB2, decimal=4)
    #     nt.assert_array_almost_equal(res3, tauB, decimal=4)
    #     nt.assert_array_almost_equal(res4, tauB3, decimal=4)

    # def test_ikine3(self):
    #     l0 = rp.RevoluteDH(alpha=np.pi / 2)
    #     l1 = rp.RevoluteDH(a=0.4318)
    #     l2 = rp.RevoluteDH(d=0.15005, a=0.0203, alpha=-np.pi / 2)
    #     l3 = rp.PrismaticDH()
    #     l4 = rp.PrismaticMDH()
    #     r0 = rp.DHRobot([l0, l1, l2])
    #     r1 = rp.DHRobot([l3, l3])
    #     r2 = rp.DHRobot([l3, l3, l3])
    #     r3 = rp.DHRobot([l4, l4, l4])

    #     q = [1, 1, 1]
    #     r0.q = q
    #     T = r0.fkine(q)
    #     # T2 = r1.fkine(q)
    #     Tt = sm.SE3([T, T])

    #     res = [2.9647, 1.7561, 0.2344]
    #     res2 = [1.0000, 0.6916, 0.2344]
    #     res3 = [2.9647, 2.4500, 3.1762]
    #     res4 = [1.0000, 1.3855, 3.1762]

    #     q0 = r0.ikine3(T.A)
    #     q1 = r0.ikine3(Tt)
    #     q2 = r0.ikine3(T, left=False, elbow_up=False)
    #     q3 = r0.ikine3(T, elbow_up=False)
    #     q4 = r0.ikine3(T, left=False)

    #     nt.assert_array_almost_equal(q0, res, decimal=4)
    #     nt.assert_array_almost_equal(q1[0, :], res, decimal=4)
    #     nt.assert_array_almost_equal(q1[1, :], res, decimal=4)
    #     nt.assert_array_almost_equal(q2, res2, decimal=4)
    #     nt.assert_array_almost_equal(q3, res3, decimal=4)
    #     nt.assert_array_almost_equal(q4, res4, decimal=4)

    #     with self.assertRaises(ValueError):
    #         r1.ikine3(T)

    #     with self.assertRaises(ValueError):
    #         r2.ikine3(T)

    #     with self.assertRaises(ValueError):
    #         r3.ikine3(T)

    # def test_ikine6s_rrp(self):
    #     l0 = rp.RevoluteDH(alpha=-np.pi / 2)
    #     l1 = rp.RevoluteDH(alpha=np.pi / 2)
    #     l2 = rp.PrismaticDH()
    #     l3 = rp.RevoluteDH(alpha=-np.pi / 2)
    #     l4 = rp.RevoluteDH(alpha=np.pi / 2)
    #     l5 = rp.RevoluteDH()
    #     r0 = rp.DHRobot([l0, l1, l2, l3, l4, l5])
    #     r1 = rp.DHRobot([l1, l0, l2, l3, l4, l5])
    #     q = [1, 1, 1, 1, 1, 1]
    #     T1 = r0.fkine(q)
    #     T2 = r1.fkine(q)

    #     qr0 = [1.0000, -2.1416, -1.0000, -1.0000, -2.1416, 1.0000]
    #     qr1 = [-2.1416, -1.0000, 1.0000, -2.1416, 1.0000, 1.0000]
    #     qr2 = [1.0000, 1.0000, 1.0000, -2.1416, -1.0000, -2.1416]
    #     qr3 = [-2.1416, 2.1416, -1.0000, -1.0000, 2.1416, -2.1416]

    #     q0, _ = r0.ikine6s(T1)
    #     q1, _ = r0.ikine6s(T1, left=False, elbow_up=False, wrist_flip=True)
    #     q2, _ = r1.ikine6s(T2)
    #     q3, _ = r1.ikine6s(T2, left=False, elbow_up=False, wrist_flip=True)

    #     nt.assert_array_almost_equal(q0, qr0, decimal=4)
    #     nt.assert_array_almost_equal(q1, qr1, decimal=4)
    #     nt.assert_array_almost_equal(q2, qr2, decimal=4)
    #     nt.assert_array_almost_equal(q3, qr3, decimal=4)

    # def test_ikine6s_simple(self):
    #     l0 = rp.RevoluteDH(alpha=-np.pi / 2)
    #     l1 = rp.RevoluteDH()
    #     l2 = rp.RevoluteDH(alpha=np.pi / 2)
    #     l3 = rp.RevoluteDH(alpha=-np.pi / 2)
    #     l4 = rp.RevoluteDH(alpha=np.pi / 2)
    #     l5 = rp.RevoluteDH()
    #     r0 = rp.DHRobot([l0, l1, l2, l3, l4, l5])
    #     r1 = rp.DHRobot([l2, l1, l0, l3, l4, l5])
    #     q = [1, 1, 1, 1, 1, 1]
    #     T1 = r0.fkine(q)
    #     T2 = r1.fkine(q)

    #     qr0 = [0, 0, 0, -0.9741, -2.2630, -0.4605]
    #     qr1 = [0, 0, 0, 0.1947, -1.3811, 1.8933]
    #     qr2 = [0, 0, 0, 2.1675, 2.2630, 2.6811]
    #     qr3 = [0, 0, 0, -2.9468, 1.3811, -1.2483]

    #     q0, _ = r0.ikine6s(T1)
    #     q1, _ = r0.ikine6s(T1, left=False, elbow_up=False, wrist_flip=True)
    #     q2, _ = r1.ikine6s(T2)
    #     q3, _ = r1.ikine6s(T2, left=False, elbow_up=False, wrist_flip=True)

    #     nt.assert_array_almost_equal(q0, qr0, decimal=4)
    #     nt.assert_array_almost_equal(q1, qr2, decimal=4)
    #     nt.assert_array_almost_equal(q2, qr1, decimal=4)
    #     nt.assert_array_almost_equal(q3, qr3, decimal=4)

    # def test_ikine6s_offset(self):
    #     self.skipTest("error introduced with DHLink change")
    #     l0 = rp.RevoluteDH(alpha=-np.pi / 2)
    #     l1 = rp.RevoluteDH(d=1.0)
    #     l2 = rp.RevoluteDH(alpha=np.pi / 2)
    #     l3 = rp.RevoluteDH(alpha=-np.pi / 2)
    #     l4 = rp.RevoluteDH(alpha=np.pi / 2)
    #     l5 = rp.RevoluteDH()
    #     r0 = rp.DHRobot([l0, l1, l2, l3, l4, l5])
    #     r1 = rp.DHRobot([l2, l1, l0, l3, l4, l5])
    #     q = [1, 1, 1, 1, 1, 1]
    #     T1 = r0.fkine(q)
    #     T2 = r1.fkine(q)

    #     qr0 = [1.0000, 3.1416, -0.0000, -1.1675, -0.8786, 2.6811]
    #     qr1 = [1.0000, -1.1059, 2.6767, 0.8372, 1.2639, 1.3761]
    #     qr2 = [1.0000, 3.1416, -3.1416, -0.8053, -1.3811, 1.8933]
    #     qr3 = [1.0000, -1.1059, -0.4649, 1.8311, 2.3192, -2.6398]

    #     q0, _ = r0.ikine6s(T1.A)
    #     q1, _ = r0.ikine6s(T1, left=False, elbow_up=False, wrist_flip=True)
    #     q2, _ = r1.ikine6s(T2)
    #     q3, _ = r1.ikine6s(T2, left=False, elbow_up=False, wrist_flip=True)

    #     nt.assert_array_almost_equal(q0, qr0, decimal=4)
    #     nt.assert_array_almost_equal(q1, qr1, decimal=4)
    #     nt.assert_array_almost_equal(q2, qr2, decimal=4)
    #     nt.assert_array_almost_equal(q3, qr3, decimal=4)

    # def test_ikine6s_traj(self):
    #     self.skipTest("error introduced with DHLink change")
    #     r0 = rp.models.DH.Puma560()
    #     q = r0.qr
    #     T = r0.fkine(q)
    #     Tt = sm.SE3([T, T, T])

    #     qr0 = [0.2689, 1.5708, -1.4768, -3.1416, 0.0940, 2.8726]

    #     q0, _ = r0.ikine6s(Tt)

    #     nt.assert_array_almost_equal(q0[0, :], qr0, decimal=4)
    #     nt.assert_array_almost_equal(q0[1, :], qr0, decimal=4)Fikin
    #     nt.assert_array_almost_equal(q0[2, :], qr0, decimal=4)

    # def test_ikine6s_fail(self):
    #     l0 = rp.RevoluteDH(alpha=np.pi / 2)
    #     l1 = rp.RevoluteDH(d=1.0)
    #     l2 = rp.RevoluteDH(alpha=np.pi / 2)
    #     l3 = rp.RevoluteDH(alpha=-np.pi / 2)
    #     l4a = rp.RevoluteDH(alpha=np.pi / 2)
    #     l4b = rp.RevoluteDH()
    #     l5 = rp.RevoluteDH()
    #     l6 = rp.RevoluteMDH()
    #     r0 = rp.DHRobot([l0, l1, l2, l3, l4a, l5])
    #     r1 = rp.DHRobot([l0, l1, l2, l3, l4b, l5])
    #     r2 = rp.DHRobot([l1, l2, l3])
    #     r3 = rp.DHRobot([l6, l6, l6, l6, l6, l6])

    #     puma = rp.models.DH.Puma560()
    #     T = sm.SE3(0, 10, 10)
    #     puma.ikine6s(T)

    #     q = [1, 1, 1, 1, 1, 1]
    #     T = r0.fkine(q)

    #     with self.assertRaises(ValueError):
    #         r0.ikine6s(T)

    #     with self.assertRaises(ValueError):
    #         r1.ikine6s(T)

    #     with self.assertRaises(ValueError):
    #         r2.ikine6s(T)

    #     with self.assertRaises(ValueError):
    #         r3.ikine6s(T)

    def test_ikine_a(self):
        puma = rp.models.DH.Puma560()

        T = puma.fkine(puma.qn)

        # test configuration validation
        config = puma.config_validate("l", ("lr", "ud", "nf"))
        self.assertEqual(len(config), 3)
        self.assertTrue("l" in config)
        self.assertTrue("u" in config)
        self.assertTrue("n" in config)
        with self.assertRaises(ValueError):
            config = puma.config_validate("lux", ("lr", "ud", "nf"))

        # analytic solution
        sol = puma.ikine_a(T)
        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=6)

        sol = puma.ikine_a(T, "l")
        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=6)
        self.assertTrue(sol.q[0] > np.pi / 2)

        sol = puma.ikine_a(T, "r")
        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=6)
        self.assertTrue(sol.q[0] < np.pi / 2)

        sol = puma.ikine_a(T, "u")
        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=6)
        self.assertTrue(sol.q[1] > 0)

        sol = puma.ikine_a(T, "d")
        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=6)
        self.assertTrue(sol.q[1] < 0)

        sol = puma.ikine_a(T, "n")
        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=6)

        sol = puma.ikine_a(T, "f")
        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=6)

    def test_ikine_LM(self):
        puma = rp.models.DH.Puma560()

        T = puma.fkine(puma.qn)

        sol = puma.ikine_LM(T)
        self.assertTrue(sol.success)
        self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=4)

    # def test_ikine_LMS(self):
    #     puma = rp.models.DH.Puma560()

    #     T = puma.fkine(puma.qn)

    #     sol = puma.ikine_LM(T)
    #     self.assertTrue(sol.success)
    #     self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=6)

    # def test_ikine_unc(self):
    #     puma = rp.models.DH.Puma560()

    #     T = puma.fkine(puma.qn)

    #     sol = puma.ikine_min(T)
    #     self.assertTrue(sol.success)
    #     self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=5)

    #     q0 = np.r_[0.1, 0.1, 0.1, 0.2, 0.3, 0.4]
    #     sol = puma.ikine_min(T, q0=q0)
    #     self.assertTrue(sol.success)
    #     self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=5)

    # def test_ikine_con(self):
    #     puma = rp.models.DH.Puma560()

    #     T = puma.fkine(puma.qn)

    #     sol = puma.ikine_min(T, qlim=True)
    #     self.assertTrue(sol.success)
    #     self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=5)

    #     q0 = np.r_[0.1, 0.1, 0.1, 0.2, 0.3, 0.4]
    #     sol = puma.ikine_min(T, q0=q0, qlim=True)
    #     self.assertTrue(sol.success)
    #     self.assertAlmostEqual(np.linalg.norm(T - puma.fkine(sol.q)), 0, places=5)

    # def test_ikine_min(self):
    #     puma = rp.models.DH.Puma560()
    #     q = puma.qn
    #     T = puma.fkine(q)
    #     Tt = sm.SE3([T, T])

    #     sol0 = puma.ikine_min(Tt)
    #     sol1 = puma.ikine_min(T.A, qlimits=False)
    #     sol2 = puma.ikine_min(
    #           T, qlimits=False, stiffness=0.1, ilimit=1)

    # print(np.sum(np.abs(T.A - puma.fkine(q0[:, 0]).A)))

    # self.assertTrue(sol0[0].success)
    # self.assertAlmostEqual(np.linalg.norm(T-puma.fkine(sol0[0].q)), 0, places=4)
    # TODO: second solution fails, even though starting value is the
    # solution.  see https://stackoverflow.com/questions/34663539/scipy-optimize-fmin-l-bfgs-b-returns-abnormal-termination-in-lnsrch
    # documentation is pretty bad.
    # self.assertTrue(sol0[1].success)
    # self.assertAlmostEqual(np.linalg.norm(T-puma.fkine(sol0[1].q)), 0, places=4)
    # self.assertTrue(sol1.success)
    # self.assertAlmostEqual(np.linalg.norm(T-puma.fkine(sol1.q)), 0, places=4)
    # self.assertTrue(sol2.success)
    # self.assertAlmostEqual(np.linalg.norm(T-puma.fkine(sol2.q)), 0, places=4)

    def test_rne(self):
        puma = rp.models.DH.Puma560()

        z = np.zeros(6)
        o = np.ones(6)
        fext = [1, 2, 3, 1, 2, 3]

        tr0 = [-0.0000, 31.6399, 6.0351, 0.0000, 0.0283, 0]
        tr1 = [3.35311, 36.0025, 7.42596, 0.190043, 0.203441, 0.194133]
        tr2 = [32.4952, 60.867, 17.7436, 1.45452, 1.29911, 0.713781]
        tr3 = [29.1421, 56.5044, 16.3528, 1.26448, 1.12392, 0.519648]
        tr4 = [32.4952, 29.2271, 11.7085, 1.45452, 1.27086, 0.713781]
        tr5 = [0.642756, 29.0866, 4.70321, 2.82843, -1.97175, 3]

        t0 = puma.rne(puma.qn, z, z)
        t1 = puma.rne(puma.qn, z, o)
        t2 = puma.rne(puma.qn, o, o)
        t3 = puma.rne(puma.qn, o, z)
        t4 = puma.rne(puma.qn, o, o, gravity=[0, 0, 0])
        t5 = puma.rne(puma.qn, z, z, fext=fext)

        nt.assert_array_almost_equal(t0, tr0, decimal=4)
        nt.assert_array_almost_equal(t1, tr1, decimal=4)
        nt.assert_array_almost_equal(t2, tr2, decimal=4)
        nt.assert_array_almost_equal(t3, tr3, decimal=4)
        nt.assert_array_almost_equal(t4, tr4, decimal=4)
        nt.assert_array_almost_equal(t5, tr5, decimal=4)

    def test_rne_traj(self):
        puma = rp.models.DH.Puma560()

        z = np.zeros(6)
        o = np.ones(6)

        tr0 = [-0.0000, 31.6399, 6.0351, 0.0000, 0.0283, 0]
        tr1 = [32.4952, 60.8670, 17.7436, 1.4545, 1.2991, 0.7138]

        t0 = puma.rne(np.c_[puma.qn, puma.qn].T, np.c_[z, o].T, np.c_[z, o].T)

        nt.assert_array_almost_equal(t0[0, :], tr0, decimal=4)
        nt.assert_array_almost_equal(t0[1, :], tr1, decimal=4)

    def test_rne_delete(self):
        puma = rp.models.DH.Puma560()

        z = np.zeros(6)

        tr0 = [-0.0000, 31.6399, 6.0351, 0.0000, 0.0283, 0]

        t0 = puma.rne(puma.qn, z, z)
        puma.delete_rne()
        t1 = puma.rne(puma.qn, z, z)

        nt.assert_array_almost_equal(t0, tr0, decimal=4)
        nt.assert_array_almost_equal(t1, tr0, decimal=4)

    def test_accel(self):
        puma = rp.models.DH.Puma560()
        puma.q = puma.qn
        q = puma.qn

        qd = [0.1, 0.2, 0.8, 0.2, 0.5, 1.0]
        torque = [1.0, 3.2, 1.8, 0.1, 0.7, 4.6]

        res = [-7.4102, -9.8432, -10.9694, -4.4314, -0.9881, 21.0228]

        qdd0 = puma.accel(q, qd, torque)
        qdd1 = puma.accel(np.c_[q, q].T, np.c_[qd, qd].T, np.c_[torque, torque].T)

        nt.assert_array_almost_equal(qdd0, res, decimal=4)
        nt.assert_array_almost_equal(qdd1[0, :], res, decimal=4)
        nt.assert_array_almost_equal(qdd1[1, :], res, decimal=4)

    def test_inertia(self):
        puma = rp.models.DH.Puma560()
        puma.q = puma.qn
        q = puma.qn

        Ir = [
            [3.6594, -0.4044, 0.1006, -0.0025, 0.0000, -0.0000],
            [-0.4044, 4.4137, 0.3509, 0.0000, 0.0024, 0.0000],
            [0.1006, 0.3509, 0.9378, 0.0000, 0.0015, 0.0000],
            [-0.0025, 0.0000, 0.0000, 0.1925, 0.0000, 0.0000],
            [0.0000, 0.0024, 0.0015, 0.0000, 0.1713, 0.0000],
            [-0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1941],
        ]

        I0 = puma.inertia(q)
        # I1 = puma.inertia(np.c_[q, q].T)

        nt.assert_array_almost_equal(I0, Ir, decimal=4)
        # nt.assert_array_almost_equal(I1[0, :, :], Ir, decimal=4)
        # nt.assert_array_almost_equal(I1[1, :, :], Ir, decimal=4)

    def test_inertia_x(self):
        puma = rp.models.DH.Puma560()
        q = puma.qn

        Mr = [
            [17.2954, -2.7542, -9.6233, -0.0000, 0.2795, 0.0000],
            [-2.7542, 12.1909, 1.2459, -0.3254, -0.0703, -0.9652],
            [-9.6233, 1.2459, 13.3348, -0.0000, 0.2767, -0.0000],
            [-0.0000, -0.3254, -0.0000, 0.1941, 0.0000, 0.1941],
            [0.2795, -0.0703, 0.2767, 0.0000, 0.1713, 0.0000],
            [0.0000, -0.9652, -0.0000, 0.1941, 0.0000, 0.5791],
        ]

        M0 = puma.inertia_x(q, representation=None)
        M1 = puma.inertia_x(np.c_[q, q].T, representation=None)

        nt.assert_array_almost_equal(M0, Mr, decimal=4)
        nt.assert_array_almost_equal(M1[0, :, :], Mr, decimal=4)
        nt.assert_array_almost_equal(M1[1, :, :], Mr, decimal=4)

    def test_coriolis(self):
        puma = rp.models.DH.Puma560()
        q = puma.qn

        qd = [1, 2, 3, 1, 2, 3]

        Cr = [
            [-0.1735, -2.0494, -0.1178, -0.0002, -0.0045, 0.0001],
            [0.6274, 1.1572, 1.9287, -0.0015, -0.0003, -0.0000],
            [-0.3608, -0.7734, -0.0018, -0.0009, -0.0064, -0.0000],
            [0.0011, 0.0005, -0.0001, 0.0002, 0.0002, -0.0001],
            [-0.0002, 0.0028, 0.0046, -0.0002, -0.0000, -0.0000],
            [0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0],
        ]

        C0 = puma.coriolis(q, qd)
        C1 = puma.coriolis(np.c_[q, q].T, np.c_[qd, qd].T)

        nt.assert_array_almost_equal(C0, Cr, decimal=4)
        nt.assert_array_almost_equal(C1[0, :, :], Cr, decimal=4)
        nt.assert_array_almost_equal(C1[1, :, :], Cr, decimal=4)

    def test_gravload(self):
        puma = rp.models.DH.Puma560()
        q = puma.qn

        grav = [0, 0, 9.81]

        taur = [-0.0000, 31.6399, 6.0351, 0.0000, 0.0283, 0]

        tau0 = puma.gravload(q)
        tau1 = puma.gravload(np.c_[q, q].T)

        nt.assert_array_almost_equal(tau0, taur, decimal=4)
        nt.assert_array_almost_equal(tau1[0, :], taur, decimal=4)
        nt.assert_array_almost_equal(tau1[1, :], taur, decimal=4)

    def test_itorque(self):
        puma = rp.models.DH.Puma560()
        q = puma.qn

        qdd = [1, 2, 3, 1, 2, 3]

        tauir = [3.1500, 9.4805, 3.6189, 0.1901, 0.3519, 0.5823]

        taui0 = puma.itorque(q, qdd)
        taui1 = puma.itorque(np.c_[q, q].T, np.c_[qdd, qdd].T)

        nt.assert_array_almost_equal(taui0, tauir, decimal=4)
        nt.assert_array_almost_equal(taui1[0, :], tauir, decimal=4)
        nt.assert_array_almost_equal(taui1[1, :], tauir, decimal=4)

    #     def test_str(self):
    #         puma = rp.models.DH.Puma560()
    #         l0 = rp.PrismaticMDH()
    #         r0 = rp.DHRobot([l0, l0, l0])
    #         str(r0)

    #         res = """
    # Puma 560 (Unimation): 6 axis, RRRRRR, std DH
    # Parameters:
    # Revolute   theta=q1 + 0.00,  d= 0.67,  a= 0.00,  alpha= 1.57
    # Revolute   theta=q2 + 0.00,  d= 0.00,  a= 0.43,  alpha= 0.00
    # Revolute   theta=q3 + 0.00,  d= 0.15,  a= 0.02,  alpha=-1.57
    # Revolute   theta=q4 + 0.00,  d= 0.43,  a= 0.00,  alpha= 1.57
    # Revolute   theta=q5 + 0.00,  d= 0.00,  a= 0.00,  alpha=-1.57
    # Revolute   theta=q6 + 0.00,  d= 0.00,  a= 0.00,  alpha= 0.00

    # tool:  t = (0, 0, 0),  RPY/xyz = (0, 0, 0) deg"""

    #         self.assertEqual(str(puma), res)

    # def test_paycap(self):
    #     self.skipTest("error introduced with DHLink change")
    #     puma = rp.models.DH.Puma560()
    #     puma.q = puma.qn
    #     q = puma.qn

    #     w = [1, 2, 1, 2, 1, 2]
    #     tauR = np.ones((6, 2))
    #     tauR[:, 1] = -1

    #     res0 = [
    #         1.15865438e+00, -3.04790052e+02, -5.00870095e+01,  6.00479950e+15,
    #         3.76356072e+00, 1.93649167e+00]

    #     wmax0, joint = puma.paycap(w, tauR, q=q, frame=0)
    #     wmax1, _ = puma.paycap(np.c_[w, w], tauR, q=np.c_[q, q], frame=0)
    #     wmax2, _ = puma.paycap(w, tauR, frame=0)

    #     nt.assert_allclose(wmax0, res0)
    #     self.assertEqual(joint, 1)
    #     nt.assert_allclose(wmax1[:, 0], res0)
    #     nt.assert_allclose(wmax1[:, 1], res0)
    #     nt.assert_allclose(wmax2, res0)

    def test_jacob_dot(self):
        puma = rp.models.DH.Puma560()
        puma.q = puma.qr
        puma.qd = puma.qr
        q = puma.qn
        qd = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]

        from roboticstoolbox.tools import hessian_numerical

        j0 = puma.jacob0_dot(q, qd)

        H = hessian_numerical(lambda q: puma.jacob0(q), q)
        Jd = np.zeros((6, puma.n))
        for i in range(puma.n):
            Jd += H[:, :, i] * qd[i]

        nt.assert_array_almost_equal(j0, Jd, decimal=4)

    def test_yoshi(self):
        puma = rp.models.DH.Puma560()
        q = puma.qn

        m1 = puma.manipulability(q)
        m2 = puma.manipulability(np.c_[q, q].T)
        m3 = puma.manipulability(q, axes="trans")
        m4 = puma.manipulability(q, axes="rot")

        a0 = 0.0786
        a2 = 0.111181
        a3 = 2.44949

        nt.assert_almost_equal(m1, a0, decimal=4)
        nt.assert_almost_equal(m2[0], a0, decimal=4)
        nt.assert_almost_equal(m2[1], a0, decimal=4)
        nt.assert_almost_equal(m3, a2, decimal=4)
        nt.assert_almost_equal(m4, a3, decimal=4)

        with self.assertRaises(ValueError):
            puma.manipulability(axes="abcdef")

    def test_asada(self):
        puma = rp.models.DH.Puma560()
        q = puma.qn

        m1 = puma.manipulability(q, method="asada")
        m2 = puma.manipulability(np.c_[q, q].T, method="asada")
        m3 = puma.manipulability(q, axes="trans", method="asada")
        m4 = puma.manipulability(q, axes="rot", method="asada")
        m5 = puma.manipulability(puma.qz, method="asada")

        a0 = 0.0044
        a2 = 0.2094
        a3 = 0.1716
        a4 = 0.0

        # ax0 = np.array([
        #     [17.2954, -2.7542, -9.6233, -0.0000,  0.2795, -0.0000],
        #     [-2.7542, 12.1909,  1.2459, -0.3254, -0.0703, -0.9652],
        #     [-9.6233,  1.2459, 13.3348, -0.0000,  0.2767,  0.0000],
        #     [-0.0000, -0.3254, -0.0000,  0.1941,  0.0000,  0.1941],
        #     [0.2795, -0.0703,  0.2767,  0.0000,  0.1713,  0.0000],
        #     [-0.0000, -0.9652,  0.0000,  0.1941,  0.0000,  0.5791]
        # ])

        # ax1 = np.array([
        #     [17.2954, -2.7542, -9.6233],
        #     [-2.7542, 12.1909,  1.2459],
        #     [-9.6233,  1.2459, 13.3348]
        # ])

        # ax2 = np.array([
        #     [0.1941, 0.0000, 0.1941],
        #     [0.0000, 0.1713, 0.0000],
        #     [0.1941, 0.0000, 0.5791]
        # ])

        # ax3 = np.zeros((6, 6))

        # nt.assert_array_almost_equal(mx0, ax0, decimal=4)
        nt.assert_almost_equal(m1, a0, decimal=4)
        # nt.assert_array_almost_equal(mx1, ax0, decimal=4)
        nt.assert_almost_equal(m2[0], a0, decimal=4)
        # nt.assert_array_almost_equal(mx2[:, :, 0], ax0, decimal=4)
        nt.assert_almost_equal(m2[1], a0, decimal=4)
        # nt.assert_array_almost_equal(mx2[:, :, 1], ax0, decimal=4)

        nt.assert_almost_equal(m3, a2, decimal=4)
        # nt.assert_array_almost_equal(mx3, ax1, decimal=4)

        nt.assert_almost_equal(m4, a3, decimal=4)
        # nt.assert_array_almost_equal(mx4, ax2, decimal=4)

        nt.assert_almost_equal(m5, a4, decimal=4)
        # nt.assert_array_almost_equal(mx5, ax3, decimal=4)

    def test_manipulability_fail(self):
        puma = rp.models.DH.Puma560()
        puma.q = puma.qn

        with self.assertRaises(ValueError):
            puma.manipulability(method="notamethod")

    def test_perturb(self):
        puma = rp.models.DH.Puma560()
        p2 = puma.perturb()
        p3 = puma.perturb(0.8)

        resI0 = np.zeros(puma.n)
        resm0 = np.zeros(puma.n)
        resI1 = np.zeros(puma.n)
        resm1 = np.zeros(puma.n)

        for i in range(puma.n):
            resI0[i] = np.divide(
                np.sum(np.abs(puma.links[i].I - p2.links[i].I)),
                np.sum(np.abs(puma.links[i].I)),
            )

            if puma.links[i].m - p2.links[i].m != 0.0:
                resm0[i] = np.abs(
                    np.divide((puma.links[i].m - p2.links[i].m), puma.links[i].m)
                )
            else:
                resm0[i] = 0

            resI1[i] = np.divide(
                np.sum(np.abs(puma.links[i].I - p3.links[i].I)),
                np.sum(np.abs(puma.links[i].I)),
            )

            if puma.links[i].m - p3.links[i].m != 0.0:
                resm1[i] = np.abs(
                    np.divide((puma.links[i].m - p3.links[i].m), puma.links[i].m)
                )
            else:
                resm1[i] = 0

            self.assertTrue(resI0[i] < 0.1)
            self.assertTrue(resm0[i] < 0.1 or np.isnan(resm0[i]))
            self.assertTrue(resI1[i] < 0.8)
            self.assertTrue(resm1[i] < 0.8 or np.isnan(resm1[i]))

    # def test_qmincon(self):
    #     panda = rp.models.DH.Panda()
    #     panda.q = panda.qr

    #     q = panda.qr
    #     qt = np.c_[q, q].T

    #     q1, s1, _ = panda.qmincon(q)
    #     q2, _, _ = panda.qmincon(qt)

    #     qres = [-0.0969, -0.3000, 0.0870, -2.2000, 0.0297, 2.0000, 0.7620]

    #     nt.assert_array_almost_equal(q1, qres, decimal=4)
    #     nt.assert_array_almost_equal(q2[0, :], qres, decimal=4)
    #     nt.assert_array_almost_equal(q2[1, :], qres, decimal=4)

    def test_teach(self):
        panda = rp.models.DH.Panda()
        e = panda.teach(panda.q, block=False)
        e.close()

    def test_teach_withq(self):
        panda = rp.models.DH.Panda()
        e = panda.teach(q=panda.qr, block=False)
        e.close()

    def test_plot(self):
        panda = rp.models.DH.Panda()
        e = panda.plot(panda.qr, block=False, backend="pyplot")
        e.close()

    def test_teach_basic(self):
        l0 = rp.DHLink(d=2)
        r0 = rp.DHRobot([l0, l0.copy()])
        e = r0.teach(r0.q, block=False)
        e.step()
        e.close()

    def test_plot_traj(self):
        panda = rp.models.DH.Panda()
        q = np.random.rand(3, 7)
        e = panda.plot(q=q, block=False, dt=0.05, backend="pyplot")
        e.close()

    def test_control_type(self):
        panda = rp.models.DH.Panda()

        panda.control_mode = "p"

        with self.assertRaises(ValueError):
            panda.control_mode = "z"

    def test_plot_vellipse(self):
        panda = rp.models.DH.Panda()

        e = panda.plot_vellipse(panda.q, block=False, limits=[1, 2, 1, 2, 1, 2])
        e.close()

        e = panda.plot_vellipse(panda.q, block=False, centre="ee", opt="rot")
        e.step()
        e.close()

    def test_plot_fellipse(self):
        panda = rp.models.DH.Panda()

        e = panda.plot_fellipse(q=panda.qr, block=False, limits=[1, 2, 1, 2, 1, 2])
        e.close()

        e = panda.plot_fellipse(panda.qr, block=False, centre="ee", opt="rot")
        e.step()
        e.close()

    def test_plot_with_vellipse(self):
        panda = rp.models.DH.Panda()
        e = panda.plot(panda.qr, block=False, vellipse=True, backend="pyplot")
        e.close()

    def test_plot_with_fellipse(self):
        panda = rp.models.DH.Panda()
        e = panda.plot(panda.qr, block=False, fellipse=True, backend="pyplot")
        e.close()

    def test_str(self):
        r0 = rp.models.DH.Puma560()
        r1 = rp.models.DH.Panda()
        str(r0)
        str(r1)

        l0 = rp.PrismaticDH(offset=1.0, qlim=[-1, 1])
        l1 = rp.RevoluteDH(flip=True, offset=1.0, qlim=[-1, 1])
        r2 = rp.DHRobot([l0, l1])
        str(r2)

        l0 = rp.PrismaticMDH(offset=1.0, qlim=[-1, 1])
        l1 = rp.RevoluteMDH(flip=True, offset=1.0, qlim=[-1, 1])
        r3 = rp.DHRobot([l0, l1])
        str(r3)

        l0 = rp.PrismaticDH(offset=1.0)
        l1 = rp.RevoluteDH(flip=True, offset=1.0)
        r4 = rp.DHRobot([l0, l1])
        str(r4)

        l0 = rp.PrismaticMDH(offset=1.0)
        l1 = rp.RevoluteMDH(flip=True, offset=1.0)
        r5 = rp.DHRobot([l0, l1], base=sm.SE3.Tx(0.1), tool=sm.SE3.Tx(0.1))
        str(r5)

    def test_alpha(self):
        r0 = rp.models.DH.Puma560()

        nt.assert_array_almost_equal(r0.alpha, np.r_[1, 0, -1, 1, -1, 0] * math.pi / 2)

    def test_ets(self):
        panda = rp.models.DH.Panda()
        panda.ets()

        panda.base = sm.SE3.Tx(0.1)
        panda.ets()

    def test_SerialLink(self):
        rp.SerialLink([rp.RevoluteDH()])


if __name__ == "__main__":
    unittest.main()
