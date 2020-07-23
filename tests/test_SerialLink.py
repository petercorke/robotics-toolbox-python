#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import ropy as rp
import spatialmath as sm
import unittest


class TestLink(unittest.TestCase):

    def test_seriallink(self):
        l0 = rp.Link()
        rp.SerialLink([l0])

    def test_isprismatic(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic()
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])

        ans = [True, False, True, False]

        self.assertEqual(r0.isprismatic(), ans)

    def test_isrevolute(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic()
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])

        ans = [False, True, False, True]

        self.assertEqual(r0.isrevolute(), ans)

    def test_todegrees(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic()
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        q = np.array([np.pi, np.pi, np.pi, np.pi/2.0])

        ans = np.array([np.pi, 180, np.pi, 90])

        nt.assert_array_almost_equal(r0.todegrees(q), ans)
        nt.assert_array_almost_equal(r0.todegrees(), np.zeros(4))

    def test_toradians(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic()
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        q = np.array([np.pi, 180, np.pi, 90])
        r0.q = q

        ans = np.array([np.pi, np.pi, np.pi, np.pi/2.0])

        nt.assert_array_almost_equal(r0.toradians(q), ans)
        nt.assert_array_almost_equal(r0.toradians(), ans)

    def test_d(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute(d=2.0)
        l2 = rp.Prismatic()
        l3 = rp.Revolute(d=4.0)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        ans = [0.0, 2.0, 0.0, 4.0]

        self.assertEqual(r0.d, ans)

    def test_a(self):
        l0 = rp.Prismatic(a=1.0)
        l1 = rp.Revolute(a=2.0)
        l2 = rp.Prismatic(a=3.0)
        l3 = rp.Revolute(a=4.0)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        ans = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(r0.a, ans)

    def test_theta(self):
        l0 = rp.Prismatic(theta=1.0)
        l1 = rp.Revolute()
        l2 = rp.Prismatic(theta=3.0)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        ans = [1.0, 0.0, 3.0, 0.0]

        self.assertEqual(r0.theta, ans)

    def test_r(self):
        r = np.array([[1], [2], [3]])
        l0 = rp.Prismatic(r=r)
        l1 = rp.Revolute(r=r)
        l2 = rp.Prismatic(r=r)
        l3 = rp.Revolute(r=r)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        r1 = rp.SerialLink([l0])
        ans = np.c_[r, r, r, r]

        nt.assert_array_almost_equal(r0.r, ans)
        nt.assert_array_almost_equal(r1.r, r)

    def test_offset(self):
        l0 = rp.Prismatic(offset=1.0)
        l1 = rp.Revolute(offset=2.0)
        l2 = rp.Prismatic(offset=3.0)
        l3 = rp.Revolute(offset=4.0)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        ans = [1.0, 2.0, 3.0, 4.0]

        self.assertEqual(r0.offset, ans)

    def test_qlim(self):
        qlim = [-1, 1]
        l0 = rp.Prismatic(qlim=qlim)
        l1 = rp.Revolute(qlim=qlim)
        l2 = rp.Prismatic(qlim=qlim)
        l3 = rp.Revolute(qlim=qlim)

        r0 = rp.SerialLink([l0, l1, l2, l3])
        r1 = rp.SerialLink([l0])
        ans = np.c_[qlim, qlim, qlim, qlim]

        nt.assert_array_almost_equal(r0.qlim, ans)
        nt.assert_array_almost_equal(r1.qlim, qlim)

    def test_fkine(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic(theta=2.0)
        l3 = rp.Revolute()

        q = np.array([1, 2, 3, 4])

        T1 = np.array([
            [-0.14550003, -0.98935825, 0, 0],
            [0.98935825, -0.14550003, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ])

        r0 = rp.SerialLink([l0, l1, l2, l3])
        r0.q = q

        nt.assert_array_almost_equal(r0.fkine(q).A, T1)
        nt.assert_array_almost_equal(r0.fkine().A, T1)

    def test_fkine_traj(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic(theta=2.0)
        l3 = rp.Revolute()

        q = np.array([1, 2, 3, 4])
        qq = np.c_[q, q, q, q]

        T1 = np.array([
            [-0.14550003, -0.98935825, 0, 0],
            [0.98935825, -0.14550003, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ])

        r0 = rp.SerialLink([l0, l1, l2, l3])

        TT = r0.fkine(qq)

        nt.assert_array_almost_equal(TT[0].A, T1)
        nt.assert_array_almost_equal(TT[1].A, T1)
        nt.assert_array_almost_equal(TT[2].A, T1)
        nt.assert_array_almost_equal(TT[3].A, T1)

    def test_links(self):
        l0 = rp.Prismatic()
        with self.assertRaises(TypeError):
            rp.SerialLink(l0)

    def test_multiple(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic(theta=2.0)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1])
        r1 = rp.SerialLink([l2, l3])
        r3 = rp.SerialLink([r0, r1])
        r4 = rp.SerialLink([r0, l2, l3])

        q = np.array([1, 2, 3, 4])

        T1 = np.array([
            [-0.14550003, -0.98935825, 0, 0],
            [0.98935825, -0.14550003, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ])

        nt.assert_array_almost_equal(r3.fkine(q).A, T1)
        nt.assert_array_almost_equal(r4.fkine(q).A, T1)

    def test_bad_list(self):
        l0 = rp.Prismatic()

        with self.assertRaises(TypeError):
            rp.SerialLink([l0, 1])

    def test_add_seriallink(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic(theta=2.0)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1])
        r1 = rp.SerialLink([l2, l3])
        r3 = r0 + r1

        q = np.array([1, 2, 3, 4])

        T1 = np.array([
            [-0.14550003, -0.98935825, 0, 0],
            [0.98935825, -0.14550003, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ])

        nt.assert_array_almost_equal(r3.fkine(q).A, T1)

    def test_add_links(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        l2 = rp.Prismatic(theta=2.0)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1])
        r3 = r0 + l2 + l3

        q = np.array([1, 2, 3, 4])

        T1 = np.array([
            [-0.14550003, -0.98935825, 0, 0],
            [0.98935825, -0.14550003, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ])

        nt.assert_array_almost_equal(r3.fkine(q).A, T1)

    def test_add_error(self):
        l0 = rp.Prismatic()
        l1 = rp.Revolute()
        r0 = rp.SerialLink([l0, l1])

        with self.assertRaises(TypeError):
            r0 + 2

    def test_dh_error(self):
        l0 = rp.Prismatic(mdh=1)
        l1 = rp.Revolute()
        r0 = rp.SerialLink([l0])
        r1 = rp.SerialLink([l1])

        with self.assertRaises(ValueError):
            rp.SerialLink([l0, l1])

        with self.assertRaises(ValueError):
            r0 + r1

        with self.assertRaises(ValueError):
            rp.SerialLink([l0, l1])
            r0 + l1

    def test_name(self):
        panda = rp.PandaMDH()

        panda.name = 'new'
        self.assertEqual(panda.name, 'new')

    def test_base(self):
        panda = rp.PandaMDH()

        panda.base = sm.SE3.Rx(2)
        nt.assert_array_almost_equal(panda.base.A, sm.SE3.Rx(2).A)

        panda.base = sm.SE3.Ty(2).A
        nt.assert_array_almost_equal(panda.base.A, sm.SE3.Ty(2).A)

    def test_A(self):
        panda = rp.PandaMDH()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        T1 = np.array([
            [0.5403, -0.8415, 0, 0],
            [0.8415, 0.5403, 0, 0],
            [0, 0, 1, 0.333],
            [0, 0, 0, 1]
        ])

        T2 = np.array([
            [-0.3279, -0.9015, -0.2826,  0.2918],
            [0.9232, -0.3693,  0.1068, 0.06026],
            [-0.2006, -0.2258,  0.9533,  0.3314],
            [0, 0, 0, 1]
        ])

        nt.assert_array_almost_equal(panda.A(0, q).A, T1, decimal=4)
        nt.assert_array_almost_equal(panda.A([1, 4], q).A, T2, decimal=4)
        nt.assert_array_almost_equal(panda.A([1, 4]).A, T2, decimal=4)

    def test_A_error(self):
        panda = rp.PandaMDH()
        q = [1, 2, 3, 4, 5, 6, 7]

        with self.assertRaises(ValueError):
            panda.A(7, q).A

    def test_islimit(self):
        panda = rp.PandaMDH()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        ans = [False, True, True, True, True, True, True]

        self.assertEqual(panda.islimit(q), ans)
        self.assertEqual(panda.islimit(), ans)

    def test_isspherical(self):
        l0 = rp.Revolute()
        l1 = rp.Revolute(alpha=-np.pi/2)
        l2 = rp.Revolute(alpha=np.pi/2)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        r1 = rp.SerialLink([l0, l1])
        r2 = rp.SerialLink([l1, l2, l3, l0])

        self.assertTrue(r0.isspherical())
        self.assertFalse(r1.isspherical())
        self.assertFalse(r2.isspherical())

    def test_payload(self):
        panda = rp.PandaMDH()
        nt.assert_array_almost_equal(panda.r[:, 6], np.zeros(3))
        nt.assert_array_almost_equal(panda.links[6].m, 0)

        m = 6
        p = [1, 2, 3]
        panda.payload(m, p)

        nt.assert_array_almost_equal(panda.r[:, 6], p)
        nt.assert_array_almost_equal(panda.links[6].m, m)

    def test_jointdynamics(self):
        # TODO
        panda = rp.PandaMDH()
        panda.jointdynamics(1, 2)
        pass

    def test_twists(self):
        # TODO
        panda = rp.PandaMDH()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        panda.twists()
        panda.twists(q)
        pass

    def test_fkine_panda(self):
        panda = rp.PandaMDH()
        q = [1, 2, 3, 4, 5, 6, 7]

        T = np.array([
            [-0.8583, 0.1178, 0.4994, 0.1372],
            [0.1980, 0.9739, 0.1106, 0.3246],
            [-0.4734, 0.1938, -0.8593, 0.4436],
            [0, 0, 0, 1]
        ])

        nt.assert_array_almost_equal(panda.fkine(q).A, T, decimal=4)

    def test_jacobe(self):
        l0 = rp.Prismatic(theta=4)
        l1 = rp.Revolute(a=2)
        l2 = rp.Prismatic(theta=2)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        q = [1, 2, 3, 4]
        r0.q = q

        Je = np.array([
            [0, -0.5588, 0, 0],
            [0, 1.9203, 0, 0],
            [1.0000, 0, 1.0000, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1.0000, 0, 1.0000]
        ])

        nt.assert_array_almost_equal(r0.jacobe(q), Je, decimal=4)
        nt.assert_array_almost_equal(r0.jacobe(), Je, decimal=4)

    def test_jacob0(self):
        l0 = rp.Prismatic(theta=4)
        l1 = rp.Revolute(a=2)
        l2 = rp.Prismatic(theta=2)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        q = [1, 2, 3, 4]
        r0.q = q

        J0 = np.array([
            [0, 0.5588, 0, 0],
            [0, 1.9203, 0, 0],
            [1.0000, 0, 1.0000, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1.0000, 0, 1.0000]
        ])

        nt.assert_array_almost_equal(r0.jacob0(q), J0, decimal=4)
        nt.assert_array_almost_equal(r0.jacob0(), J0, decimal=4)

    def test_jacobe_panda(self):
        panda = rp.PandaMDH()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        Je = np.array([
            [0.3058, 0.1315, -0.2364, -0.0323, 0.0018, 0.2095, 0],
            [0.0954, 0.0303, -0.0721, 0.1494, -0.0258, 0.0144, 0],
            [-0.1469, 0.3385, 0.0506, 0.0847, -0.0000, -0.0880, 0],
            [-0.4734, 0.8292, -0.0732, 0.8991, -0.2788, -0.0685, 0],
            [0.1938, 0.4271, 0.7224, 0.3461, -0.0191, 0.9976, 0],
            [-0.8593, -0.3605, 0.6876, -0.2679, -0.9602, 0.0000, 1.0000]
        ])

        nt.assert_array_almost_equal(panda.jacobe(q), Je, decimal=4)
        nt.assert_array_almost_equal(panda.jacobe(), Je, decimal=4)

    def test_jacob0v(self):
        l0 = rp.Prismatic(theta=4)
        l1 = rp.Revolute(a=2)
        l2 = rp.Prismatic(theta=2)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        q = [1, 2, 3, 4]
        r0.q = q

        J = np.array([
            [0.8439, 0.5366, 0, 0, 0, 0],
            [-0.5366, 0.8439, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0.8439, 0.5366, 0],
            [0, 0, 0, -0.5366, 0.8439, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        nt.assert_array_almost_equal(r0.jacob0v(q), J, decimal=4)
        nt.assert_array_almost_equal(r0.jacob0v(), J, decimal=4)

    def test_jacobev(self):
        l0 = rp.Prismatic(theta=4)
        l1 = rp.Revolute(a=2)
        l2 = rp.Prismatic(theta=2)
        l3 = rp.Revolute()

        r0 = rp.SerialLink([l0, l1, l2, l3])
        q = [1, 2, 3, 4]
        r0.q = q

        J = np.array([
            [0.8439, -0.5366, 0, 0, 0, 0],
            [0.5366, 0.8439, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0.8439, -0.5366, 0],
            [0, 0, 0, 0.5366, 0.8439, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        nt.assert_array_almost_equal(r0.jacobev(q), J, decimal=4)
        nt.assert_array_almost_equal(r0.jacobev(), J, decimal=4)

    def test_nofriction(self):
        l0 = rp.Link(Tc=2, B=3)
        l1 = rp.Link(Tc=2, B=3)
        l2 = rp.Link(Tc=2, B=3)
        l3 = rp.Link(Tc=2, B=3)
        L = [l0, l1, l2, l3]

        r0 = rp.SerialLink(L)

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

    def test_pay(self):
        panda = rp.PandaMDH()
        panda.q = [1, 2, 3, 4, 5, 6, 7]

        w = [1, 2, 3, 4, 5, 6]

        wT = np.c_[w, w, w, w]
        qT = np.c_[panda.q, panda.q, panda.q, panda.q]

        tau = np.array(
            [6.0241, -4.4972, -7.2160, -4.2400,  7.0215, -4.6884, -6.0000])

        tau0 = np.array(
            [-5.9498, 1.4604, -3.4544, 1.5026, -3.7777, -6.6578, 2.6047])

        tauT = np.c_[tau, tau, tau, tau]
        tau0T = np.c_[tau0, tau0, tau0, tau0]

        Je = panda.jacobe()
        J0 = panda.jacob0()

        JeT = np.zeros((6, 7, 4))
        for i in range(4):
            JeT[:, :, i] = Je

        panda.pay(w)

        nt.assert_array_almost_equal(panda.pay(w), tau, decimal=4)
        nt.assert_array_almost_equal(panda.pay(w, frame=0), tau0, decimal=4)

        nt.assert_array_almost_equal(panda.pay(w, q=panda.q), tau, decimal=4)
        nt.assert_array_almost_equal(panda.pay(wT, q=qT), tauT, decimal=4)
        nt.assert_array_almost_equal(
            panda.pay(wT, q=qT, frame=0), tau0T, decimal=4)

        nt.assert_array_almost_equal(panda.pay(w, J=Je), tau, decimal=4)
        nt.assert_array_almost_equal(panda.pay(w, J=J0), tau0, decimal=4)

        nt.assert_array_almost_equal(panda.pay(wT, J=JeT), tauT, decimal=4)

        with self.assertRaises(ValueError):
            panda.pay(wT, panda.q)

        with self.assertRaises(TypeError):
            panda.pay(wT)

    def test_friction(self):
        l0 = rp.Revolute(d=2, B=3, G=2, Tc=[2, -1])
        qd = [1, 2, 3, 4]

        r0 = rp.SerialLink([l0, l0, l0, l0])

        tau = np.array([-16, -28, -40, -52])

        nt.assert_array_almost_equal(r0.friction(qd), tau)

    def test_allfkine(self):
        panda = rp.PandaMDH()
        q = [1, 2, 3, 4, 5, 6, 7]
        panda.q = q

        t0 = np.array([
            [0.5403, -0.8415, 0, 0],
            [0.8415, 0.5403, 0, 0],
            [0, 0, 1, 0.333],
            [0, 0, 0, 1]
        ])
        t1 = np.array([
            [-0.2248, -0.4913, -0.8415, 0],
            [-0.3502, -0.7651, 0.5403, 0],
            [-0.9093, 0.4161, 0, 0.333],
            [0, 0, 0, 1]
        ])
        t2 = np.array([
            [0.1038, 0.8648, 0.4913, 0.1552],
            [0.4229, -0.4855, 0.7651, 0.2418],
            [0.9002, 0.1283, -0.4161, 0.2015],
            [0, 0, 0, 1]
        ])
        t3 = np.array([
            [-0.4397, -0.2425, -0.8648, 0.1638],
            [-0.8555, -0.1801, 0.4855, 0.2767],
            [-0.2735, 0.9533, -0.1283, 0.2758],
            [0, 0, 0, 1]
        ])
        t4 = np.array([
            [-0.9540, -0.1763, -0.2425, 0.107],
            [0.2229, -0.9581, -0.1801, 0.2781],
            [-0.2006, -0.2258, 0.9533, 0.6644],
            [0, 0, 0, 1]
        ])
        t5 = np.array([
            [-0.8482, -0.4994, 0.1763, 0.107],
            [0.2643, -0.1106, 0.9581, 0.2781],
            [-0.4590, 0.8593, 0.2258, 0.6644],
            [0, 0, 0, 1]
        ])
        t6 = np.array([
            [-0.5236,  0.6902, 0.4994, 0.08575],
            [0.8287, 0.5487, 0.1106, 0.3132],
            [-0.1977, 0.4718, -0.8593, 0.5321],
            [0, 0, 0, 1]
        ])

        Tall = panda.allfkine(q)
        Tall2 = panda.allfkine()

        nt.assert_array_almost_equal(Tall[0].A, t0, decimal=4)
        nt.assert_array_almost_equal(Tall[1].A, t1, decimal=4)
        nt.assert_array_almost_equal(Tall[2].A, t2, decimal=4)
        nt.assert_array_almost_equal(Tall[3].A, t3, decimal=4)
        nt.assert_array_almost_equal(Tall[4].A, t4, decimal=4)
        nt.assert_array_almost_equal(Tall[5].A, t5, decimal=4)
        nt.assert_array_almost_equal(Tall[6].A, t6, decimal=4)
        nt.assert_array_almost_equal(Tall2[0].A, t0, decimal=4)

    def test_gravjac(self):
        l0 = rp.Revolute(d=2, B=3, G=2, Tc=[2, -1], alpha=0.4, a=0.2,
                         r=[0.1, 0.2, 0.05], m=0.5)
        l1 = rp.Prismatic(theta=0.1, B=3, G=2, Tc=[2, -1], a=0.2,
                          r=[0.1, 0.2, 0.05], m=0.5)

        r0 = rp.SerialLink([l0, l0, l0, l0])
        r1 = rp.SerialLink([l0, l0, l0, l1])
        q = [0.3, 0.4, 0.2, 0.1]
        qT = np.c_[q, q]
        r0.q = q

        grav = [0.3, 0.5, 0.7]

        tauB = [0, 4.6280, 3.1524, 0.9324]
        tauB2 = [1.9412, 1.1374, 0.3494, -0.0001]
        tauB3 = [0, 3.2819, 2.0195, 1.9693]

        res0 = r0.gravjac(qT)
        res1 = r0.gravjac(q)
        res2 = r0.gravjac(q, grav)
        res3 = r0.gravjac()
        res4 = r1.gravjac(q)

        nt.assert_array_almost_equal(res0[:, 0], tauB, decimal=4)
        nt.assert_array_almost_equal(res0[:, 1], tauB, decimal=4)
        nt.assert_array_almost_equal(res1[:, 0], tauB, decimal=4)
        nt.assert_array_almost_equal(res2[:, 0], tauB2, decimal=4)
        nt.assert_array_almost_equal(res3[:, 0], tauB, decimal=4)
        nt.assert_array_almost_equal(res4[:, 0], tauB3, decimal=4)

    def test_ikcon(self):
        panda = rp.PandaMDH()
        q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4])
        T = panda.fkine(q)
        Tt = sm.SE3([T, T, T])

        qr = [7.69161412e-04, 9.01409257e-01, -1.46372859e-02,
              -6.98000000e-02, 1.38978915e-02, 9.62104811e-01,
              7.84926515e-01]

        qa, err, success = panda.ikcon(T)
        qa2, err, success = panda.ikcon(Tt)

        nt.assert_array_almost_equal(qa, qr, decimal=4)
        nt.assert_array_almost_equal(qa2[:, 0], qr, decimal=4)
        nt.assert_array_almost_equal(qa2[:, 1], qr, decimal=4)

    def test_ikine(self):
        panda = rp.PandaMDH()
        q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4])
        T = panda.fkine(q)
        Tt = sm.SE3([T, T])

        l0 = rp.Revolute(d=2.0)
        l1 = rp.Prismatic(theta=1.0)
        l2 = rp.Prismatic(theta=1, qlim=[0, 2])
        r0 = rp.SerialLink([l0, l1])
        r1 = rp.SerialLink([l0, l2])

        qr = [0.0342, 1.6482, 0.0312, 1.2658, -0.0734, 0.4836, 0.7489]

        qa, success, err = panda.ikine(T)
        qa2, success, err = panda.ikine(Tt)
        qa3, success, err = panda.ikine(Tt, q0=np.zeros((7, 2)))
        qa4, success, err = panda.ikine(T, q0=np.zeros(7))

        # Untested
        qa5, success, err = r0.ikine(
            T.A, mask=[1, 1, 0, 0, 0, 0],
            transpose=5, ilimit=5)
        qa5, success, err = r0.ikine(T, mask=[1, 1, 0, 0, 0, 0])
        qa6, success, err = r0.ikine(T, mask=[1, 1, 0, 0, 0, 0], ilimit=1)
        qa7, success, err = r1.ikine(
            T, mask=[1, 1, 0, 0, 0, 0],
            ilimit=1, search=True, slimit=1)

        nt.assert_array_almost_equal(qa, qr, decimal=4)
        nt.assert_array_almost_equal(qa2[:, 0], qr, decimal=4)
        nt.assert_array_almost_equal(qa2[:, 1], qr, decimal=4)
        nt.assert_array_almost_equal(qa3[:, 1], qr, decimal=4)
        nt.assert_array_almost_equal(qa4, qr, decimal=4)

        with self.assertRaises(ValueError):
            panda.ikine(Tt, q0=np.zeros(7))

        with self.assertRaises(ValueError):
            r0.ikine(T)

        with self.assertRaises(ValueError):
            r0.ikine(
                T, mask=[1, 1, 0, 0, 0, 0], ilimit=1,
                search=True, slimit=1)
