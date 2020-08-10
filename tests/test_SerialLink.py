#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import ropy as rp
import spatialmath as sm
import unittest


class TestSerialLink(unittest.TestCase):

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
        q = np.array([np.pi, np.pi, np.pi, np.pi / 2.0])

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

        ans = np.array([np.pi, np.pi, np.pi, np.pi / 2.0])

        nt.assert_array_almost_equal(r0.toradians(q), ans)

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
        r1 = rp.SerialLink([l1, l2, l3])
        r3 = r0 + l2 + l3
        r4 = l0 + r1

        q = np.array([1, 2, 3, 4])

        T1 = np.array([
            [-0.14550003, -0.98935825, 0, 0],
            [0.98935825, -0.14550003, 0, 0],
            [0, 0, 1, 4],
            [0, 0, 0, 1]
        ])

        nt.assert_array_almost_equal(r3.fkine(q).A, T1)
        nt.assert_array_almost_equal(r4.fkine(q).A, T1)

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
            [-0.3279, -0.9015, -0.2826, 0.2918],
            [0.9232, -0.3693, 0.1068, 0.06026],
            [-0.2006, -0.2258, 0.9533, 0.3314],
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
        l1 = rp.Revolute(alpha=-np.pi / 2)
        l2 = rp.Revolute(alpha=np.pi / 2)
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
            [6.0241, -4.4972, -7.2160, -4.2400, 7.0215, -4.6884, -6.0000])

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
            [-0.5236, 0.6902, 0.4994, 0.08575],
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
        nt.assert_array_almost_equal(res1, tauB, decimal=4)
        nt.assert_array_almost_equal(res2, tauB2, decimal=4)
        nt.assert_array_almost_equal(res3, tauB, decimal=4)
        nt.assert_array_almost_equal(res4, tauB3, decimal=4)

    def test_ikcon(self):
        panda = rp.PandaMDH()
        q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        T = panda.fkine(q)
        Tt = sm.SE3([T, T, T])

        qr = [7.69161412e-04, 9.01409257e-01, -1.46372859e-02,
              -6.98000000e-02, 1.38978915e-02, 9.62104811e-01,
              7.84926515e-01]

        qa, success, err = panda.ikcon(T.A, q0=np.zeros(7))
        qa2, success, err = panda.ikcon(Tt)
        qa3, _, _ = panda.ikcon(Tt, q0=np.zeros((7, 3)))

        nt.assert_array_almost_equal(qa, qr, decimal=4)
        nt.assert_array_almost_equal(qa2[:, 0], qr, decimal=4)
        nt.assert_array_almost_equal(qa2[:, 1], qr, decimal=4)
        nt.assert_array_almost_equal(qa3[:, 0], qr, decimal=4)
        nt.assert_array_almost_equal(qa3[:, 1], qr, decimal=4)

    def test_ikine(self):
        panda = rp.PandaMDH()
        q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
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

    def test_ikine3(self):
        l0 = rp.Revolute(alpha=np.pi / 2)
        l1 = rp.Revolute(a=0.4318)
        l2 = rp.Revolute(d=0.15005, a=0.0203, alpha=-np.pi / 2)
        l3 = rp.Prismatic()
        l4 = rp.Prismatic(mdh=1)
        r0 = rp.SerialLink([l0, l1, l2])
        r1 = rp.SerialLink([l3, l3])
        r2 = rp.SerialLink([l3, l3, l3])
        r3 = rp.SerialLink([l4, l4, l4])

        q = [1, 1, 1]
        r0.q = q
        T = r0.fkine(q)
        # T2 = r1.fkine(q)
        Tt = sm.SE3([T, T])

        res = [2.9647, 1.7561, 0.2344]
        res2 = [1.0000, 0.6916, 0.2344]
        res3 = [2.9647, 2.4500, 3.1762]
        res4 = [1.0000, 1.3855, 3.1762]

        q0 = r0.ikine3(T.A)
        q1 = r0.ikine3(Tt)
        q2 = r0.ikine3(T, left=False, elbow_up=False)
        q3 = r0.ikine3(T, elbow_up=False)
        q4 = r0.ikine3(T, left=False)

        nt.assert_array_almost_equal(q0, res, decimal=4)
        nt.assert_array_almost_equal(q1[:, 0], res, decimal=4)
        nt.assert_array_almost_equal(q1[:, 1], res, decimal=4)
        nt.assert_array_almost_equal(q2, res2, decimal=4)
        nt.assert_array_almost_equal(q3, res3, decimal=4)
        nt.assert_array_almost_equal(q4, res4, decimal=4)

        with self.assertRaises(ValueError):
            r1.ikine3(T)

        with self.assertRaises(ValueError):
            r2.ikine3(T)

        with self.assertRaises(ValueError):
            r3.ikine3(T)

    def test_ikine6s_puma(self):
        r0 = rp.Puma560()
        q = r0.qr
        T = r0.fkine(q)

        qr0 = [0.2689, 1.5708, -1.4768, -3.1416, 0.0940, 2.8726]
        qr1 = [0.0000, 1.5238, -1.4768, -0.0000, -0.0470, -0.0000]

        q0, _ = r0.ikine6s(T)
        q1, _ = r0.ikine6s(T, left=False, elbow_up=False, wrist_flip=True)

        nt.assert_array_almost_equal(q0, qr0, decimal=4)
        nt.assert_array_almost_equal(q1, qr1, decimal=4)

    def test_ikine6s_rrp(self):
        l0 = rp.Revolute(alpha=-np.pi / 2)
        l1 = rp.Revolute(alpha=np.pi / 2)
        l2 = rp.Prismatic()
        l3 = rp.Revolute(alpha=-np.pi / 2)
        l4 = rp.Revolute(alpha=np.pi / 2)
        l5 = rp.Revolute()
        r0 = rp.SerialLink([l0, l1, l2, l3, l4, l5])
        r1 = rp.SerialLink([l1, l0, l2, l3, l4, l5])
        q = [1, 1, 1, 1, 1, 1]
        T1 = r0.fkine(q)
        T2 = r1.fkine(q)

        qr0 = [1.0000, -2.1416, -1.0000, -1.0000, -2.1416, 1.0000]
        qr1 = [-2.1416, -1.0000, 1.0000, -2.1416, 1.0000, 1.0000]
        qr2 = [1.0000, 1.0000, 1.0000, -2.1416, -1.0000, -2.1416]
        qr3 = [-2.1416, 2.1416, -1.0000, -1.0000, 2.1416, -2.1416]

        q0, _ = r0.ikine6s(T1)
        q1, _ = r0.ikine6s(T1, left=False, elbow_up=False, wrist_flip=True)
        q2, _ = r1.ikine6s(T2)
        q3, _ = r1.ikine6s(T2, left=False, elbow_up=False, wrist_flip=True)

        nt.assert_array_almost_equal(q0, qr0, decimal=4)
        nt.assert_array_almost_equal(q1, qr1, decimal=4)
        nt.assert_array_almost_equal(q2, qr2, decimal=4)
        nt.assert_array_almost_equal(q3, qr3, decimal=4)

    def test_ikine6s_simple(self):
        l0 = rp.Revolute(alpha=-np.pi / 2)
        l1 = rp.Revolute()
        l2 = rp.Revolute(alpha=np.pi / 2)
        l3 = rp.Revolute(alpha=-np.pi / 2)
        l4 = rp.Revolute(alpha=np.pi / 2)
        l5 = rp.Revolute()
        r0 = rp.SerialLink([l0, l1, l2, l3, l4, l5])
        r1 = rp.SerialLink([l2, l1, l0, l3, l4, l5])
        q = [1, 1, 1, 1, 1, 1]
        T1 = r0.fkine(q)
        T2 = r1.fkine(q)

        qr0 = [0, 0, 0, -0.9741, -2.2630, -0.4605]
        qr1 = [0, 0, 0, 0.1947, -1.3811, 1.8933]
        qr2 = [0, 0, 0, 2.1675, 2.2630, 2.6811]
        qr3 = [0, 0, 0, -2.9468, 1.3811, -1.2483]

        q0, _ = r0.ikine6s(T1)
        q1, _ = r0.ikine6s(T1, left=False, elbow_up=False, wrist_flip=True)
        q2, _ = r1.ikine6s(T2)
        q3, _ = r1.ikine6s(T2, left=False, elbow_up=False, wrist_flip=True)

        nt.assert_array_almost_equal(q0, qr0, decimal=4)
        nt.assert_array_almost_equal(q1, qr2, decimal=4)
        nt.assert_array_almost_equal(q2, qr1, decimal=4)
        nt.assert_array_almost_equal(q3, qr3, decimal=4)

    def test_ikine6s_offset(self):
        l0 = rp.Revolute(alpha=-np.pi / 2)
        l1 = rp.Revolute(d=1.0)
        l2 = rp.Revolute(alpha=np.pi / 2)
        l3 = rp.Revolute(alpha=-np.pi / 2)
        l4 = rp.Revolute(alpha=np.pi / 2)
        l5 = rp.Revolute()
        r0 = rp.SerialLink([l0, l1, l2, l3, l4, l5])
        r1 = rp.SerialLink([l2, l1, l0, l3, l4, l5])
        q = [1, 1, 1, 1, 1, 1]
        T1 = r0.fkine(q)
        T2 = r1.fkine(q)

        qr0 = [1.0000, 3.1416, -0.0000, -1.1675, -0.8786, 2.6811]
        qr1 = [1.0000, -1.1059, 2.6767, 0.8372, 1.2639, 1.3761]
        qr2 = [1.0000, 3.1416, -3.1416, -0.8053, -1.3811, 1.8933]
        qr3 = [1.0000, -1.1059, -0.4649, 1.8311, 2.3192, -2.6398]

        q0, _ = r0.ikine6s(T1.A)
        q1, _ = r0.ikine6s(T1, left=False, elbow_up=False, wrist_flip=True)
        q2, _ = r1.ikine6s(T2)
        q3, _ = r1.ikine6s(T2, left=False, elbow_up=False, wrist_flip=True)

        nt.assert_array_almost_equal(q0, qr0, decimal=4)
        nt.assert_array_almost_equal(q1, qr1, decimal=4)
        nt.assert_array_almost_equal(q2, qr2, decimal=4)
        nt.assert_array_almost_equal(q3, qr3, decimal=4)

    def test_ikine6s_traj(self):
        r0 = rp.Puma560()
        q = r0.qr
        T = r0.fkine(q)
        Tt = sm.SE3([T, T, T])

        qr0 = [0.2689, 1.5708, -1.4768, -3.1416, 0.0940, 2.8726]

        q0, _ = r0.ikine6s(Tt)

        nt.assert_array_almost_equal(q0[:, 0], qr0, decimal=4)
        nt.assert_array_almost_equal(q0[:, 1], qr0, decimal=4)
        nt.assert_array_almost_equal(q0[:, 2], qr0, decimal=4)

    def test_ikine6s_fail(self):
        l0 = rp.Revolute(alpha=np.pi / 2)
        l1 = rp.Revolute(d=1.0)
        l2 = rp.Revolute(alpha=np.pi / 2)
        l3 = rp.Revolute(alpha=-np.pi / 2)
        l4a = rp.Revolute(alpha=np.pi / 2)
        l4b = rp.Revolute()
        l5 = rp.Revolute()
        l6 = rp.Revolute(mdh=1)
        r0 = rp.SerialLink([l0, l1, l2, l3, l4a, l5])
        r1 = rp.SerialLink([l0, l1, l2, l3, l4b, l5])
        r2 = rp.SerialLink([l1, l2, l3])
        r3 = rp.SerialLink([l6, l6, l6, l6, l6, l6])

        puma = rp.Puma560()
        T = sm.SE3(0, 10, 10)
        puma.ikine6s(T)

        q = [1, 1, 1, 1, 1, 1]
        T = r0.fkine(q)

        with self.assertRaises(ValueError):
            r0.ikine6s(T)

        with self.assertRaises(ValueError):
            r1.ikine6s(T)

        with self.assertRaises(ValueError):
            r2.ikine6s(T)

        with self.assertRaises(ValueError):
            r3.ikine6s(T)

    def test_ikinem(self):
        puma = rp.Puma560()
        q = puma.qr
        T = puma.fkine(q)
        Tt = sm.SE3([T, T])

        q0, _, _ = puma.ikinem(Tt)
        q1, success, _ = puma.ikinem(T.A, qlimits=False)
        q2, success, _ = puma.ikinem(T, qlimits=False, stiffness=0.1, ilimit=1)

        print(np.sum(np.abs(T.A - puma.fkine(q0[:, 0]).A)))

        self.assertTrue(
            np.sum(np.abs(T.A - puma.fkine(q0[:, 0]).A)) < 0.7)
        self.assertTrue(
            np.sum(np.abs(T.A - puma.fkine(q0[:, 1]).A)) < 0.7)
        self.assertTrue(
            np.sum(np.abs(T.A - puma.fkine(q1).A)) < 0.7)

    def test_ikunc(self):
        puma = rp.Puma560()
        q = puma.qr
        T = puma.fkine(q)
        Tt = sm.SE3([T, T])

        q0, _, _ = puma.ikunc(Tt)
        q1, success, _ = puma.ikunc(T.A)
        q2, success, _ = puma.ikunc(T, ilimit=1)

        nt.assert_array_almost_equal(
            T.A - puma.fkine(q0[:, 0]).A, np.zeros((4, 4)), decimal=4)

        nt.assert_array_almost_equal(
            T.A - puma.fkine(q0[:, 1]).A, np.zeros((4, 4)), decimal=4)

        nt.assert_array_almost_equal(
            T.A - puma.fkine(q1).A, np.zeros((4, 4)), decimal=4)

    def test_rne(self):
        puma = rp.Puma560()
        puma.q = puma.qn

        z = np.zeros(6)
        o = np.ones(6)
        fext = [1, 2, 3, 1, 2, 3]

        tr0 = [-0.0000, 31.6399, 6.0351, 0.0000, 0.0283, 0]
        tr1 = [29.1421, 56.5044, 16.3528, 1.2645, 1.1239, 0.5196]
        tr2 = [32.4952, 60.8670, 17.7436, 1.4545, 1.2991, 0.7138]
        tr3 = [29.7849, 53.9511, 15.0208,  4.0929, -0.8761, 3.5196]

        t0 = puma.rne(z, z, puma.qn)
        t1 = puma.rne(z, o, puma.qn)

        puma.gravity = [0, 0, 9.81]
        t2 = puma.rne(o, o, puma.qn)
        t3 = puma.rne(z, z, grav=[0, 0, 9.81])
        t4 = puma.rne(z, o, q=puma.qn, fext=fext)

        nt.assert_array_almost_equal(t0, tr0, decimal=4)
        nt.assert_array_almost_equal(t1, tr1, decimal=4)
        nt.assert_array_almost_equal(t2, tr2, decimal=4)
        nt.assert_array_almost_equal(t3, tr0, decimal=4)
        nt.assert_array_almost_equal(t4, tr3, decimal=4)

    def test_rne_traj(self):
        puma = rp.Puma560()

        z = np.zeros(6)
        o = np.ones(6)

        tr0 = [-0.0000, 31.6399, 6.0351, 0.0000, 0.0283, 0]
        tr1 = [32.4952, 60.8670, 17.7436, 1.4545, 1.2991, 0.7138]

        t0 = puma.rne(np.c_[z, o], np.c_[z, o], np.c_[puma.qn, puma.qn])

        nt.assert_array_almost_equal(t0[:, 0], tr0, decimal=4)
        nt.assert_array_almost_equal(t0[:, 1], tr1, decimal=4)

    def test_rne_delete(self):
        puma = rp.Puma560()

        z = np.zeros(6)

        tr0 = [-0.0000, 31.6399, 6.0351, 0.0000, 0.0283, 0]

        t0 = puma.rne(z, z, puma.qn)
        puma.delete_rne()
        t1 = puma.rne(z, z, puma.qn)

        nt.assert_array_almost_equal(t0, tr0, decimal=4)
        nt.assert_array_almost_equal(t1, tr0, decimal=4)

    def test_accel(self):
        puma = rp.Puma560()
        puma.q = puma.qn
        q = puma.qn

        qd = [0.1, 0.2, 0.8, 0.2, 0.5, 1.0]
        torque = [1.0, 3.2, 1.8, 0.1, 0.7, 4.6]

        res = [-7.4102, -9.8432, -10.9694, -4.4314, -0.9881, 21.0228]

        qdd0 = puma.accel(qd, torque, q)
        qdd1 = puma.accel(np.c_[qd, qd], np.c_[torque, torque], np.c_[q, q])
        qdd2 = puma.accel(qd, torque)

        nt.assert_array_almost_equal(qdd0, res, decimal=4)
        nt.assert_array_almost_equal(qdd1[:, 0], res, decimal=4)
        nt.assert_array_almost_equal(qdd1[:, 1], res, decimal=4)
        nt.assert_array_almost_equal(qdd2, res, decimal=4)

    def test_inertia(self):
        puma = rp.Puma560()
        puma.q = puma.qn
        q = puma.qn

        Ir = [
            [3.6594, -0.4044, 0.1006, -0.0025, 0.0000, -0.0000],
            [-0.4044, 4.4137, 0.3509, 0.0000, 0.0024, 0.0000],
            [0.1006, 0.3509, 0.9378, 0.0000, 0.0015, 0.0000],
            [-0.0025, 0.0000, 0.0000, 0.1925, 0.0000, 0.0000],
            [0.0000, 0.0024, 0.0015, 0.0000, 0.1713, 0.0000],
            [-0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1941]]

        I0 = puma.inertia(q)
        I1 = puma.inertia(np.c_[q, q])
        I2 = puma.inertia()

        nt.assert_array_almost_equal(I0, Ir, decimal=4)
        nt.assert_array_almost_equal(I1[:, :, 0], Ir, decimal=4)
        nt.assert_array_almost_equal(I1[:, :, 1], Ir, decimal=4)
        nt.assert_array_almost_equal(I2, Ir, decimal=4)

    def test_cinertia(self):
        puma = rp.Puma560()
        puma.q = puma.qn
        q = puma.qn

        Mr = [
            [17.2954, -2.7542, -9.6233, -0.0000, 0.2795, 0.0000],
            [-2.7542, 12.1909, 1.2459, -0.3254, -0.0703, -0.9652],
            [-9.6233, 1.2459, 13.3348, -0.0000, 0.2767, -0.0000],
            [-0.0000, -0.3254, -0.0000, 0.1941, 0.0000, 0.1941],
            [0.2795, -0.0703, 0.2767, 0.0000, 0.1713, 0.0000],
            [0.0000, -0.9652, -0.0000, 0.1941, 0.0000, 0.5791]]

        M0 = puma.cinertia(q)
        M1 = puma.cinertia(np.c_[q, q])
        M2 = puma.cinertia()

        nt.assert_array_almost_equal(M0, Mr, decimal=4)
        nt.assert_array_almost_equal(M1[:, :, 0], Mr, decimal=4)
        nt.assert_array_almost_equal(M1[:, :, 1], Mr, decimal=4)
        nt.assert_array_almost_equal(M2, Mr, decimal=4)

    def test_coriolis(self):
        puma = rp.Puma560()
        puma.q = puma.qn
        q = puma.qn

        qd = [1, 2, 3, 1, 2, 3]

        Cr = [
            [-0.1735, -2.0494, -0.1178, -0.0002, -0.0045, 0.0001],
            [0.6274, 1.1572, 1.9287, -0.0015, -0.0003, -0.0000],
            [-0.3608, -0.7734, -0.0018, -0.0009, -0.0064, -0.0000],
            [0.0011, 0.0005, -0.0001, 0.0002, 0.0002, -0.0001],
            [-0.0002, 0.0028, 0.0046, -0.0002, -0.0000, -0.0000],
            [0.0001, 0.0000, 0.0000, 0.0000, 0.0000, 0]]

        C0 = puma.coriolis(qd, q)
        C1 = puma.coriolis(np.c_[qd, qd], np.c_[q, q])
        C2 = puma.coriolis(qd)

        nt.assert_array_almost_equal(C0, Cr, decimal=4)
        nt.assert_array_almost_equal(C1[:, :, 0], Cr, decimal=4)
        nt.assert_array_almost_equal(C1[:, :, 1], Cr, decimal=4)
        nt.assert_array_almost_equal(C2, Cr, decimal=4)

    def test_gravload(self):
        puma = rp.Puma560()
        puma.q = puma.qn
        q = puma.qn

        grav = [0, 0, 9.81]

        taur = [-0.0000, 31.6399, 6.0351, 0.0000, 0.0283, 0]

        tau0 = puma.gravload(q)
        tau1 = puma.gravload(np.c_[q, q])
        tau2 = puma.gravload(q=np.c_[q, q], grav=np.c_[grav, grav])
        tau3 = puma.gravload(grav=grav)

        nt.assert_array_almost_equal(tau0, taur, decimal=4)
        nt.assert_array_almost_equal(tau1[:, 0], taur, decimal=4)
        nt.assert_array_almost_equal(tau1[:, 1], taur, decimal=4)
        nt.assert_array_almost_equal(tau2[:, 0], taur, decimal=4)
        nt.assert_array_almost_equal(tau2[:, 1], taur, decimal=4)
        nt.assert_array_almost_equal(tau3, taur, decimal=4)

    def test_itorque(self):
        puma = rp.Puma560()
        puma.q = puma.qn
        q = puma.qn

        qdd = [1, 2, 3, 1, 2, 3]

        tauir = [3.1500, 9.4805, 3.6189, 0.1901, 0.3519, 0.5823]

        taui0 = puma.itorque(qdd, q)
        taui1 = puma.itorque(np.c_[qdd, qdd], np.c_[q, q])
        taui2 = puma.itorque(qdd)

        nt.assert_array_almost_equal(taui0, tauir, decimal=4)
        nt.assert_array_almost_equal(taui1[:, 0], tauir, decimal=4)
        nt.assert_array_almost_equal(taui1[:, 1], tauir, decimal=4)
        nt.assert_array_almost_equal(taui2, tauir, decimal=4)

    def test_str(self):
        puma = rp.Puma560()
        l0 = rp.Prismatic(mdh=1)
        r0 = rp.SerialLink([l0, l0, l0])
        str(r0)

        res = (
            "\nPuma 560 (Unimation): 6 axis, RRRRRR, std DH\n"
            "Parameters:\n"
            "Revolute   theta= 0.00  d= 0.00  a= 0.00"
            "  alpha= 1.57  offset= 0.00\n"
            "Revolute   theta= 0.00  d= 0.00  a= 0.43"
            "  alpha= 0.00  offset= 0.00\n"
            "Revolute   theta= 0.00  d= 0.15  a= 0.02"
            "  alpha=-1.57  offset= 0.00\n"
            "Revolute   theta= 0.00  d= 0.43  a= 0.00"
            "  alpha= 1.57  offset= 0.00\n"
            "Revolute   theta= 0.00  d= 0.00  a= 0.00"
            "  alpha=-1.57  offset= 0.00\n"
            "Revolute   theta= 0.00  d= 0.00  a= 0.00"
            "  alpha= 0.00  offset= 0.00\n\n"
            "tool:  t = (0, 0, 0),  RPY/xyz = (0, 0, 0) deg")

        self.assertEqual(str(puma), res)

    def test_paycap(self):
        puma = rp.Puma560()
        puma.q = puma.qn
        q = puma.qn

        w = [1, 2, 1, 2, 1, 2]
        tauR = np.ones((6, 2))
        tauR[:, 1] = -1

        res0 = [
            1.15865438e+00, -3.04790052e+02, -5.00870095e+01,  6.00479950e+15,
            3.76356072e+00, 1.93649167e+00]

        wmax0, joint = puma.paycap(w, tauR, q=q, frame=0)
        wmax1, _ = puma.paycap(np.c_[w, w], tauR, q=np.c_[q, q], frame=0)
        wmax2, _ = puma.paycap(w, tauR, frame=0)

        nt.assert_allclose(wmax0, res0)
        self.assertEqual(joint, 1)
        nt.assert_allclose(wmax1[:, 0], res0)
        nt.assert_allclose(wmax1[:, 1], res0)
        nt.assert_allclose(wmax2, res0)

    def test_jacob_dot(self):
        puma = rp.Puma560()
        puma.q = puma.qr
        puma.qd = puma.qr
        q = puma.qr

        j0 = puma.jacob_dot(q, q)
        j1 = puma.jacob_dot()

        res = [-0.0000, -1.0654, -0.3702,  2.4674, 0, 0]

        nt.assert_array_almost_equal(j0, res, decimal=4)
        nt.assert_array_almost_equal(j1, res, decimal=4)

    def test_yoshi(self):
        puma = rp.Puma560()
        puma.q = puma.qn
        q = puma.qn

        m0 = puma.maniplty()
        m1 = puma.maniplty(q)
        m2 = puma.maniplty(np.c_[q, q])
        m3 = puma.maniplty(q, axes=[1, 1, 1, 0, 0, 0])
        m4 = puma.maniplty(q, axes=[0, 0, 0, 1, 1, 1])

        a0 = 0.0786
        a2 = 0.111181
        a3 = 2.44949

        nt.assert_almost_equal(m0, a0, decimal=4)
        nt.assert_almost_equal(m1, a0, decimal=4)
        nt.assert_almost_equal(m2[0], a0, decimal=4)
        nt.assert_almost_equal(m2[1], a0, decimal=4)
        nt.assert_almost_equal(m3, a2, decimal=4)
        nt.assert_almost_equal(m4, a3, decimal=4)

    def test_asada(self):
        puma = rp.Puma560()
        puma.q = puma.qn
        q = puma.qn

        m0, mx0 = puma.maniplty(method='asada')
        m1, mx1 = puma.maniplty(q, method='asada')
        m2, mx2 = puma.maniplty(np.c_[q, q], method='asada')
        m3, mx3 = puma.maniplty(q, axes=[1, 1, 1, 0, 0, 0], method='asada')
        m4, mx4 = puma.maniplty(q, axes=[0, 0, 0, 1, 1, 1], method='asada')
        m5, mx5 = puma.maniplty(puma.qz, method='asada')

        a0 = 0.0044
        a2 = 0.2094
        a3 = 0.1716
        a4 = 0.0

        ax0 = np.array([
            [17.2954, -2.7542, -9.6233, -0.0000,  0.2795, -0.0000],
            [-2.7542, 12.1909,  1.2459, -0.3254, -0.0703, -0.9652],
            [-9.6233,  1.2459, 13.3348, -0.0000,  0.2767,  0.0000],
            [-0.0000, -0.3254, -0.0000,  0.1941,  0.0000,  0.1941],
            [0.2795, -0.0703,  0.2767,  0.0000,  0.1713,  0.0000],
            [-0.0000, -0.9652,  0.0000,  0.1941,  0.0000,  0.5791]
        ])

        ax1 = np.array([
            [17.2954, -2.7542, -9.6233],
            [-2.7542, 12.1909,  1.2459],
            [-9.6233,  1.2459, 13.3348]
        ])

        ax2 = np.array([
            [0.1941, 0.0000, 0.1941],
            [0.0000, 0.1713, 0.0000],
            [0.1941, 0.0000, 0.5791]
        ])

        ax3 = np.zeros((6, 6))

        nt.assert_almost_equal(m0, a0, decimal=4)
        nt.assert_array_almost_equal(mx0, ax0, decimal=4)
        nt.assert_almost_equal(m1, a0, decimal=4)
        nt.assert_array_almost_equal(mx1, ax0, decimal=4)
        nt.assert_almost_equal(m2[0], a0, decimal=4)
        nt.assert_array_almost_equal(mx2[:, :, 0], ax0, decimal=4)
        nt.assert_almost_equal(m2[1], a0, decimal=4)
        nt.assert_array_almost_equal(mx2[:, :, 1], ax0, decimal=4)

        nt.assert_almost_equal(m3, a2, decimal=4)
        nt.assert_array_almost_equal(mx3, ax1, decimal=4)

        nt.assert_almost_equal(m4, a3, decimal=4)
        nt.assert_array_almost_equal(mx4, ax2, decimal=4)

        nt.assert_almost_equal(m5, a4, decimal=4)
        nt.assert_array_almost_equal(mx5, ax3, decimal=4)

    def test_maniplty_fail(self):
        puma = rp.Puma560()
        puma.q = puma.qn

        with self.assertRaises(ValueError):
            puma.maniplty(method='notamethod')

    def test_perterb(self):
        puma = rp.Puma560()
        p2 = puma.perterb()
        p3 = puma.perterb(0.8)

        resI0 = np.zeros(puma.n)
        resm0 = np.zeros(puma.n)
        resI1 = np.zeros(puma.n)
        resm1 = np.zeros(puma.n)

        for i in range(puma.n):
            resI0[i] = np.divide(
                np.sum(np.abs(puma.links[i].I - p2.links[i].I)),
                np.sum(np.abs(puma.links[i].I)))

            if puma.links[i].m - p2.links[i].m != 0.0:
                resm0[i] = np.abs(np.divide(
                    (puma.links[i].m - p2.links[i].m),
                    puma.links[i].m))
            else:
                resm0[i] = 0

            resI1[i] = np.divide(
                np.sum(np.abs(puma.links[i].I - p3.links[i].I)),
                np.sum(np.abs(puma.links[i].I)))

            if puma.links[i].m - p3.links[i].m != 0.0:
                resm1[i] = np.abs(np.divide(
                    (puma.links[i].m - p3.links[i].m),
                    puma.links[i].m))
            else:
                resm1[i] = 0

            self.assertTrue(resI0[i] < 0.1)
            self.assertTrue(resm0[i] < 0.1 or np.isnan(resm0[i]))
            self.assertTrue(resI1[i] < 0.8)
            self.assertTrue(resm1[i] < 0.8 or np.isnan(resm1[i]))

    def test_qmincon(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr

        q = panda.qr
        qt = np.c_[q, q]

        q0, s0, _ = panda.qmincon()
        q1, s1, _ = panda.qmincon(q)
        q2, _, _ = panda.qmincon(qt)

        qres = [-0.0969, -0.3000, 0.0870, -2.2000, 0.0297, 2.0000, 0.7620]

        nt.assert_array_almost_equal(q0, qres, decimal=4)
        nt.assert_array_almost_equal(q1, qres, decimal=4)
        nt.assert_array_almost_equal(q2[:, 0], qres, decimal=4)
        nt.assert_array_almost_equal(q2[:, 1], qres, decimal=4)

    def test_teach(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr
        e = panda.teach(block=False)
        e.close()

        e2 = panda.teach(block=False, q=panda.qr)
        e2.close()

    def test_plot(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr
        e = panda.plot(block=False)
        e.close()

    def test_teach_basic(self):
        l0 = rp.Link(d=2)
        r0 = rp.SerialLink([l0, l0])
        e = r0.teach(False)
        e.step()
        e.close()

    def test_plot_traj(self):
        panda = rp.PandaMDH()
        q = np.random.rand(7, 3)
        e = panda.plot(block=False, q=q, dt=0)
        e.close()

    def test_control_type(self):
        panda = rp.PandaMDH()

        panda.control_type = 'p'

        with self.assertRaises(ValueError):
            panda.control_type = 'z'

    def test_plot_vellipse(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr

        e = panda.plot_vellipse(block=False, limits=[1, 2, 1, 2, 1, 2])
        e.close()

        e = panda.plot_vellipse(
            block=False, q=panda.qr, centre='ee', opt='rot')
        e.step(0)
        e.close()

        with self.assertRaises(TypeError):
            panda.plot_vellipse(vellipse=10)

        with self.assertRaises(ValueError):
            panda.plot_vellipse(centre='ff')

    def test_plot_fellipse(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr

        e = panda.plot_fellipse(block=False, limits=[1, 2, 1, 2, 1, 2])
        e.close()

        e = panda.plot_fellipse(
            block=False, q=panda.qr, centre='ee', opt='rot')
        e.step(0)
        e.close()

        with self.assertRaises(TypeError):
            panda.plot_fellipse(fellipse=10)

        with self.assertRaises(ValueError):
            panda.plot_fellipse(centre='ff')

    def test_plot_with_vellipse(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr
        e = panda.plot(block=False, vellipse=True)
        e.close()

    def test_plot_with_fellipse(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr
        e = panda.plot(block=False, fellipse=True)
        e.close()

    def test_plot2(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr
        e = panda.plot2(block=False, name=True)
        e.close()

    def test_plot2_traj(self):
        panda = rp.PandaMDH()
        q = np.random.rand(7, 3)
        e = panda.plot2(block=False, q=q, dt=0)
        e.close()

    def test_teach2_basic(self):
        l0 = rp.Link(d=2)
        r0 = rp.SerialLink([l0, l0])
        e = r0.teach2(False)
        e.step()
        e.close()

    def test_teach2(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr
        e = panda.teach(block=False)
        e.close()

        e2 = panda.teach2(block=False, q=panda.qr)
        e2.close()

    def test_plot_with_vellipse2(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr
        e = panda.plot2(block=False, vellipse=True, limits=[1, 2, 1, 2])
        e.step()
        e.close()

    def test_plot_with_fellipse2(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr
        e = panda.plot2(block=False, fellipse=True)
        e.close()

    def test_plot_with_vellipse2_fail(self):
        panda = rp.PandaMDH()
        panda.q = panda.qr
        e = rp.backend.PyPlot2()
        e.launch()
        e.add(panda.fellipse(
                q=panda.qr, centre=[0, 1]))

        with self.assertRaises(ValueError):
            e.add(panda.fellipse(
                q=panda.qr, centre='ee', opt='rot'))
