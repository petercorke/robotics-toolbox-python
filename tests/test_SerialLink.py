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
