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
        r.dyntable()

    def test_linkcolormap(self):
        r = rp.models.DH.Puma560()
        r.linkcolormap()

        r.linkcolormap(['r', 'r', 'r', 'r', 'r', 'r'])

    def test_base_error(self):
        r = rp.models.DH.Puma560()

        with self.assertRaises(ValueError):
            r.base = 2

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

    def test_ikine(self):
        panda = rp.models.DH.Panda()
        q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        T = panda.fkine(q)
        Tt = sm.SE3([T, T])

        l0 = rp.RevoluteDH(d=2.0)
        l1 = rp.PrismaticDH(theta=1.0)
        l2 = rp.PrismaticDH(theta=1, qlim=[0, 2])
        r0 = rp.DHRobot([l0, l1])
        r1 = rp.DHRobot([l0, l2])

        qr = [0.0342, 1.6482, 0.0312, 1.2658, -0.0734, 0.4836, 0.7489]

        qa, success, err = panda.ikine(T)
        qa2, success, err = panda.ikine(Tt)
        qa3, success, err = panda.ikine(Tt, q0=np.zeros((2, 7)))
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
        nt.assert_array_almost_equal(qa2[0, :], qr, decimal=4)
        nt.assert_array_almost_equal(qa2[1, :], qr, decimal=4)
        nt.assert_array_almost_equal(qa3[1, :], qr, decimal=4)
        nt.assert_array_almost_equal(qa4, qr, decimal=4)

        with self.assertRaises(ValueError):
            panda.ikine(Tt, q0=np.zeros(7))

        with self.assertRaises(ValueError):
            r0.ikine(T)

        with self.assertRaises(ValueError):
            r0.ikine(
                T, mask=[1, 1, 0, 0, 0, 0], ilimit=1,
                search=True, slimit=1)

    def test_ikcon(self):
        panda = rp.models.DH.Panda()
        q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        T = panda.fkine(q)
        Tt = sm.SE3([T, T, T])

        qr = [7.69161412e-04, 9.01409257e-01, -1.46372859e-02,
              -6.98000000e-02, 1.38978915e-02, 9.62104811e-01,
              7.84926515e-01]

        qa, success, err = panda.ikcon(T.A, q0=np.zeros(7))
        qa2, success, err = panda.ikcon(Tt)
        qa3, _, _ = panda.ikcon(Tt, q0=np.zeros((3, 7)))

        nt.assert_array_almost_equal(qa, qr, decimal=4)
        nt.assert_array_almost_equal(qa2[0, :], qr, decimal=4)
        nt.assert_array_almost_equal(qa2[1, :], qr, decimal=4)
        nt.assert_array_almost_equal(qa3[0, :], qr, decimal=4)
        nt.assert_array_almost_equal(qa3[1, :], qr, decimal=4)

    def test_ikunc(self):
        puma = rp.models.DH.Puma560()
        q = puma.qr
        T = puma.fkine(q)
        Tt = sm.SE3([T, T])

        q0, _, _ = puma.ikunc(Tt)
        q1, success, _ = puma.ikunc(T.A)
        q2, success, _ = puma.ikunc(T, ilimit=1)

        nt.assert_array_almost_equal(
            T.A - puma.fkine(q0[0, :]).A, np.zeros((4, 4)), decimal=4)

        nt.assert_array_almost_equal(
            T.A - puma.fkine(q0[1, :]).A, np.zeros((4, 4)), decimal=4)

        nt.assert_array_almost_equal(
            T.A - puma.fkine(q1).A, np.zeros((4, 4)), decimal=4)

    def test_plot_swift(self):
        r = rp.models.Panda()

        env = r.plot(r.q, block=False)
        env.close()


if __name__ == '__main__':  # pragma nocover
    unittest.main()
    # pytest.main(['tests/test_SerialLink.py'])
