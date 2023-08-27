"""
@author: Peter Corke
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
import unittest
import os
import spatialgeometry as sg
from spatialmath.base import tr2jac

# from spatialmath import SE3


class TestRobot(unittest.TestCase):
    def test_fkine(self):
        panda = rtb.models.ETS.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        # q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        # q3 = np.expand_dims(q1, 0)

        ans = np.array(
            [
                [-0.50827907, -0.57904589, 0.63746234, 0.44682295],
                [0.83014553, -0.52639462, 0.18375824, 0.16168396],
                [0.22915229, 0.62258699, 0.74824773, 0.96798113],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        nt.assert_array_almost_equal(panda.fkine(q1).A, ans)

    def test_jacob0(self):
        panda = rtb.models.ETS.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array(
            [
                [
                    -1.61683957e-01,
                    1.07925929e-01,
                    -3.41453006e-02,
                    3.35029257e-01,
                    -1.07195463e-02,
                    1.03187865e-01,
                    0.00000000e00,
                ],
                [
                    4.46822947e-01,
                    6.25741987e-01,
                    4.16474664e-01,
                    -8.04745724e-02,
                    7.78257566e-02,
                    -1.17720983e-02,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    -2.35276631e-01,
                    -8.20187641e-02,
                    -5.14076923e-01,
                    -9.98040745e-03,
                    -2.02626953e-01,
                    0.00000000e00,
                ],
                [
                    1.29458954e-16,
                    -9.85449730e-01,
                    3.37672585e-02,
                    -6.16735653e-02,
                    6.68449878e-01,
                    -1.35361558e-01,
                    6.37462344e-01,
                ],
                [
                    9.07021273e-18,
                    1.69967143e-01,
                    1.95778638e-01,
                    9.79165111e-01,
                    1.84470262e-01,
                    9.82748279e-01,
                    1.83758244e-01,
                ],
                [
                    1.00000000e00,
                    -2.26036604e-17,
                    9.80066578e-01,
                    -1.93473657e-01,
                    7.20517510e-01,
                    -1.26028049e-01,
                    7.48247732e-01,
                ],
            ]
        )

        panda.q = q1
        # nt.assert_array_almost_equal(panda.jacob0(), ans)
        nt.assert_array_almost_equal(panda.jacob0(q2), ans)
        nt.assert_array_almost_equal(panda.jacob0(q3), ans)
        nt.assert_array_almost_equal(panda.jacob0(q4), ans)
        self.assertRaises(TypeError, panda.jacob0, "Wfgsrth")

    def test_hessian0(self):
        panda = rtb.models.ETS.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array(
            [
                [
                    [
                        -4.46822947e-01,
                        -6.25741987e-01,
                        -4.16474664e-01,
                        8.04745724e-02,
                        -7.78257566e-02,
                        1.17720983e-02,
                        0.00000000e00,
                    ],
                    [
                        -6.25741987e-01,
                        -3.99892968e-02,
                        -1.39404950e-02,
                        -8.73761859e-02,
                        -1.69634134e-03,
                        -3.44399243e-02,
                        0.00000000e00,
                    ],
                    [
                        -4.16474664e-01,
                        -1.39404950e-02,
                        -4.24230421e-01,
                        -2.17748413e-02,
                        -7.82283735e-02,
                        -2.81325889e-02,
                        0.00000000e00,
                    ],
                    [
                        8.04745724e-02,
                        -8.73761859e-02,
                        -2.17748413e-02,
                        -5.18935898e-01,
                        5.28476698e-03,
                        -2.00682834e-01,
                        0.00000000e00,
                    ],
                    [
                        -7.78257566e-02,
                        -1.69634134e-03,
                        -7.82283735e-02,
                        5.28476698e-03,
                        -5.79159088e-02,
                        -2.88966443e-02,
                        0.00000000e00,
                    ],
                    [
                        1.17720983e-02,
                        -3.44399243e-02,
                        -2.81325889e-02,
                        -2.00682834e-01,
                        -2.88966443e-02,
                        -2.00614904e-01,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        -1.61683957e-01,
                        1.07925929e-01,
                        -3.41453006e-02,
                        3.35029257e-01,
                        -1.07195463e-02,
                        1.03187865e-01,
                        0.00000000e00,
                    ],
                    [
                        1.07925929e-01,
                        -2.31853293e-01,
                        -8.08253690e-02,
                        -5.06596965e-01,
                        -9.83518983e-03,
                        -1.99678676e-01,
                        0.00000000e00,
                    ],
                    [
                        -3.41453006e-02,
                        -8.08253690e-02,
                        -3.06951191e-02,
                        3.45709946e-01,
                        -1.01688580e-02,
                        1.07973135e-01,
                        0.00000000e00,
                    ],
                    [
                        3.35029257e-01,
                        -5.06596965e-01,
                        3.45709946e-01,
                        -9.65242924e-02,
                        1.45842251e-03,
                        -3.24608603e-02,
                        0.00000000e00,
                    ],
                    [
                        -1.07195463e-02,
                        -9.83518983e-03,
                        -1.01688580e-02,
                        1.45842251e-03,
                        -1.05221866e-03,
                        2.09794626e-01,
                        0.00000000e00,
                    ],
                    [
                        1.03187865e-01,
                        -1.99678676e-01,
                        1.07973135e-01,
                        -3.24608603e-02,
                        2.09794626e-01,
                        -4.04324654e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -6.34981134e-01,
                        -4.04611266e-01,
                        2.23596800e-02,
                        -7.48714002e-02,
                        -5.93773551e-03,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -4.04611266e-01,
                        2.07481281e-02,
                        -6.83089775e-02,
                        4.72662062e-03,
                        -2.05994912e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        2.23596800e-02,
                        -6.83089775e-02,
                        -3.23085806e-01,
                        5.69641385e-03,
                        -1.00311930e-01,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -7.48714002e-02,
                        4.72662062e-03,
                        5.69641385e-03,
                        5.40000550e-02,
                        -2.69041502e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        -5.93773551e-03,
                        -2.05994912e-02,
                        -1.00311930e-01,
                        -2.69041502e-02,
                        -9.98142073e-02,
                        0.00000000e00,
                    ],
                    [
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        -9.07021273e-18,
                        -2.77555756e-17,
                        -2.77555756e-17,
                        -1.11022302e-16,
                        -2.77555756e-17,
                        0.00000000e00,
                        -2.77555756e-17,
                    ],
                    [
                        -1.69967143e-01,
                        -1.97756387e-17,
                        4.11786040e-17,
                        -1.48932398e-16,
                        -5.07612940e-17,
                        -8.38219650e-17,
                        -4.90138154e-17,
                    ],
                    [
                        -1.95778638e-01,
                        1.66579116e-01,
                        -1.38777878e-17,
                        1.04083409e-17,
                        -1.38777878e-17,
                        3.46944695e-18,
                        0.00000000e00,
                    ],
                    [
                        -9.79165111e-01,
                        -3.28841647e-02,
                        -9.97525009e-01,
                        -4.16333634e-17,
                        -1.14491749e-16,
                        1.38777878e-17,
                        -6.24500451e-17,
                    ],
                    [
                        -1.84470262e-01,
                        1.22464303e-01,
                        -3.97312016e-02,
                        7.41195745e-01,
                        -2.77555756e-17,
                        1.12757026e-16,
                        2.77555756e-17,
                    ],
                    [
                        -9.82748279e-01,
                        -2.14206274e-02,
                        -9.87832342e-01,
                        6.67336352e-02,
                        -7.31335770e-01,
                        2.08166817e-17,
                        -6.07153217e-17,
                    ],
                    [
                        -1.83758244e-01,
                        1.27177529e-01,
                        -3.36043908e-02,
                        7.68210453e-01,
                        5.62842325e-03,
                        7.58497864e-01,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        1.29458954e-16,
                        -1.11022302e-16,
                        8.67361738e-17,
                        -4.16333634e-17,
                        5.55111512e-17,
                        2.77555756e-17,
                        5.55111512e-17,
                    ],
                    [
                        -9.85449730e-01,
                        -6.36381327e-17,
                        -1.02735399e-16,
                        -1.83043043e-17,
                        -5.63484308e-17,
                        8.08886307e-18,
                        1.07112702e-18,
                    ],
                    [
                        3.37672585e-02,
                        9.65806345e-01,
                        8.32667268e-17,
                        -2.55871713e-17,
                        1.07552856e-16,
                        2.08166817e-17,
                        -5.20417043e-18,
                    ],
                    [
                        -6.16735653e-02,
                        -1.90658563e-01,
                        -5.39111251e-02,
                        -6.59194921e-17,
                        -2.77555756e-17,
                        2.38524478e-17,
                        -4.16333634e-17,
                    ],
                    [
                        6.68449878e-01,
                        7.10033786e-01,
                        6.30795483e-01,
                        -8.48905588e-02,
                        0.00000000e00,
                        3.46944695e-17,
                        2.77555756e-17,
                    ],
                    [
                        -1.35361558e-01,
                        -1.24194307e-01,
                        -1.28407717e-01,
                        1.84162966e-02,
                        -1.32869389e-02,
                        2.77555756e-17,
                        -2.08166817e-17,
                    ],
                    [
                        6.37462344e-01,
                        7.37360525e-01,
                        5.99489263e-01,
                        -7.71850655e-02,
                        -4.08633244e-02,
                        2.09458434e-02,
                        0.00000000e00,
                    ],
                ],
                [
                    [
                        0.00000000e00,
                        -6.59521910e-17,
                        -1.31033786e-16,
                        -1.92457571e-16,
                        1.54134782e-17,
                        -7.69804929e-17,
                        1.11140361e-17,
                    ],
                    [
                        0.00000000e00,
                        -2.77555756e-17,
                        7.15573434e-17,
                        1.65666092e-16,
                        1.38777878e-17,
                        -8.67361738e-18,
                        3.46944695e-17,
                    ],
                    [
                        0.00000000e00,
                        -1.98669331e-01,
                        8.67361738e-18,
                        -1.46584134e-16,
                        6.02816408e-17,
                        -3.12250226e-17,
                        6.11490025e-17,
                    ],
                    [
                        0.00000000e00,
                        -9.54435515e-01,
                        4.51380881e-02,
                        1.38777878e-17,
                        1.08420217e-16,
                        3.46944695e-18,
                        6.24500451e-17,
                    ],
                    [
                        0.00000000e00,
                        -2.95400686e-01,
                        -1.24639152e-01,
                        -6.65899738e-01,
                        -4.85722573e-17,
                        -5.20417043e-18,
                        -5.55111512e-17,
                    ],
                    [
                        0.00000000e00,
                        -9.45442009e-01,
                        5.96856167e-02,
                        7.19317248e-02,
                        6.81888149e-01,
                        -2.77555756e-17,
                        1.04083409e-17,
                    ],
                    [
                        0.00000000e00,
                        -2.89432165e-01,
                        -1.18596498e-01,
                        -6.35513913e-01,
                        5.24032975e-03,
                        -6.51338823e-01,
                        0.00000000e00,
                    ],
                ],
            ]
        )

        ans_new = np.empty((7, 6, 7))

        for i in range(7):
            ans_new[i, :, :] = ans[:, :, i]

        nt.assert_array_almost_equal(panda.hessian0(q1), ans_new)
        nt.assert_array_almost_equal(panda.hessian0(q2), ans_new)
        nt.assert_array_almost_equal(panda.hessian0(q3), ans_new)
        nt.assert_array_almost_equal(panda.hessian0(q4), ans_new)
        nt.assert_array_almost_equal(panda.hessian0(J0=panda.jacob0(q1)), ans_new)
        nt.assert_array_almost_equal(
            panda.hessian0(q=None, J0=panda.jacob0(q1)), ans_new
        )

    def test_manipulability(self):
        panda = rtb.models.ETS.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]

        ans = 0.006559178039088341

        panda.q = q1
        nt.assert_array_almost_equal(panda.manipulability(q2), ans)
        # self.assertRaises(ValueError, panda.manipulability)
        self.assertRaises(TypeError, panda.manipulability, "Wfgsrth")
        self.assertRaises(ValueError, panda.manipulability, [1, 3])

    def test_qlim(self):
        panda = rtb.models.ETS.Panda()

        self.assertEqual(panda.qlim.shape[0], 2)
        self.assertEqual(panda.qlim.shape[1], panda.n)

    def test_manuf(self):
        panda = rtb.models.ETS.Panda()

        self.assertIsInstance(panda.manufacturer, str)

    def test_complex(self):
        l0 = rtb.Link(rtb.ET.tx(0.1) * rtb.ET.Rx())
        l1 = rtb.Link(rtb.ET.tx(0.1) * rtb.ET.Ry(), parent=l0)
        l2 = rtb.Link(rtb.ET.tx(0.1) * rtb.ET.Rz(), parent=l1)
        l3 = rtb.Link(rtb.ET.tx(0.1) * rtb.ET.tx(), parent=l2)
        l4 = rtb.Link(rtb.ET.tx(0.1) * rtb.ET.ty(), parent=l3)
        l5 = rtb.Link(rtb.ET.tx(0.1) * rtb.ET.tz(), parent=l4)

        r = rtb.Robot([l0, l1, l2, l3, l4, l5])
        q = [1.0, 2, 3, 1, 2, 3]

        ans = np.array(
            [
                [-0.0, 0.08752679, -0.74761985, 0.41198225, 0.05872664, 0.90929743],
                [
                    1.46443609,
                    2.80993063,
                    0.52675075,
                    -0.68124272,
                    -0.64287284,
                    0.35017549,
                ],
                [
                    -1.04432,
                    -1.80423571,
                    -2.20308833,
                    0.60512725,
                    -0.76371834,
                    -0.2248451,
                ],
                [1.0, 0.0, 0.90929743, 0.0, 0.0, 0.0],
                [0.0, 0.54030231, 0.35017549, 0.0, 0.0, 0.0],
                [0.0, 0.84147098, -0.2248451, 0.0, 0.0, 0.0],
            ]
        )

        nt.assert_array_almost_equal(r.jacob0(q), ans)

    def test_copy_init(self):
        r = rtb.models.Panda()

        r2 = rtb.Robot(r)

        r2.jacob0(r.q)

        self.assertEqual(r.n, r2.n)

    def test_init2(self):
        r = rtb.Robot(rtb.ETS(rtb.ET.Ry(qlim=[-1, 1])))

        self.assertEqual(r.n, 1)

    def test_to_dict(self):
        r = rtb.models.Panda()

        rdict = r._to_dict(collision_alpha=0.5)
        rdict2 = r._to_dict()

        self.assertTrue(len(rdict) > len(rdict2))

        self.assertIsInstance(rdict, list)

    def test_fk_dict(self):
        r = rtb.models.Panda()

        rdict = r._fk_dict(collision_alpha=0.5)
        rdict2 = r._fk_dict()

        self.assertTrue(len(rdict) > len(rdict2))

    def test_URDF(self):
        r = rtb.Robot.URDF("fetch_description/robots/fetch.urdf", gripper=6)

        self.assertEqual(r.n, 5)

    def test_URDF2(self):
        r = rtb.Robot.URDF(
            "fetch_description/robots/fetch.urdf", gripper="forearm_roll_link"
        )

        self.assertEqual(r.n, 7)

    def test_showgraph(self):
        r = rtb.models.Panda()

        file = r.showgraph(display_graph=False)

        self.assertIsNotNone(file)

        self.assertTrue(file[-4:] == ".pdf")  # type: ignore

    def test_dotfile(self):
        r = rtb.models.Panda()

        r.dotfile("test.dot")
        try:
            os.remove("test.dot")
        except PermissionError:
            pass

    def test_dotfile2(self):
        r = rtb.models.Frankie()

        r.dotfile("test.dot", jtype=True, etsbox=True)
        try:
            os.remove("test.dot")
        except PermissionError:
            pass

    def test_dotfile3(self):
        r = rtb.models.Panda()

        r.dotfile("test.dot", ets="brief")
        try:
            os.remove("test.dot")
        except PermissionError:
            pass

    def test_dotfile4(self):
        r = rtb.models.Panda()

        r.dotfile("test.dot", ets="None")  # type: ignore
        try:
            os.remove("test.dot")
        except PermissionError:
            pass

    def test_fkine_all(self):
        r = rtb.models.ETS.Panda()

        r.fkine_all(r.q)

    def test_fkine_all2(self):
        r = rtb.models.YuMi()

        r.fkine_all(r.q)

    def test_yoshi(self):
        puma = rtb.models.Puma560()
        q = puma.qn  # type: ignore

        m1 = puma.manipulability(q)
        m2 = puma.manipulability(np.c_[q, q].T)
        m3 = puma.manipulability(q, axes="trans")
        m4 = puma.manipulability(q, axes="rot")

        a0 = 0.0805
        a2 = 0.1354
        a3 = 2.44949

        nt.assert_almost_equal(m1, a0, decimal=4)
        nt.assert_almost_equal(m2[0], a0, decimal=4)  # type: ignore
        nt.assert_almost_equal(m2[1], a0, decimal=4)  # type: ignore
        nt.assert_almost_equal(m3, a2, decimal=4)
        nt.assert_almost_equal(m4, a3, decimal=4)

        with self.assertRaises(ValueError):
            puma.manipulability(axes="abcdef")  # type: ignore

    def test_asada(self):
        l1 = rtb.Link(ets=rtb.ETS(rtb.ET.Ry()), m=1, r=[0.5, 0, 0], name="l1")
        l2 = rtb.Link(
            ets=rtb.ETS(rtb.ET.tx(1)) * rtb.ET.Ry(),
            m=1,
            r=[0.5, 0, 0],
            parent=l1,
            name="l2",
        )
        r = rtb.Robot([l1, l2], name="simple 2 link")
        q = np.array([1.0, 1.5])

        m1 = r.manipulability(q, method="asada")
        m2 = r.manipulability(np.c_[q, q].T, method="asada")
        m3 = r.manipulability(q, axes="trans", method="asada")
        m4 = r.manipulability(q, axes="rot", method="asada")

        a0 = 0.0
        a2 = 0.0
        a3 = 0.0

        nt.assert_almost_equal(m1, a0, decimal=4)
        nt.assert_almost_equal(m2[0], a0, decimal=4)  # type: ignore
        nt.assert_almost_equal(m2[1], a0, decimal=4)  # type: ignore
        nt.assert_almost_equal(m3, a2, decimal=4)
        nt.assert_almost_equal(m4, a3, decimal=4)

    def test_cond(self):
        r = rtb.models.Panda()

        m = r.manipulability(r.qr, method="invcondition")

        self.assertAlmostEqual(m, 0.11222, places=4)  # type: ignore

    def test_minsingular(self):
        r = rtb.models.Panda()

        m = r.manipulability(r.qr, method="minsingular")

        self.assertAlmostEqual(m, 0.209013, places=4)  # type: ignore

    def test_jtraj(self):
        r = rtb.models.Panda()

        q1 = r.q + 0.2

        q = r.jtraj(r.fkine(q1), r.fkine(r.qr), 5)

        self.assertEqual(q.s.shape, (5, 7))

    def test_jtraj2(self):
        r = rtb.models.DH.Puma560()

        q1 = r.q + 0.2

        q = r.jtraj(r.fkine(q1), r.fkine(r.qr), 5)

        self.assertEqual(q.s.shape, (5, 6))

    def test_manip(self):
        r = rtb.models.Panda()
        q = r.qr

        m1 = r.manipulability(q)
        m2 = r.manipulability(q, axes="trans")
        m3 = r.manipulability(q, axes="rot")

        nt.assert_almost_equal(m1, 0.0837, decimal=4)
        nt.assert_almost_equal(m2, 0.1438, decimal=4)
        nt.assert_almost_equal(m3, 2.7455, decimal=4)

        with self.assertRaises(ValueError):
            r.manipulability(axes="abcdef")  # type: ignore

    def test_jacobm(self):
        r = rtb.models.Panda()
        q = r.qr

        m1 = r.jacobm(q)
        m2 = r.jacobm(q, axes="trans")
        m3 = r.jacobm(q, axes="rot")

        a1 = np.array(
            [
                [0.00000000e00],
                [-2.62678438e-03],
                [1.18662211e-19],
                [4.06398364e-02],
                [1.21226717e-19],
                [-2.73383661e-02],
                [0.00000000e00],
            ]
        )

        a2 = np.array(
            [
                [-4.03109907e-32],
                [2.14997718e-02],
                [2.57991732e-18],
                [9.51555140e-02],
                [1.09447194e-18],
                [3.78529920e-02],
                [0.00000000e00],
            ]
        )

        a3 = np.array(
            [
                [-1.22157098e-31],
                [-7.51299508e-01],
                [9.51556025e-17],
                [7.49956218e-01],
                [8.40346012e-18],
                [-5.17677915e-01],
                [0.00000000e00],
            ]
        )

        nt.assert_almost_equal(m1, a1, decimal=4)
        nt.assert_almost_equal(m2, a2, decimal=4)
        nt.assert_almost_equal(m3, a3, decimal=4)

        with self.assertRaises(ValueError):
            r.jacobm(axes="abcdef")  # type: ignore

    def test_collided2(self):
        p = rtb.models.Panda()
        s0 = sg.Cuboid([0.01, 0.01, 0.01], pose=p.fkine(p.q))

        c0 = p.collided(p.q, s0)

        self.assertTrue(c0)

    def test_velocity_damper(self):
        r = rtb.models.Panda()

        Ain, Bin = r.joint_velocity_damper()

        a1 = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )

        a2 = np.array([0.0, 0.0, 0.0, -2.396, 0.0, -0.65, 0.0])

        nt.assert_almost_equal(Ain, a1, decimal=4)
        nt.assert_almost_equal(Bin, a2, decimal=4)

    def test_link_collision_damper(self):
        r = rtb.models.Panda()

        s = sg.Cuboid([0.01, 0.01, 0.01])

        Ain, Bin = r.link_collision_damper(s, r.q)

        a1 = np.array(
            [
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -1.93649378e-34,
                    -1.71137143e-18,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    5.05166785e-18,
                    -8.25000000e-02,
                    5.05166785e-18,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    8.25000000e-02,
                    1.01033361e-17,
                    8.25000000e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -8.25000000e-02,
                    -1.01033361e-17,
                    -8.25000000e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -2.75268296e-35,
                    2.90883510e-18,
                    -1.55445626e-34,
                    -8.25000000e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    -8.25000000e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    -8.25000000e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -1.57039486e-11,
                    1.11890634e-10,
                    -1.57039486e-11,
                    -6.09481971e-02,
                    -1.57039486e-11,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    -4.89389206e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    -7.69198080e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    1.20407156e-14,
                    -2.80950032e-13,
                    1.20407156e-14,
                    9.05269566e-14,
                    1.20407156e-14,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -1.13663976e-10,
                    6.90000000e-01,
                    -1.13663996e-10,
                    -3.74000000e-01,
                    -1.13663972e-10,
                    1.00000000e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    1.07768918e-17,
                    -8.80000000e-02,
                    1.07768918e-17,
                    5.50000000e-03,
                    1.07768918e-17,
                    8.80000000e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -1.07768918e-17,
                    8.80000000e-02,
                    -1.07768918e-17,
                    -5.50000000e-03,
                    -1.07768918e-17,
                    -8.80000000e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -1.42800820e-02,
                    -5.22756456e-01,
                    -1.42800820e-02,
                    2.60933208e-01,
                    -1.42800820e-02,
                    -1.20480747e-01,
                    3.09364536e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    5.96886490e-02,
                    -3.78932484e-01,
                    5.96886490e-02,
                    1.87911850e-01,
                    5.96886490e-02,
                    -9.58635879e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    5.61245282e-02,
                    -3.41025601e-01,
                    5.61245282e-02,
                    1.75113444e-01,
                    5.61245282e-02,
                    -1.05419509e-01,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    6.22253967e-02,
                    -3.91030050e-01,
                    6.22253967e-02,
                    1.67584307e-01,
                    6.22253967e-02,
                    -1.03944697e-01,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -4.90028111e-02,
                    2.75492780e-01,
                    -4.90028111e-02,
                    -1.50373572e-01,
                    -4.90028111e-02,
                    1.14302308e-01,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    4.90028111e-02,
                    -3.83963034e-01,
                    4.90028111e-02,
                    1.57152963e-01,
                    4.90028111e-02,
                    -5.83205400e-03,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    6.22246462e-02,
                    -3.48609107e-01,
                    6.22246462e-02,
                    1.25159489e-01,
                    6.22246462e-02,
                    -1.46371610e-01,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -2.68888497e-02,
                    1.03162870e-01,
                    -2.68888497e-02,
                    -8.10072778e-02,
                    -2.68888497e-02,
                    1.10725707e-01,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    2.68888497e-02,
                    -2.61882491e-01,
                    2.68888497e-02,
                    9.09272541e-02,
                    2.68888497e-02,
                    4.79939142e-02,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
            ]
        )
        a2 = np.array(
            [
                2.00000000e-02,
                -1.27216162e-01,
                -3.34978685e-02,
                -2.00000000e-02,
                8.72000000e-01,
                -2.60000000e-01,
                -4.60000000e-01,
                -2.20000000e-01,
                -2.20000000e-01,
                6.00000000e-02,
                4.20000000e-01,
                -1.80000000e-01,
                -4.60000000e-01,
                -2.20000000e-01,
                -2.20000000e-01,
                6.20000000e-01,
                7.80000000e-01,
                3.80000000e-01,
                1.00622747e-01,
                7.36135591e-02,
                5.36875734e-01,
                -2.60000000e-01,
                -3.80000000e-01,
                -1.40000000e-01,
                -3.80000000e-01,
                2.77555756e-17,
                -1.40000000e-01,
                -5.82574174e-02,
                -9.75448830e-02,
                -8.31641223e-02,
                -2.20000000e-01,
                -1.30350690e-01,
                -1.30350690e-01,
                1.00004623e-01,
                1.41807468e-01,
                1.41807468e-01,
            ]
        )

        nt.assert_almost_equal(Ain, a1, decimal=4)  # type: ignore
        nt.assert_almost_equal(Bin, a2, decimal=4)  # type: ignore

    def test_hessiane(self):
        deg = np.pi / 180
        mm = 1e-3
        tool_offset = (103) * mm

        l0 = rtb.ET.tz(0.333) * rtb.ET.Rz(jindex=0)

        l1 = rtb.ET.Rx(-90 * deg) * rtb.ET.Rz(jindex=1)

        l2 = rtb.ET.Rx(90 * deg) * rtb.ET.tz(0.316) * rtb.ET.Rz(jindex=2)

        l3 = rtb.ET.tx(0.0825) * rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=3)

        l4 = (
            rtb.ET.tx(-0.0825)
            * rtb.ET.Rx(-90 * deg)
            * rtb.ET.tz(0.384)
            * rtb.ET.Rz(jindex=4)
        )

        l5 = rtb.ET.Rx(90 * deg) * rtb.ET.Rz(jindex=5)

        l6 = (
            rtb.ET.tx(0.088)
            * rtb.ET.Rx(90 * deg)
            * rtb.ET.tz(0.107)
            * rtb.ET.Rz(jindex=6)
        )

        ee = rtb.ET.tz(tool_offset) * rtb.ET.Rz(-np.pi / 4)

        r = rtb.Robot(l0 + l1 + l2 + l3 + l4 + l5 + l6 + ee)

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        H0 = r.hessian0(q1)
        He = np.empty((r.n, 6, r.n))
        T = r.fkine(q1, include_base=False).A

        for i in range(r.n):
            He[i, :, :] = tr2jac(T.T) @ H0[i, :, :]

        J = r.jacobe(q1)

        nt.assert_array_almost_equal(r.hessiane(q1), He)
        nt.assert_array_almost_equal(r.hessiane(q2), He)
        nt.assert_array_almost_equal(r.hessiane(q3), He)
        nt.assert_array_almost_equal(r.hessiane(q4), He)

        nt.assert_array_almost_equal(r.hessiane(Je=J), He)
        nt.assert_array_almost_equal(r.hessiane(Je=J), He)
        nt.assert_array_almost_equal(r.hessiane(Je=J), He)
        nt.assert_array_almost_equal(r.hessiane(Je=J), He)

    def test_erobot(self):
        ets = rtb.ETS(rtb.ET.Rz())
        robot = rtb.ERobot(
            ets, name="myname", manufacturer="I made it", comment="other stuff"
        )
        self.assertEqual(robot.name, "myname")
        self.assertEqual(robot.manufacturer, "I made it")
        self.assertEqual(robot.comment, "other stuff")

    def test_erobot2(self):
        ets = rtb.ETS2(rtb.ET2.R())
        robot = rtb.ERobot2(
            ets, name="myname", manufacturer="I made it", comment="other stuff"
        )
        self.assertEqual(robot.name, "myname")
        self.assertEqual(robot.manufacturer, "I made it")
        self.assertEqual(robot.comment, "other stuff")

    def test_qlim_setters(self):
        et = rtb.ET.Rz(qlim=[-1, 1])
        ets = rtb.ETS([et])
        l = rtb.Link(ets)
        r = rtb.Robot([l])

        nt.assert_almost_equal(et.qlim, np.array([-1, 1]))
        nt.assert_almost_equal(ets.qlim, np.array([[-1, 1]]).T)
        nt.assert_almost_equal(l.qlim, np.array([-1, 1]))
        nt.assert_almost_equal(r.qlim, np.array([[-1, 1]]).T)

        et.qlim = [-2, 2]
        nt.assert_almost_equal(et.qlim, np.array([-2, 2]))
        nt.assert_almost_equal(ets.qlim, np.array([[-1, 1]]).T)
        nt.assert_almost_equal(l.qlim, np.array([-1, 1]))
        nt.assert_almost_equal(r.qlim, np.array([[-1, 1]]).T)

        ets.qlim = np.array([[-2, 2]]).T
        nt.assert_almost_equal(et.qlim, np.array([-2, 2]))
        nt.assert_almost_equal(ets.qlim, np.array([[-2, 2]]).T)
        nt.assert_almost_equal(l.qlim, np.array([-2, 2]))
        nt.assert_almost_equal(r.qlim, np.array([[-2, 2]]).T)

        l.qlim = [-3, 3]
        nt.assert_almost_equal(et.qlim, np.array([-2, 2]))
        nt.assert_almost_equal(ets.qlim, np.array([[-3, 3]]).T)
        nt.assert_almost_equal(l.qlim, np.array([-3, 3]))
        nt.assert_almost_equal(r.qlim, np.array([[-3, 3]]).T)

        r.qlim = np.array([[-4, 4]]).T
        nt.assert_almost_equal(et.qlim, np.array([-2, 2]))
        nt.assert_almost_equal(ets.qlim, np.array([[-4, 4]]).T)
        nt.assert_almost_equal(l.qlim, np.array([-4, 4]))
        nt.assert_almost_equal(r.qlim, np.array([[-4, 4]]).T)


if __name__ == "__main__":  # pragma nocover
    unittest.main()
