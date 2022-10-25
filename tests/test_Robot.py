"""
@author: Peter Corke
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm
import unittest


class TestRobot(unittest.TestCase):
    def test_fkine(self):
        panda = rtb.models.ETS.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)

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

    # def test_init(self):
    #     l0 = rp.PrismaticDH()
    #     l1 = rp.RevoluteDH()

    #     with self.assertRaises(TypeError):
    #         rp.DHRobot([l0, l1], keywords=1)

    #     with self.assertRaises(TypeError):
    #         rp.Robot(l0)

    #     with self.assertRaises(TypeError):
    #         rp.Robot([l0, 1])

    # def test_configurations_str(self):
    #     r = rp.models.DH.Puma560()
    #     r.configurations_str()

    #     r2 = rp.models.ETS.Frankie()
    #     r2.configurations_str()

    # def test_dyntable(self):
    #     r = rp.models.DH.Puma560()
    #     r.dynamics()

    # def test_linkcolormap(self):
    #     r = rp.models.DH.Puma560()
    #     r.linkcolormap()

    #     r.linkcolormap(["r", "r", "r", "r", "r", "r"])

    # # def test_tool_error(self):
    # #     r = rp.models.DH.Puma560()

    # #     with self.assertRaises(ValueError):
    # #         r.tool = 2

    # def test_links(self):

    #     l0 = rp.PrismaticDH()
    #     l1 = rp.RevoluteDH()
    #     l2 = rp.PrismaticDH()
    #     l3 = rp.RevoluteDH()

    #     r0 = rp.DHRobot([l0, l1, l2, l3])

    #     self.assertIs(r0[0], l0)
    #     self.assertIs(r0[1], l1)
    #     self.assertIs(r0[2], l2)
    #     self.assertIs(r0[3], l3)

    # def test_ikine_LM(self):
    #     panda = rp.models.DH.Panda()
    #     q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
    #     T = panda.fkine(q)
    #     Tt = sm.SE3([T, T])

    #     qr = [0.0342, 1.6482, 0.0312, 1.2658, -0.0734, 0.4836, 0.7489]

    #     sol1 = panda.ikine_LM(T)
    #     sol2 = panda.ikine_LM(Tt)
    #     sol3 = panda.ikine_LM(T, q0=[0, 1.4, 0, 1, 0, 0.5, 1])

    #     self.assertTrue(sol1.success)
    #     self.assertAlmostEqual(np.linalg.norm(T - panda.fkine(sol1.q)), 0, places=4)

    #     self.assertTrue(sol2.success[0])
    #     self.assertAlmostEqual(
    #         np.linalg.norm(T - panda.fkine(sol2.q[0, :])), 0, places=4
    #     )
    #     self.assertTrue(sol2.success[0])
    #     self.assertAlmostEqual(
    #         np.linalg.norm(T - panda.fkine(sol2.q[1, :])), 0, places=4
    #     )

    #     self.assertTrue(sol3.success)
    #     self.assertAlmostEqual(np.linalg.norm(T - panda.fkine(sol3.q)), 0, places=4)

    #     with self.assertRaises(ValueError):
    #         panda.ikine_LM(T, q0=[1, 2])

    # def test_ikine_LM_mask(self):

    #     # simple RR manipulator, solve with mask

    #     l0 = rp.RevoluteDH(a=2.0)
    #     l1 = rp.RevoluteDH(a=1)

    #     r = rp.DHRobot([l0, l1])  # RR manipulator
    #     T = sm.SE3(-1, 2, 0)
    #     sol = r.ikine_LM(T, mask=[1, 1, 0, 0, 0, 0])

    #     self.assertTrue(sol.success)
    #     self.assertAlmostEqual(
    #         np.linalg.norm(T.t[:2] - r.fkine(sol.q).t[:2]), 0, places=4
    #     )

    #     # test initial condition search, drop iteration limit so it has to do
    #     # some searching
    #     sol = r.ikine_LM(T, mask=[1, 1, 0, 0, 0, 0], ilimit=8, search=True)

    #     self.assertTrue(sol.success)
    #     self.assertAlmostEqual(
    #         np.linalg.norm(T.t[:2] - r.fkine(sol.q).t[:2]), 0, places=4
    #     )

    # def test_ikine_LM_transpose(self):
    #     # need to test this on a robot with squarish Jacobian, choose Puma

    #     r = rp.models.DH.Puma560()
    #     T = r.fkine(r.qn)

    #     sol = r.ikine_LM(T, transpose=0.5, ilimit=1000, tol=1e-6)

    #     self.assertTrue(sol.success)
    #     self.assertAlmostEqual(np.linalg.norm(T - r.fkine(sol.q)), 0, places=4)

    # def test_ikine_con(self):
    #     panda = rp.models.DH.Panda()
    #     q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
    #     T = panda.fkine(q)
    #     Tt = sm.SE3([T, T, T])

    #     # qr = [7.69161412e-04, 9.01409257e-01, -1.46372859e-02,
    #     #       -6.98000000e-02, 1.38978915e-02, 9.62104811e-01,
    #     #       7.84926515e-01]

    #     sol1 = panda.ikine_min(T, qlim=True, q0=np.zeros(7))
    #     sol2 = panda.ikine_min(Tt, qlim=True)

    #     self.assertTrue(sol1.success)
    #     self.assertAlmostEqual(np.linalg.norm(T - panda.fkine(sol1.q)), 0, places=4)
    #     nt.assert_array_almost_equal(
    #         T.A - panda.fkine(sol1.q).A, np.zeros((4, 4)), decimal=4
    #     )

    #     self.assertTrue(sol2[0].success)
    #     nt.assert_array_almost_equal(
    #         T.A - panda.fkine(sol2[0].q).A, np.zeros((4, 4)), decimal=4
    #     )
    #     self.assertTrue(sol2[1].success)
    #     nt.assert_array_almost_equal(
    #         T.A - panda.fkine(sol2[1].q).A, np.zeros((4, 4)), decimal=4
    #     )

    # def test_ikine_unc(self):
    #     puma = rp.models.DH.Puma560()
    #     q = puma.qn
    #     T = puma.fkine(q)
    #     Tt = sm.SE3([T, T])

    #     sol1 = puma.ikine_min(Tt)
    #     sol2 = puma.ikine_min(T)
    #     sol3 = puma.ikine_min(T)

    #     self.assertTrue(sol1[0].success)
    #     nt.assert_array_almost_equal(
    #         T.A - puma.fkine(sol1[0].q).A, np.zeros((4, 4)), decimal=4
    #     )
    #     self.assertTrue(sol1[1].success)
    #     nt.assert_array_almost_equal(
    #         T.A - puma.fkine(sol1[1].q).A, np.zeros((4, 4)), decimal=4
    #     )

    #     self.assertTrue(sol2.success)
    #     nt.assert_array_almost_equal(
    #         T.A - puma.fkine(sol2.q).A, np.zeros((4, 4)), decimal=4
    #     )

    #     self.assertTrue(sol3.success)
    #     nt.assert_array_almost_equal(
    #         T.A - puma.fkine(sol3.q).A, np.zeros((4, 4)), decimal=4
    # )

    # def test_plot_swift(self):
    #     r = rp.models.Panda()

    #     env = r.plot(r.q, block=False)
    #     env.close()


if __name__ == "__main__":  # pragma nocover
    unittest.main()
