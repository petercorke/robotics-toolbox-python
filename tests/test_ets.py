#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rp
import unittest
import spatialmath as sp


class TestETS(unittest.TestCase):

    def test_panda(self):
        panda = rp.Panda()
        qz = np.array([0, 0, 0, 0, 0, 0, 0])
        qr = np.array([0, -90, -90, 90, 0, -90, 90]) * np.pi/180

        nt.assert_array_almost_equal(panda.qr, qr)
        nt.assert_array_almost_equal(panda.qz, qz)

    def test_q(self):
        panda = rp.Panda()

        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)

        panda.q = q1
        nt.assert_array_almost_equal(panda.q, q1)
        panda.q = q2
        nt.assert_array_almost_equal(panda.q, q2)
        panda.q = q3
        nt.assert_array_almost_equal(np.expand_dims(panda.q, 0), q3)

    def test_base(self):
        panda = rp.Panda()

        pose = sp.SE3()

        panda.base = pose.A
        nt.assert_array_almost_equal(np.eye(4), panda.base)

        panda.base = pose
        nt.assert_array_almost_equal(np.eye(4), panda.base)

    def test_str(self):
        panda = rp.Panda()

        ans = '\nPanda (Franka Emika): 7 axis, RzRzRzRzRzRzRz, ETS\n'\
            'Elementary Transform Sequence:\n'\
            '[tz(0.333), Rz(q1), Rx(-90), Rz(q2), Rx(90), tz(0.316), '\
            'Rz(q3), tx(0.0825), Rx(90), Rz(q4), tx(-0.0825), Rx(-90), '\
            'tz(0.384), Rz(q5), Rx(90), Rz(q6), tx(0.088), Rx(90), '\
            'tz(0.107), Rz(q7)]\n'\
            'tool:  t = (0, 0, 0.103),  RPY/xyz = (0, 0, -45) deg'

        self.assertEqual(str(panda), ans)

    def test_str_ets(self):
        panda = rp.Panda()

        ans = '[tz(0.333), Rz(q1), Rx(-90), Rz(q2), Rx(90), tz(0.316), '\
            'Rz(q3), tx(0.0825), Rx(90), Rz(q4), tx(-0.0825), Rx(-90), '\
            'tz(0.384), Rz(q5), Rx(90), Rz(q6), tx(0.088), Rx(90), '\
            'tz(0.107), Rz(q7)]'

        self.assertEqual(str(panda.ets), ans)

    def test_fkine(self):
        panda = rp.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)

        ans = np.array([
            [-0.50827907, -0.57904589,  0.63746234,  0.44682295],
            [0.83014553,  -0.52639462,  0.18375824,  0.16168396],
            [0.22915229,   0.62258699,  0.74824773,  0.96798113],
            [0.,           0.,          0.,          1.]
        ])

        nt.assert_array_almost_equal(panda.fkine(q1), ans)
        nt.assert_array_almost_equal(panda.fkine(q2), ans)
        nt.assert_array_almost_equal(panda.fkine(q3), ans)
        nt.assert_array_almost_equal(panda.fkine(q3), ans)
        self.assertRaises(TypeError, panda.fkine, 'Wfgsrth')

    def test_jacob0(self):
        panda = rp.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array([
            [-1.61683957e-01,  1.07925929e-01, -3.41453006e-02,
                3.35029257e-01, -1.07195463e-02,  1.03187865e-01,
                0.00000000e+00],
            [4.46822947e-01,  6.25741987e-01,  4.16474664e-01,
                -8.04745724e-02,  7.78257566e-02, -1.17720983e-02,
                0.00000000e+00],
            [0.00000000e+00, -2.35276631e-01, -8.20187641e-02,
                -5.14076923e-01, -9.98040745e-03, -2.02626953e-01,
                0.00000000e+00],
            [1.29458954e-16, -9.85449730e-01,  3.37672585e-02,
                -6.16735653e-02,  6.68449878e-01, -1.35361558e-01,
                6.37462344e-01],
            [9.07021273e-18,  1.69967143e-01,  1.95778638e-01,
                9.79165111e-01,  1.84470262e-01,  9.82748279e-01,
                1.83758244e-01],
            [1.00000000e+00, -2.26036604e-17,  9.80066578e-01,
                -1.93473657e-01,  7.20517510e-01, -1.26028049e-01,
                7.48247732e-01]
        ])

        nt.assert_array_almost_equal(panda.jacob0(q1), ans)
        nt.assert_array_almost_equal(panda.jacob0(q2), ans)
        nt.assert_array_almost_equal(panda.jacob0(q3), ans)
        nt.assert_array_almost_equal(panda.jacob0(q4), ans)
        self.assertRaises(TypeError, panda.jacob0, 'Wfgsrth')

    def test_hessian0(self):
        panda = rp.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array([
            [
                [-4.46822947e-01, -6.25741987e-01, -4.16474664e-01,
                    8.04745724e-02, -7.78257566e-02,  1.17720983e-02,
                    0.00000000e+00],
                [-6.25741987e-01, -3.99892968e-02, -1.39404950e-02,
                    -8.73761859e-02, -1.69634134e-03, -3.44399243e-02,
                    0.00000000e+00],
                [-4.16474664e-01, -1.39404950e-02, -4.24230421e-01,
                    -2.17748413e-02, -7.82283735e-02, -2.81325889e-02,
                    0.00000000e+00],
                [8.04745724e-02, -8.73761859e-02, -2.17748413e-02,
                    -5.18935898e-01,  5.28476698e-03, -2.00682834e-01,
                    0.00000000e+00],
                [-7.78257566e-02, -1.69634134e-03, -7.82283735e-02,
                    5.28476698e-03, -5.79159088e-02, -2.88966443e-02,
                    0.00000000e+00],
                [1.17720983e-02, -3.44399243e-02, -2.81325889e-02,
                    -2.00682834e-01, -2.88966443e-02, -2.00614904e-01,
                    0.00000000e+00],
                [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00]
            ],
            [
                [-1.61683957e-01,  1.07925929e-01, -3.41453006e-02,
                    3.35029257e-01, -1.07195463e-02,  1.03187865e-01,
                    0.00000000e+00],
                [1.07925929e-01, -2.31853293e-01, -8.08253690e-02,
                    -5.06596965e-01, -9.83518983e-03, -1.99678676e-01,
                    0.00000000e+00],
                [-3.41453006e-02, -8.08253690e-02, -3.06951191e-02,
                    3.45709946e-01, -1.01688580e-02,  1.07973135e-01,
                    0.00000000e+00],
                [3.35029257e-01, -5.06596965e-01,  3.45709946e-01,
                    -9.65242924e-02,  1.45842251e-03, -3.24608603e-02,
                    0.00000000e+00],
                [-1.07195463e-02, -9.83518983e-03, -1.01688580e-02,
                    1.45842251e-03, -1.05221866e-03,  2.09794626e-01,
                    0.00000000e+00],
                [1.03187865e-01, -1.99678676e-01,  1.07973135e-01,
                    -3.24608603e-02,  2.09794626e-01, -4.04324654e-02,
                    0.00000000e+00],
                [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00]
            ],
            [
                [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00],
                [0.00000000e+00, -6.34981134e-01, -4.04611266e-01,
                    2.23596800e-02, -7.48714002e-02, -5.93773551e-03,
                    0.00000000e+00],
                [0.00000000e+00, -4.04611266e-01,  2.07481281e-02,
                    -6.83089775e-02,  4.72662062e-03, -2.05994912e-02,
                    0.00000000e+00],
                [0.00000000e+00,  2.23596800e-02, -6.83089775e-02,
                    -3.23085806e-01,  5.69641385e-03, -1.00311930e-01,
                    0.00000000e+00],
                [0.00000000e+00, -7.48714002e-02,  4.72662062e-03,
                    5.69641385e-03,  5.40000550e-02, -2.69041502e-02,
                    0.00000000e+00],
                [0.00000000e+00, -5.93773551e-03, -2.05994912e-02,
                    -1.00311930e-01, -2.69041502e-02, -9.98142073e-02,
                    0.00000000e+00],
                [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                    0.00000000e+00]
            ],
            [
                [-9.07021273e-18, -2.77555756e-17, -2.77555756e-17,
                    -1.11022302e-16, -2.77555756e-17,  0.00000000e+00,
                    -2.77555756e-17],
                [-1.69967143e-01, -1.97756387e-17,  4.11786040e-17,
                    -1.48932398e-16, -5.07612940e-17, -8.38219650e-17,
                    -4.90138154e-17],
                [-1.95778638e-01,  1.66579116e-01, -1.38777878e-17,
                    1.04083409e-17, -1.38777878e-17,  3.46944695e-18,
                    0.00000000e+00],
                [-9.79165111e-01, -3.28841647e-02, -9.97525009e-01,
                    -4.16333634e-17, -1.14491749e-16,  1.38777878e-17,
                    -6.24500451e-17],
                [-1.84470262e-01,  1.22464303e-01, -3.97312016e-02,
                    7.41195745e-01, -2.77555756e-17,  1.12757026e-16,
                    2.77555756e-17],
                [-9.82748279e-01, -2.14206274e-02, -9.87832342e-01,
                    6.67336352e-02, -7.31335770e-01,  2.08166817e-17,
                    -6.07153217e-17],
                [-1.83758244e-01,  1.27177529e-01, -3.36043908e-02,
                    7.68210453e-01,  5.62842325e-03,  7.58497864e-01,
                    0.00000000e+00]
            ],
            [
                [1.29458954e-16, -1.11022302e-16,  8.67361738e-17,
                    -4.16333634e-17,  5.55111512e-17,  2.77555756e-17,
                    5.55111512e-17],
                [-9.85449730e-01, -6.36381327e-17, -1.02735399e-16,
                    -1.83043043e-17, -5.63484308e-17,  8.08886307e-18,
                    1.07112702e-18],
                [3.37672585e-02,  9.65806345e-01,  8.32667268e-17,
                    -2.55871713e-17,  1.07552856e-16,  2.08166817e-17,
                    -5.20417043e-18],
                [-6.16735653e-02, -1.90658563e-01, -5.39111251e-02,
                    -6.59194921e-17, -2.77555756e-17,  2.38524478e-17,
                    -4.16333634e-17],
                [6.68449878e-01,  7.10033786e-01,  6.30795483e-01,
                    -8.48905588e-02,  0.00000000e+00,  3.46944695e-17,
                    2.77555756e-17],
                [-1.35361558e-01, -1.24194307e-01, -1.28407717e-01,
                    1.84162966e-02, -1.32869389e-02,  2.77555756e-17,
                    -2.08166817e-17],
                [6.37462344e-01,  7.37360525e-01,  5.99489263e-01,
                    -7.71850655e-02, -4.08633244e-02,  2.09458434e-02,
                    0.00000000e+00]
            ],
            [
                [0.00000000e+00, -6.59521910e-17, -1.31033786e-16,
                    -1.92457571e-16,  1.54134782e-17, -7.69804929e-17,
                    1.11140361e-17],
                [0.00000000e+00, -2.77555756e-17,  7.15573434e-17,
                    1.65666092e-16,  1.38777878e-17, -8.67361738e-18,
                    3.46944695e-17],
                [0.00000000e+00, -1.98669331e-01,  8.67361738e-18,
                    -1.46584134e-16,  6.02816408e-17, -3.12250226e-17,
                    6.11490025e-17],
                [0.00000000e+00, -9.54435515e-01,  4.51380881e-02,
                    1.38777878e-17,  1.08420217e-16,  3.46944695e-18,
                    6.24500451e-17],
                [0.00000000e+00, -2.95400686e-01, -1.24639152e-01,
                    -6.65899738e-01, -4.85722573e-17, -5.20417043e-18,
                    -5.55111512e-17],
                [0.00000000e+00, -9.45442009e-01,  5.96856167e-02,
                    7.19317248e-02,  6.81888149e-01, -2.77555756e-17,
                    1.04083409e-17],
                [0.00000000e+00, -2.89432165e-01, -1.18596498e-01,
                    -6.35513913e-01,  5.24032975e-03, -6.51338823e-01,
                    0.00000000e+00]
            ]
        ])

        nt.assert_array_almost_equal(panda.hessian0(q1), ans)
        nt.assert_array_almost_equal(panda.hessian0(q2), ans)
        nt.assert_array_almost_equal(panda.hessian0(q3), ans)
        nt.assert_array_almost_equal(panda.hessian0(q4), ans)
        nt.assert_array_almost_equal(panda.hessian0(J0=panda.jacob0(q1)), ans)
        nt.assert_array_almost_equal(panda.hessian0(
            q1, J0=panda.jacob0(q1)), ans)
        self.assertRaises(ValueError, panda.hessian0)
        self.assertRaises(ValueError, panda.hessian0, [1, 3])
        self.assertRaises(TypeError, panda.hessian0, 'Wfgsrth')
        self.assertRaises(
            ValueError, panda.hessian0, [1, 3], np.array([1, 5]))
        self.assertRaises(TypeError, panda.hessian0, [1, 3], 'qwe')

    def test_manipulability(self):
        panda = rp.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = 0.006559178039088341

        nt.assert_array_almost_equal(panda.manipulability(q1), ans)
        nt.assert_array_almost_equal(panda.manipulability(q2), ans)
        nt.assert_array_almost_equal(panda.manipulability(q3), ans)
        nt.assert_array_almost_equal(panda.manipulability(q4), ans)
        self.assertRaises(ValueError, panda.manipulability)
        self.assertRaises(TypeError, panda.manipulability, 'Wfgsrth')
        self.assertRaises(
            ValueError, panda.manipulability, [1, 3], np.array([1, 5]))
        self.assertRaises(TypeError, panda.manipulability, [1, 3], 'qwe')

    def test_jacobm(self):
        panda = rp.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array([
            [1.27080875e-17],
            [2.38242538e-02],
            [6.61029519e-03],
            [8.18202121e-03],
            [7.74546204e-04],
            [-1.10885380e-02],
            [0.00000000e+00]
        ])

        nt.assert_array_almost_equal(panda.jacobm(q1), ans)
        nt.assert_array_almost_equal(panda.jacobm(q2), ans)
        nt.assert_array_almost_equal(panda.jacobm(q3), ans)
        nt.assert_array_almost_equal(panda.jacobm(q4), ans)
        nt.assert_array_almost_equal(panda.jacobm(J=panda.jacob0(q1)), ans)
        self.assertRaises(ValueError, panda.jacobm)
        self.assertRaises(TypeError, panda.jacobm, 'Wfgsrth')
        self.assertRaises(ValueError, panda.jacobm, [1, 3], np.array([1, 5]))
        self.assertRaises(TypeError, panda.jacobm, [1, 3], 'qwe')
        self.assertRaises(
            TypeError, panda.jacobm, [1, 3], panda.jacob0(q1), [1, 2, 3])
        self.assertRaises(
            ValueError, panda.jacobm, [1, 3], panda.jacob0(q1),
            np.array([1, 2, 3]))


if __name__ == '__main__':

    unittest.main()
