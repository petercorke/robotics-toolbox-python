# #!/usr/bin/env python3
# """
# @author: Jesse Haviland
# """

# import numpy.testing as nt
# import numpy as np
# import roboticstoolbox as rtb
# import unittest
# import spatialmath as sm
# import fknm


# class Testfknm(unittest.TestCase):

#     def test_r2q(self):

#         q = np.empty(4)

#         for _ in range(100):
#             r = sm.SE3.Rand()
#             qr = sm.base.r2q(r.R, order='xyzs')
#             fknm.r2q(r.A, q)
#             nt.assert_array_almost_equal(qr, q)

#     def test_fkine(self):
#         r = rtb.models.Panda()
#         q = np.array([1.0, 2, 3, 4, 5, 6, 7])

#         nt.assert_almost_equal(r.fkine(q).A, r.fkine(q, fast=True))

#     def test_fkine2(self):
#         r = rtb.models.Panda()
#         q = np.array([1.0, 2, 3, 4, 5, 6, 7])

#         nt.assert_almost_equal(
#             r.fkine(q, end=r.links[4]).A,
#             r.fkine(q, end=r.links[4], fast=True))

#     def test_fkine3(self):
#         r = rtb.models.Panda()
#         q = np.array([1.0, 2, 3, 4, 5, 6, 7])

#         nt.assert_almost_equal(
#             r.fkine(q, start=r.links[2], end=r.links[6]).A,
#             r.fkine(q, start=r.links[2], end=r.links[6], fast=True))

#     def test_fkine4(self):
#         r = rtb.models.Panda()
#         q = np.array([1.0, 2, 3, 4, 5, 6, 7])

#         nt.assert_almost_equal(
#             r.fkine(q, end=r.links[4], tool=sm.SE3(0.1, 0, 0)).A,
#             r.fkine(q, end=r.links[4], tool=sm.SE3(0.1, 0, 0).A, fast=True))

#     def test_jacob0(self):
#         r = rtb.models.Panda()
#         q = np.array([1.0, 2, 3, 4, 5, 6, 7])

#         nt.assert_almost_equal(r.jacob0(q), r.jacob0(q, fast=True))

#     def test_jacob0_2(self):
#         r = rtb.models.Panda()
#         q = np.array([1.0, 2, 3, 4, 5, 6, 7])

#         nt.assert_almost_equal(
#             r.jacob0(q, end=r.links[4]),
#             r.jacob0(q, end=r.links[4], fast=True))

#     def test_jacob0_3(self):
#         r = rtb.models.Panda()
#         q = np.array([1.0, 2, 3, 4, 5, 6, 7])

#         nt.assert_almost_equal(
#             r.jacob0(q, start=r.links[2], end=r.links[4]),
#             r.jacob0(q, start=r.links[2], end=r.links[4], fast=True))

#     def test_jacob0_4(self):
#         r = rtb.models.Panda()
#         q = np.array([1.0, 2, 3, 4, 5, 6, 7])

#         nt.assert_almost_equal(
#             r.jacob0(q, end=r.links[4], tool=sm.SE3(0.1, 0, 0)),
#             r.jacob0(q, end=r.links[4], tool=sm.SE3(0.1, 0, 0).A, fast=True))
