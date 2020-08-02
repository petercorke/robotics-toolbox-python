#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import ropy as rp
import spatialmath as sm
import unittest


class TestModels(unittest.TestCase):

    def test_puma(self):
        puma = rp.Puma560()
        puma.qr
        puma.qz
        puma.qs
        puma.qn

    def test_frankie(self):
        frankie = rp.Frankie()
        frankie.qr
        frankie.qz
