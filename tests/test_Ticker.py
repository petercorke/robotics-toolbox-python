#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

from roboticstoolbox.tools.Ticker import Ticker
import unittest


class TestTicker(unittest.TestCase):

    def test_ticker(self):
        self.skipTest('Not working on windows or mac')

        t = Ticker(0.1)

        t.start()
        for i in range(2):
            t.wait()

        t.stop()
