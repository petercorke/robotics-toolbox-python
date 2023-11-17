#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

from roboticstoolbox.tools.Ticker import Ticker
import unittest
import platform


class TestTicker(unittest.TestCase):

    def test_ticker(self):

        if platform.system() in ['Windows', 'Darwin']:
            self.skipTest('Not working on windows or mac')

        t = Ticker(0.1)

        t.start()
        for i in range(2):
            t.wait()

        t.stop()
