#!/usr/bin/env python3
"""
@author: Jesse Haviland
"""

import unittest
import sys
sys.path.append('./examples')


class TestExamples(unittest.TestCase):

    def test_RRMC(self):
        import RRMC


if __name__ == '__main__':

    unittest.main()
