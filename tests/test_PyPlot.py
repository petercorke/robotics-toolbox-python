#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

# import numpy.testing as nt
# import numpy as np
import roboticstoolbox as rp

# import spatialmath as sm
import unittest


class TestPyPlot(unittest.TestCase):
    def test_PyPlot(self):
        panda = rp.models.DH.Panda()
        from roboticstoolbox.backends.PyPlot import PyPlot

        env = PyPlot()
        env.launch()
        env.add(panda)
        env.step()
        # env._plot_handler(None, None)
        env.close()

    def test_PyPlot_invisible(self):
        panda = rp.models.DH.Panda()
        from roboticstoolbox.backends.PyPlot import PyPlot

        env = PyPlot()
        env.launch()
        env.add(panda, display=False)
        env.step()
        # env._plot_handler(None, None)
        env.close()

    def test_unimplemented(self):
        # TODO remove these as implemented
        from roboticstoolbox.backends.PyPlot import PyPlot

        env = PyPlot()
        env.reset()

        env.restart()
        env.remove(0)


if __name__ == "__main__":

    unittest.main()
