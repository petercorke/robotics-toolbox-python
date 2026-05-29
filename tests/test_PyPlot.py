#!/usr/bin/env python
"""
@author: Jesse Haviland
"""

# import numpy.testing as nt
# import numpy as np
import roboticstoolbox as rp
import matplotlib.pyplot as plt

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

    def test_launch_with_external_3d_axes(self):
        from roboticstoolbox.backends.PyPlot import PyPlot

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        env = PyPlot()
        env.launch(fig=fig, ax=ax)

        self.assertIs(env.fig, fig)
        self.assertIs(env.ax, ax)
        env.close()

    def test_launch_rejects_2d_axes(self):
        from roboticstoolbox.backends.PyPlot import PyPlot

        fig = plt.figure()
        ax = fig.add_subplot(111)

        env = PyPlot()
        with self.assertRaises(ValueError):
            env.launch(fig=fig, ax=ax)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
