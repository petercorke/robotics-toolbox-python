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

    def test_plot_movie(self):
        # robot.plot(..., movie=...) used to crash outright: BaseRobot.plot()
        # referenced PyPlot in an isinstance check with no import anywhere in
        # the module (NameError), and getframe() called the long-removed
        # matplotlib Agg canvas method tostring_rgb() (AttributeError on
        # modern matplotlib). Covers both bugs end-to-end via a real saved
        # GIF, not just "no exception raised".
        import tempfile
        import os
        from PIL import Image

        panda = rp.models.Panda()
        qt = rp.jtraj(panda.qr, panda.qz, 3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "movie.gif")
            env = panda.plot(qt.q, backend="pyplot", movie=path)
            env.close()

            self.assertTrue(os.path.exists(path))
            img = Image.open(path)
            n_frames = 0
            try:
                while True:
                    img.seek(n_frames)
                    n_frames += 1
            except EOFError:
                pass
            self.assertEqual(n_frames, 3)


if __name__ == "__main__":
    unittest.main()
