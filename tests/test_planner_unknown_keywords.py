"""
Unknown keyword arguments to the planners' constructor and plot methods.

RTB 1.4.4 silently swallowed them (``**unused``), RTB 1.4.5 first made them an
error, which broke code printed in the book, for example
``dx.plot(labelvalues=True)``.  They are ignored with a ``DeprecationWarning``.
"""

import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboticstoolbox import Bug2, DistanceTransformPlanner, PRMPlanner


def grid():
    return np.zeros((8, 8), dtype=int)


class TestUnknownKeywords(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_warns_and_ignores(self):
        dx = DistanceTransformPlanner(grid(), goal=(6, 6))
        dx.plan()
        with self.assertWarns(DeprecationWarning) as cm:
            dx.plot(labelvalues=True)
        self.assertIn("labelvalues", str(cm.warning))
        # and it did plot
        self.assertGreater(len(plt.gcf().axes), 0)

    def test_plot_names_every_keyword(self):
        dx = DistanceTransformPlanner(grid(), goal=(6, 6))
        dx.plan()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dx.plot(labelvalues=True, colorbar_size=3)
        text = " ".join(str(x.message) for x in w)
        self.assertIn("labelvalues", text)
        self.assertIn("colorbar_size", text)

    def test_known_keywords_do_not_warn(self):
        dx = DistanceTransformPlanner(grid(), goal=(6, 6))
        dx.plan()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            dx.plot(path_style=dict(color="r"), bgargs=dict(inflated=True))

    def test_constructor_warns_and_ignores(self):
        with self.assertWarns(DeprecationWarning) as cm:
            dx = DistanceTransformPlanner(grid(), goal=(6, 6), nonsense=1)
        self.assertIn("nonsense", str(cm.warning))
        dx.plan()

    def test_plot_bg_warns_and_ignores(self):
        dx = DistanceTransformPlanner(grid(), goal=(6, 6))
        dx.plan()
        with self.assertWarns(DeprecationWarning):
            dx.plot_bg(labelvalues=True)

    def test_warning_points_at_the_caller(self):
        dx = DistanceTransformPlanner(grid(), goal=(6, 6))
        dx.plan()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dx.plot(labelvalues=True)
        self.assertTrue(any(x.filename == __file__ for x in w))


if __name__ == "__main__":
    unittest.main()
