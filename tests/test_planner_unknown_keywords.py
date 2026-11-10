"""
Unknown keyword arguments to the planners' constructor and plot methods.

RTB 1.4.4 silently swallowed them (``**unused``), RTB 1.5.0 first made them an
error, which broke code printed in the book, for example
``dx.plot(nonsense=True)``.  They are ignored with a ``DeprecationWarning``.
"""

import unittest
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboticstoolbox import DistanceTransformPlanner


def grid():
    return np.zeros((8, 8), dtype=int)


class TestUnknownKeywords(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_plot_warns_and_ignores(self):
        dx = DistanceTransformPlanner(grid(), goal=(6, 6))
        dx.plan()
        with self.assertWarns(DeprecationWarning) as cm:
            dx.plot(nonsense=True)
        self.assertIn("nonsense", str(cm.warning))
        # and it did plot
        self.assertGreater(len(plt.gcf().axes), 0)

    def test_plot_names_every_keyword(self):
        dx = DistanceTransformPlanner(grid(), goal=(6, 6))
        dx.plan()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dx.plot(nonsense=True, colorbar_size=3)
        text = " ".join(str(x.message) for x in w)
        self.assertIn("nonsense", text)
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
            dx.plot_bg(nonsense=True)

    def test_warning_points_at_the_caller(self):
        dx = DistanceTransformPlanner(grid(), goal=(6, 6))
        dx.plan()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dx.plot(nonsense=True)
        self.assertTrue(any(x.filename == __file__ for x in w))


class TestLabelValues(unittest.TestCase):
    """dx.plot(labels=True), in the book's chapter 5 notebook"""

    def setUp(self):
        # the 6x6 example from the book
        g = np.zeros((6, 6), dtype=int)
        g[2:5, 3:5] = 1
        g[3:5, 2] = 1
        self.grid = g
        self.dx = DistanceTransformPlanner(g, metric="manhattan")
        self.dx.plan(goal=(1, 1))

    def tearDown(self):
        plt.close("all")

    def texts(self):
        return {
            tuple(map(int, t.get_position())): t.get_text() for t in plt.gca().texts
        }

    def test_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            self.dx.plot(labels=True)

    def test_labels_is_the_argument(self):
        self.dx.plot(labels=True)
        self.assertEqual(
            len(plt.gca().texts), len(np.argwhere(np.isfinite(self.dx.distancemap)))
        )

    def test_labelvalues_is_deprecated(self):
        # the book's name, which never did anything
        with self.assertWarns(DeprecationWarning) as cm:
            self.dx.plot(labelvalues=True)
        self.assertIn("labels", str(cm.warning))
        a = {t.get_position(): t.get_text() for t in plt.gca().texts}
        plt.close("all")
        self.dx.plot(labels=True)
        b = {t.get_position(): t.get_text() for t in plt.gca().texts}
        self.assertEqual(a, b)

    def test_labelvalues_warning_points_at_the_caller(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.dx.plot(labelvalues=True)
        self.assertTrue(any(x.filename == __file__ for x in w))

    def test_plot_bg_labelvalues_is_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            self.dx.plot_bg(labelvalues=True)

    def test_plot_bg_labels(self):
        self.dx.plot_bg(labels=True)
        self.assertGreater(len(plt.gca().texts), 0)

    def test_default_has_no_labels(self):
        self.dx.plot()
        self.assertEqual(len(plt.gca().texts), 0)

    def test_every_free_cell_is_labelled_with_its_distance(self):
        self.dx.plot(labels=True)
        texts = self.texts()
        d = self.dx.distancemap
        free = np.argwhere(np.isfinite(d))
        self.assertEqual(len(texts), len(free))
        for r, c in free:
            self.assertEqual(texts[(c, r)], str(int(d[r, c])))
        self.assertEqual(texts[(1, 1)], "0")

    def test_obstacles_are_not_labelled(self):
        self.dx.plot(labels=True)
        texts = self.texts()
        for r, c in np.argwhere(self.grid > 0):
            self.assertNotIn((c, r), texts)

    def test_euclidean_values_have_one_decimal(self):
        dx = DistanceTransformPlanner(self.grid, metric="euclidean")
        dx.plan(goal=(1, 1))
        dx.plot(labels=True)
        self.assertEqual(self.texts()[(2, 2)], "1.4")

    def test_default_labels_are_red(self):
        # the book says the cell labels are red
        self.dx.plot(labels=True)
        self.assertEqual({t.get_color() for t in plt.gca().texts}, {"red"})

    def test_default_labels_have_a_white_outline(self):
        from matplotlib import patheffects

        self.dx.plot(labels=True)
        for text in plt.gca().texts:
            (effect,) = text.get_path_effects()
            self.assertIsInstance(effect, patheffects.withStroke)
            self.assertEqual(effect._gc["foreground"], "white")

    def test_label_style_can_remove_the_outline(self):
        self.dx.plot(labels=True, label_style=dict(path_effects=[]))
        self.assertEqual(plt.gca().texts[0].get_path_effects(), [])

    def test_label_style(self):
        self.dx.plot(labels=True, label_style=dict(fontsize=14, color="m"))
        text = plt.gca().texts[0]
        self.assertEqual(text.get_fontsize(), 14)
        self.assertEqual(text.get_color(), "m")

    def test_without_a_distance_field(self):
        dx = DistanceTransformPlanner(self.grid, goal=(1, 1))
        with self.assertWarns(UserWarning):
            dx.plot(labels=True)


class TestBackgroundArguments(unittest.TestCase):
    """colorbar, colorbar_style, inflated and cmap are arguments of plot()"""

    def tearDown(self):
        plt.close("all")

    def planner(self):
        dx = DistanceTransformPlanner(grid(), goal=(6, 6), inflate=1)
        dx.plan()
        return dx

    def test_colorbar_style_reaches_the_colorbar(self):
        self.planner().plot(colorbar_style=dict(label="cost"))
        labels = [a.get_ylabel() for a in plt.gcf().axes]
        self.assertIn("cost", labels)

    def test_colorbar_false(self):
        self.planner().plot(colorbar=False)
        self.assertEqual(len(plt.gcf().axes), 1)

    def test_colorbar_default(self):
        self.planner().plot()
        self.assertEqual(len(plt.gcf().axes), 2)

    def test_cmap(self):
        self.planner().plot(cmap="viridis")
        self.assertEqual(plt.gca().images[0].get_cmap().name, "viridis")

    def test_inflated(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            self.planner().plot(inflated=False)

    def test_bgargs_still_works(self):
        self.planner().plot(bgargs=dict(colorbar=False))
        self.assertEqual(len(plt.gcf().axes), 1)

    def test_explicit_argument_does_not_override_bgargs(self):
        self.planner().plot(colorbar=True, bgargs=dict(colorbar=False))
        self.assertEqual(len(plt.gcf().axes), 1)


class TestDistanceIsDeprecated(unittest.TestCase):
    """The book's notebook gives distance="manhattan", which was ignored"""

    def test_distance_is_metric_with_a_warning(self):
        with self.assertWarns(DeprecationWarning) as cm:
            dx = DistanceTransformPlanner(grid(), distance="manhattan")
        self.assertIn("metric", str(cm.warning))
        self.assertEqual(dx.metric, "manhattan")

    def test_warning_points_at_the_caller(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            DistanceTransformPlanner(grid(), distance="manhattan")
        self.assertTrue(any(x.filename == __file__ for x in w))

    def test_metric_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            dx = DistanceTransformPlanner(grid(), metric="manhattan")
        self.assertEqual(dx.metric, "manhattan")

    def test_default_is_unchanged(self):
        self.assertEqual(DistanceTransformPlanner(grid()).metric, "euclidean")

    def test_conflict(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaises(ValueError):
                DistanceTransformPlanner(
                    grid(), metric="manhattan", distance="euclidean"
                )
            # the same thing twice is fine
            DistanceTransformPlanner(grid(), metric="manhattan", distance="manhattan")

    def test_bad_value(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaises(ValueError):
                DistanceTransformPlanner(grid(), distance="chebyshev")


if __name__ == "__main__":
    unittest.main()
