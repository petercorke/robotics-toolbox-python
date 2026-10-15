"""
Tests of PoseGraph: access to poses and scans, the removed unfinished grid
conversions, load errors, and the plot style arguments.
"""

import contextlib
import io
import tempfile
import unittest
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from roboticstoolbox import BinaryOccupancyGrid, PoseGraph
from roboticstoolbox.mobile.PoseGraph import PGVertex


def quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def write(text, suffix=".g2o"):
    path = Path(tempfile.mkdtemp()) / f"graph{suffix}"
    path.write_text(text)
    return str(path)


class TestAccess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg = quiet(PoseGraph, "data/pg1.g2o")

    def test_pose_without_lidar(self):
        # vindex used to be filled only when there was lidar data
        nt.assert_allclose(self.pg.pose(0), [0, 0, 0], atol=1e-12)
        nt.assert_allclose(self.pg.pose(1), [11, 2, 0.1])

    def test_pose_is_the_vertex_coordinate(self):
        for i, v in enumerate(self.pg.graph):
            self.assertIs(self.pg.pose(i), v.coord)

    def test_unknown_vertex(self):
        with self.assertRaises(IndexError):
            self.pg.pose(99)

    def test_time_needs_lidar(self):
        with self.assertRaises(ValueError):
            self.pg.time(0)

    def test_scan_needs_lidar(self):
        for method in (self.pg.scan, self.pg.scanxy):
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError) as cm:
                    method(0)
                self.assertIn("lidar", str(cm.exception))


class TestLidar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg = quiet(PoseGraph, "data/killian.g2o.zip", lidar=True)

    def tearDown(self):
        plt.close("all")

    def test_scan(self):
        r, theta = self.pg.scan(0)
        self.assertEqual(r.shape, theta.shape)
        self.assertEqual(self.pg.scanxy(0).shape, (2, len(r)))
        self.assertGreater(self.pg.time(0), 0)

    def test_plot_scan_marker(self):
        # the arguments used to be (x, y, ".", "MarkerSize", 10)
        import unittest.mock as mock

        with mock.patch.object(plt, "pause"):
            self.pg.plot_scan(0)
        (line,) = plt.gca().lines
        self.assertEqual(line.get_markersize(), 10)
        self.assertEqual(line.get_linestyle(), "None")

    def test_scanmap_step(self):
        occgrid = BinaryOccupancyGrid(workspace=[-60, 60, -60, 60], cellsize=0.5)
        quiet(self.pg.scanmap, occgrid, maxrange=40, step=1000)
        self.assertTrue((occgrid.grid != 0).any())


class TestRemoved(unittest.TestCase):
    def test_unfinished_grid_conversions_are_gone(self):
        # they read attributes that were never set, so always raised
        for name in ("w2g", "g2w", "plot_occgrid"):
            with self.subTest(name=name):
                self.assertFalse(hasattr(PoseGraph, name))

    def test_vertex_counter_is_not_global_state(self):
        self.assertFalse(hasattr(PGVertex, "nvertices"))

    def test_indices_belong_to_the_graph(self):
        a = quiet(PoseGraph, "data/pg1.g2o")
        b = quiet(PoseGraph, "data/killian-small.toro")
        self.assertEqual([v.index for v in a.graph], [0, 1, 2, 3])
        self.assertEqual([v.index for v in b.graph][:3], [0, 1, 2])


class TestLoad(unittest.TestCase):
    def test_lidar_record_before_any_vertex(self):
        text = "ROBOTLASER1 0 -1.5 3 0.01 80 0 0 2 0.5 0.6 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        path = write(text)
        with self.assertRaises(ValueError) as cm:
            quiet(PoseGraph, path, lidar=True)
        self.assertIn("line 1", str(cm.exception))
        self.assertIn(path, str(cm.exception))


class TestVerbose(unittest.TestCase):
    @staticmethod
    def printed(f, *args, **kwargs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = f(*args, **kwargs)
        return result, out.getvalue()

    def test_silent_by_default(self):
        pg, text = self.printed(PoseGraph, "data/pg1.g2o")
        self.assertEqual(text, "")
        _, text = self.printed(pg.linearize_and_solve)
        self.assertEqual(text, "")
        _, text = self.printed(pg.optimize, iterations=2)
        self.assertEqual(text, "")

    def test_verbose_load(self):
        pg, text = self.printed(PoseGraph, "data/pg1.g2o", verbose=True)
        self.assertIn("loaded g2o format file: 4 vertices, 4 edges", text)

    def test_verbose_lidar_summary(self):
        _, text = self.printed(
            PoseGraph, "data/killian.g2o.zip", lidar=True, verbose=True
        )
        self.assertIn("lidar scans", text)

    def test_verbose_solve(self):
        pg, _ = self.printed(PoseGraph, "data/pg1.g2o", verbose=True)
        _, text = self.printed(pg.linearize_and_solve)
        self.assertIn("Total cost", text)

    def test_verbose_attribute(self):
        pg, _ = self.printed(PoseGraph, "data/pg1.g2o")
        self.assertFalse(pg.verbose)
        pg.verbose = True
        _, text = self.printed(pg.linearize_and_solve)
        self.assertIn("Total cost", text)


class TestOptimize(unittest.TestCase):
    def test_landmarks_are_not_supported(self):
        path = write(
            "VERTEX_SE2 1 0 0 0\nVERTEX_SE2 2 1 0 0\nVERTEX_XY 3 2 2\n"
            "EDGE_SE2 1 2 1 0 0 500 0 0 500 0 500\n"
        )
        pg = quiet(PoseGraph, path)
        with self.assertRaises(NotImplementedError):
            quiet(pg.optimize, iterations=1)


class TestOptimizeStyles(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_animate_styles(self):
        import unittest.mock as mock

        pg = quiet(PoseGraph, "data/pg1.g2o")
        with mock.patch.object(pg.graph, "plot") as graph_plot:
            with mock.patch.object(plt, "pause"):
                quiet(
                    pg.optimize,
                    iterations=2,
                    animate=True,
                    vertex_style={"markersize": 2},
                    edge_style={"linewidth": 3},
                )
        for call in graph_plot.call_args_list:
            self.assertEqual(call.kwargs["vertex_opt"], {"markersize": 2})
            self.assertEqual(call.kwargs["edge_opt"]["linewidth"], 3)

    def test_deprecated_names(self):
        import unittest.mock as mock

        pg = quiet(PoseGraph, "data/pg1.g2o")
        with mock.patch.object(pg.graph, "plot"):
            with mock.patch.object(plt, "pause"):
                with self.assertWarns(DeprecationWarning):
                    quiet(
                        pg.optimize, iterations=1, animate=True, eopt={"linewidth": 3}
                    )


class TestNoPgraphDeprecations(unittest.TestCase):
    """Our own calls to pgraph use its current option names"""

    def tearDown(self):
        plt.close("all")

    def test_plot_and_optimize(self):
        import unittest.mock as mock

        pg = quiet(PoseGraph, "data/pg1.g2o")
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=DeprecationWarning)
            pg.plot()
            with mock.patch.object(plt, "pause"):
                quiet(pg.optimize, iterations=2, animate=True)

    def test_text_option_is_forwarded(self):
        import unittest.mock as mock

        pg = quiet(PoseGraph, "data/pg1.g2o")
        with mock.patch.object(pg.graph, "plot") as graph_plot:
            pg.plot(text=False)
        self.assertIs(graph_plot.call_args.kwargs["labels"], False)
        self.assertNotIn("text", graph_plot.call_args.kwargs)

    def test_prm_plot(self):
        from roboticstoolbox import PRMPlanner

        prm = PRMPlanner(np.zeros((20, 20)), seed=0, progress=False)
        prm.plan(npoints=20)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=DeprecationWarning)
            prm.plot()


class TestPlotStyles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pg = quiet(PoseGraph, "data/pg1.g2o")

    def setUp(self):
        import unittest.mock as mock

        self.patcher = mock.patch.object(self.pg.graph, "plot")
        self.graph_plot = self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        plt.close("all")

    def passed(self):
        return self.graph_plot.call_args.kwargs

    def test_defaults(self):
        self.pg.plot()
        self.assertEqual(self.passed()["vertex_opt"]["markersize"], 8)
        self.assertEqual(self.passed()["edge_opt"]["linewidth"], 1)

    def test_styles_merge_over_defaults(self):
        self.pg.plot(vertex_style={"markersize": 3}, edge_style={"color": "r"})
        self.assertEqual(self.passed()["vertex_opt"]["markersize"], 3)
        self.assertEqual(self.passed()["vertex_opt"]["markerfacecolor"], "blue")
        self.assertEqual(self.passed()["edge_opt"]["linewidth"], 1)
        self.assertEqual(self.passed()["edge_opt"]["color"], "r")

    def test_style_dicts_are_not_modified(self):
        style = {"markersize": 3}
        self.pg.plot(vertex_style=style)
        self.assertEqual(style, {"markersize": 3})
        self.pg.plot()
        self.assertEqual(self.passed()["vertex_opt"]["markersize"], 8)

    def test_deprecated_names(self):
        with self.assertWarns(DeprecationWarning):
            self.pg.plot(vopt={"markersize": 2})
        self.assertEqual(self.passed()["vertex_opt"]["markersize"], 2)
        with self.assertWarns(DeprecationWarning):
            self.pg.plot(eopt={"linewidth": 4})
        self.assertEqual(self.passed()["edge_opt"]["linewidth"], 4)


if __name__ == "__main__":
    unittest.main()
