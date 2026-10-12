import unittest
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pgraph

from roboticstoolbox.mobile.LatticePlanner import LatticeVertex
from roboticstoolbox.mobile.PlannerBase import PlannerBase
from roboticstoolbox.mobile.PoseGraph import PGVertex, PoseGraph
from roboticstoolbox.mobile.PRMPlanner import PRMPlanner


class TestPGraphCompat(unittest.TestCase):
    def test_pgraph_vertex_subclasses_use_vertex(self):
        self.assertTrue(issubclass(PGVertex, pgraph.Vertex))
        self.assertTrue(issubclass(LatticeVertex, pgraph.Vertex))

    def test_prm_plot_uses_pgraph_1_options(self):
        planner = PRMPlanner.__new__(PRMPlanner)
        planner._graph = Mock()
        ax = Mock()

        with patch.object(PlannerBase, "plot", return_value=ax):
            planner.plot(vertex={"markersize": 7}, edge={"linewidth": 2})

        planner.graph.plot.assert_called_once_with(
            labels=False,
            vertex_opt={"markersize": 7},
            edge_opt={"linewidth": 2},
            ax=ax,
        )

    def test_pose_graph_plot_forwards_public_options(self):
        pose_graph = PoseGraph.__new__(PoseGraph)
        pose_graph.graph = Mock()
        vertex_opt = {"markersize": 7}
        edge_opt = {"linewidth": 2}

        with (
            patch.object(plt, "xlabel"),
            patch.object(plt, "ylabel"),
            patch.object(plt, "grid"),
        ):
            pose_graph.plot(vopt=vertex_opt, eopt=edge_opt)

        pose_graph.graph.plot.assert_called_once_with(
            vertex_opt=vertex_opt,
            edge_opt=edge_opt,
            color_components=False,
            force_2d=True,
        )

    def test_pose_graph_optimize_uses_pgraph_1_options(self):
        pose_graph = PoseGraph.__new__(PoseGraph)
        pose_graph.graph = Mock()
        pose_graph.linearize_and_solve = Mock(return_value=0)
        ax = Mock()

        with (
            patch("roboticstoolbox.mobile.PoseGraph.smb.axes_logic", return_value=ax),
            patch.object(plt, "pause"),
        ):
            pose_graph.optimize(
                iterations=1, animate=True, retain=True, eopt={"linewidth": 2}
            )

        call = pose_graph.graph.plot.call_args
        self.assertEqual(call.kwargs["edge_opt"]["linewidth"], 2)
        self.assertIs(call.kwargs["force_2d"], True)
        self.assertIs(call.kwargs["color_components"], False)
        self.assertIs(call.kwargs["ax"], ax)
        self.assertNotIn("eopt", call.kwargs)
        self.assertNotIn("force2d", call.kwargs)
        self.assertNotIn("colorcomponents", call.kwargs)
