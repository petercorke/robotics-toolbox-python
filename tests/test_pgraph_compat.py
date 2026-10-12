from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pgraph

from roboticstoolbox.mobile.LatticePlanner import LatticeVertex
from roboticstoolbox.mobile.PlannerBase import PlannerBase
from roboticstoolbox.mobile.PoseGraph import PGVertex, PoseGraph
from roboticstoolbox.mobile.PRMPlanner import PRMPlanner


def test_pgraph_vertex_subclasses_use_vertex():
    assert issubclass(PGVertex, pgraph.Vertex)
    assert issubclass(LatticeVertex, pgraph.Vertex)


def test_prm_plot_uses_pgraph_1_options():
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


def test_pose_graph_plot_forwards_public_options():
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


def test_pose_graph_optimize_uses_pgraph_1_options():
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
    assert call.kwargs["edge_opt"]["linewidth"] == 2
    assert call.kwargs["force_2d"] is True
    assert call.kwargs["color_components"] is False
    assert call.kwargs["ax"] is ax
    assert "eopt" not in call.kwargs
    assert "force2d" not in call.kwargs
    assert "colorcomponents" not in call.kwargs
