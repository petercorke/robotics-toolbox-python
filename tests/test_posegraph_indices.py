"""Check that independently loaded pose graphs have independent vertex indices."""

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import PoseGraph


@pytest.mark.parametrize("format_name", ["g2o", "toro"])
@pytest.mark.parametrize("interleaved", [False, True])
def test_graph_indices_are_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, format_name: str, interleaved: bool
) -> None:
    """Interleaving another loader must not change indices or optimisation results."""
    vertex, edge = (
        ("VERTEX_SE2", "EDGE_SE2") if format_name == "g2o" else ("VERTEX2", "EDGE2")
    )
    info = "1 0 0 1 0 1" if format_name == "g2o" else "1 0 1 1 0 0"
    path = tmp_path / f"outer.{format_name}"
    path.write_text(
        f"{vertex} 10 0 0 0\n{vertex} 20 1.2 0.1 0.1\n{edge} 10 20 1 0 0 {info}\n"
    )
    inner_path = tmp_path / "inner.g2o"
    inner_path.write_text(
        "VERTEX_SE2 1 0 0 0\nVERTEX_SE2 2 1 0 0\nVERTEX_SE2 3 2 0 0\n"
    )
    reference = PoseGraph(str(path))
    reference.linearize_and_solve()
    module = importlib.import_module("roboticstoolbox.mobile.PoseGraph")
    original_init = module.PGVertex.__init__
    nested_graphs = []
    triggered = False

    # Force a repeatable interleaving without relying on thread scheduling.
    def interleave_vertex_init(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal triggered
        original_init(self, *args, **kwargs)
        if not triggered:
            triggered = True
            nested_graphs.append(PoseGraph(str(inner_path)))

    if interleaved:
        monkeypatch.setattr(module.PGVertex, "__init__", interleave_vertex_init)
    graph = PoseGraph(str(path))
    nt.assert_equal([v.index for v in graph.graph], [0, 1])
    for nested in nested_graphs:
        nt.assert_equal([v.index for v in nested.graph], [0, 1, 2])
    graph.linearize_and_solve()
    nt.assert_allclose(
        np.array([v.coord for v in graph.graph]),
        np.array([v.coord for v in reference.graph]),
    )


@pytest.mark.parametrize("with_landmark", [False, True])
def test_interleaved_lidar_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_landmark: bool
) -> None:
    """Laser scans remain attached to their graph-local vertex numbers."""
    scan = "ROBOTLASER1 0 -1 2 1 10 0 0 2 3 4 " + "0 " * 12 + "123\n"
    path = tmp_path / "scans.g2o"
    path.write_text(
        "VERTEX_SE2 10 0 0 0\n"
        + scan
        + ("VERTEX_XY 15 0.5 0.5\n" if with_landmark else "")
        + "VERTEX_SE2 20 1 0 0\n"
        + scan
    )
    inner_path = tmp_path / "inner.g2o"
    inner_path.write_text(
        "VERTEX_SE2 1 0 0 0\nVERTEX_SE2 2 1 0 0\nVERTEX_SE2 3 2 0 0\n"
    )
    module = importlib.import_module("roboticstoolbox.mobile.PoseGraph")
    original_init = module.PGVertex.__init__
    triggered = False

    # Force a repeatable interleaving without relying on thread scheduling.
    def interleave_vertex_init(self: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal triggered
        original_init(self, *args, **kwargs)
        if not triggered:
            triggered = True
            PoseGraph(str(inner_path))

    monkeypatch.setattr(module.PGVertex, "__init__", interleave_vertex_init)
    graph = PoseGraph(str(path), lidar=True)
    last_index = 2 if with_landmark else 1
    nt.assert_equal(sorted(graph.vindex), [0, last_index])
    nt.assert_equal([v.index for v in graph.graph], list(range(last_index + 1)))
    nt.assert_allclose(graph.pose(last_index), [1, 0, 0])
    nt.assert_equal(graph.time(last_index), 123)
    ranges, angles = graph.scan(last_index)
    nt.assert_allclose(ranges, [3, 4])
    nt.assert_allclose(angles, [-1, 0])
