"""Scan matching ignores the no-return beams retained by scanxy as NaNs."""

from pathlib import Path

import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE2

from roboticstoolbox import PoseGraph


def lidar_graph(
    tmp_path: Path, reference: np.ndarray, source: np.ndarray, angle: float
) -> PoseGraph:
    records = []
    for i, ranges in enumerate((reference, source)):
        n = len(ranges)
        # Zero remissions, six pose values, five velocity/safety values, then
        # timestamp/host/logger time complete each g2o RobotLaser record.
        record = (
            f"ROBOTLASER1 0 {-1.0 - i * angle} {0.3 * (n - 1)} 0.3 10 0.01 0 {n} "
            + " ".join(map(str, ranges))
            + " "
            + "0 " * 12
            + f"{100 + i} sensor {101 + i}\n"
        )
        records.append(f"VERTEX_SE2 {i} 0 0 0\n{record}")
    path = tmp_path / "scans.g2o"
    path.write_text("".join(records))
    return PoseGraph(str(path), lidar=True)


@pytest.mark.parametrize("angle", [0.0, 0.08])
@pytest.mark.parametrize("affected", ["reference", "source", "both"])
@pytest.mark.parametrize("no_return", [10.0, np.inf, np.nan])
def test_matching_skips_invalid_beams_in_either_scan(
    tmp_path: Path, angle: float, affected: str, no_return: float
) -> None:
    reference = np.array([1, 1.4, 2, 2.5, 2.4, 1.2, 1.8])
    source = reference.copy()
    if affected in ("reference", "both"):
        reference[3] = no_return
    if affected in ("source", "both"):
        source[3] = no_return
    graph = lidar_graph(tmp_path, reference, source, angle)
    before = [graph.scanxy(i) for i in (0, 1)]

    # The second scan's bearings differ by exactly -angle. Match it back to
    # the first with the real ICP implementation, without mocking its input.
    nt.assert_allclose(graph.scanmatch(0, 1).A, SE2.Rot(angle).A, atol=1e-10)
    for i in (0, 1):
        nt.assert_allclose(graph.scanxy(i), before[i], equal_nan=True)


@pytest.mark.parametrize("angle", [0.0, 0.08])
def test_all_finite_scans_keep_their_matching_result(
    tmp_path: Path, angle: float
) -> None:
    ranges = np.array([1, 1.4, 2, 2.5, 2.4, 1.2, 1.8])
    graph = lidar_graph(tmp_path, ranges, ranges, angle)
    nt.assert_allclose(graph.scanmatch(0, 1).A, SE2.Rot(angle).A, atol=1e-10)


@pytest.mark.parametrize("empty_scan", [0, 1])
def test_scan_without_finite_returns_has_a_clear_error(
    tmp_path: Path, empty_scan: int
) -> None:
    scans = [np.array([1, 1.4, 2, 2.5, 2.4, 1.2, 1.8]) for _ in (0, 1)]
    scans[empty_scan][:] = 10
    graph = lidar_graph(tmp_path, *scans, 0.0)

    with pytest.raises(ValueError, match=f"scan {empty_scan} has no finite points"):
        graph.scanmatch(0, 1)
