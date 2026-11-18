"""Lidar fields belong to their own ROBOTLASER1 record."""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from roboticstoolbox import PoseGraph


def laser_record(
    ranges: list[float],
    timestamp: float,
    *,
    maxrange: float = 10,
    angle: float = -1,
    step: float = 0.5,
    remissions: tuple[float, ...] = (),
) -> str:
    # g2o RobotLaser: configuration, ranges, remissions, two poses,
    # velocities/safety fields, acquisition timestamp, host, logger timestamp.
    fields = [
        "ROBOTLASER1",
        0,
        angle,
        step * (len(ranges) - 1),
        step,
        maxrange,
        0.01,
        int(bool(remissions)),
        len(ranges),
        *ranges,
        len(remissions),
        *remissions,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        timestamp,
        "sensor",
        timestamp + 1,
    ]
    return " ".join(map(str, fields)) + "\n"


def load_records(tmp_path: Path, records: list[str], zipped: bool = False) -> PoseGraph:
    contents = "".join(
        f"VERTEX_SE2 {i} {i} 0 0\n{record}" for i, record in enumerate(records)
    )
    path = tmp_path / "records.g2o"
    if zipped:
        path = path.with_suffix(".g2o.zip")
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("records.g2o", contents)
    else:
        path.write_text(contents)
    return PoseGraph(str(path), lidar=True)


@pytest.mark.parametrize("zipped", [False, True])
@pytest.mark.parametrize("counts", [(2, 4), (4, 2), (2, 0)])
def test_each_scan_uses_its_declared_beam_count(
    tmp_path: Path, zipped: bool, counts: tuple[int, int]
) -> None:
    ranges = [[float(i + 1) for i in range(n)] for n in counts]
    graph = load_records(
        tmp_path,
        [laser_record(r, 100 + i, angle=0.1 * i) for i, r in enumerate(ranges)],
        zipped,
    )

    for i, expected in enumerate(ranges):
        values, angles = graph.scan(i)
        np.testing.assert_array_equal(values, expected)
        np.testing.assert_allclose(angles, 0.1 * i + 0.5 * np.arange(len(expected)))
        assert graph.scanxy(i).shape == (2, len(expected))
        assert graph.time(i) == 100 + i


@pytest.mark.parametrize("remissions", [(), (0.2, 0.4, 0.6)])
def test_timestamp_follows_the_remission_array(
    tmp_path: Path, remissions: tuple[float, ...]
) -> None:
    graph = load_records(
        tmp_path, [laser_record([1, 2, 3], 123.5, remissions=remissions)]
    )

    np.testing.assert_array_equal(graph.scan(0)[0], [1, 2, 3])
    assert graph.time(0) == 123.5


@pytest.mark.parametrize("limits", [(5, 10), (10, 5)])
def test_scanxy_uses_the_maximum_range_from_its_record(
    tmp_path: Path, limits: tuple[int, int]
) -> None:
    graph = load_records(
        tmp_path,
        [laser_record([4, 5, 7, 10], i, maxrange=m) for i, m in enumerate(limits)],
    )

    for i, maximum in enumerate(limits):
        ranges, angles = graph.scan(i)
        expected = np.array([ranges * np.cos(angles), ranges * np.sin(angles)])
        expected[:, ranges >= maximum] = np.nan
        np.testing.assert_allclose(graph.scanxy(i), expected, equal_nan=True)


def test_uniform_records_keep_the_existing_results(tmp_path: Path) -> None:
    graph = load_records(
        tmp_path, [laser_record([3, 4], 100), laser_record([3, 4], 101)]
    )

    for i in range(2):
        np.testing.assert_array_equal(graph.scan(i)[0], [3, 4])
        np.testing.assert_array_equal(graph.scan(i)[1], [-1, -0.5])
        assert graph.time(i) == 100 + i
