"""Headless regression tests for range-bearing sensor visualization."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from roboticstoolbox import Bicycle, LandmarkMap, RangeBearingSensor


def test_sensor_beam_and_sector_move_without_duplicate_artists() -> None:
    robot = Bicycle(x0=[0.0, 0.0, 0.0])
    landmarks = LandmarkMap(np.array([[2.0], [0.0]]))
    sensor = RangeBearingSensor(
        robot,
        landmarks,
        range=(0.5, 3.0),
        angle=np.pi / 4,
        plot=True,
        line_style={"color": "blue", "linewidth": 2},
        poly_style={"facecolor": "green", "alpha": 0.25},
    )
    try:
        z, idx = sensor.reading()
        assert idx == 0
        assert z is not None
        ax = sensor._plot_ax
        line, poly = sensor._plot_line, sensor._plot_poly
        assert ax is not None
        assert line in ax.lines
        assert poly in ax.patches
        assert line.get_color() == "blue"
        assert line.get_linewidth() == 2
        assert poly.get_alpha() == 0.25
        original_vertices = poly.get_xy().copy()
        nt.assert_allclose(line.get_xdata(), [0, 2])
        nt.assert_allclose(line.get_ydata(), [0, 0])

        robot._x = np.array([0.4, -0.2, 0.3])
        sensor.plot(0)
        assert sensor._plot_line is line
        assert sensor._plot_poly is poly
        assert len(ax.patches) == 1
        assert line.get_visible()
        nt.assert_allclose(line.get_xdata(), [0.4, 2.0])
        nt.assert_allclose(line.get_ydata(), [-0.2, 0.0])
        assert not np.allclose(poly.get_xy(), original_vertices)

        sensor.plot(None)
        assert not line.get_visible()
        assert len(ax.patches) == 1
    finally:
        plt.close("all")


def test_no_observation_hides_beam_and_keeps_sector() -> None:
    robot = Bicycle(x0=[0.0, 0.0, 0.0])
    landmarks = LandmarkMap(np.array([[2.0], [0.0]]))
    sensor = RangeBearingSensor(
        robot, landmarks, range=3.0, angle=np.pi / 4, plot=True, every=2
    )
    try:
        assert sensor.reading() == (None, None)
        assert sensor._plot_line is not None
        assert not sensor._plot_line.get_visible()
        assert sensor._plot_poly is not None
        z, idx = sensor.reading()
        assert z is not None and idx == 0
        assert sensor._plot_line.get_visible()
        robot._x = np.array([0.0, 0.0, np.pi])
        assert sensor.reading() == (None, None)
        assert not sensor._plot_line.get_visible()
    finally:
        plt.close("all")


def test_sensor_init_recreates_artists() -> None:
    robot = Bicycle(x0=[0, 0, 0])
    sensor = RangeBearingSensor(
        robot, LandmarkMap(np.array([[2.0], [0.0]])), range=3, angle=1, plot=True
    )
    try:
        sensor.plot(0)
        old_line, old_poly = sensor._plot_line, sensor._plot_poly
        ax = sensor._plot_ax
        sensor.init()
        assert old_line not in ax.lines
        assert old_poly not in ax.patches
        sensor.plot(0)
        assert sensor._plot_line is not old_line
        assert sensor._plot_poly is not old_poly
    finally:
        plt.close("all")


def test_sensor_without_limits_draws_beam_only() -> None:
    robot = Bicycle(x0=[0, 0, 0])
    sensor = RangeBearingSensor(
        robot, LandmarkMap(np.array([[2.0], [0.0]])), plot=True
    )
    try:
        z, idx = sensor.reading()
        assert z is not None and idx == 0
        assert sensor._plot_line is not None
        assert sensor._plot_poly is None
    finally:
        plt.close("all")
