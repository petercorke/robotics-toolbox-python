import matplotlib.pyplot as plt
import numpy as np
import pytest
import roboticstoolbox as rtb


def test_distance_transform_plot_colorbar_without_explicit_axes():
    floorplan = np.zeros((10, 10), dtype=int)
    floorplan[4:7, 4:7] = 1

    planner = rtb.DistanceTransformPlanner(floorplan, inflate=1)
    planner.plan((8, 8))

    # Regression test: colorbar creation must not require caller-provided axes.
    planner.plot(block=False)
    plt.close("all")


def test_distance_transform_next_before_plan_raises():
    # Regression test: next() used to call the undefined name Error(...)
    # instead of raising, so calling it before plan() crashed with
    # NameError rather than the intended ValueError.
    floorplan = np.zeros((10, 10), dtype=int)
    planner = rtb.DistanceTransformPlanner(floorplan, inflate=1)

    with pytest.raises(ValueError, match="No distance map computed"):
        planner.next((0, 0))


def test_distance_transform_next_uses_all_diagonals():
    floorplan = np.zeros((5, 5), dtype=int)
    planner = rtb.DistanceTransformPlanner(floorplan, metric="euclidean")
    planner.plan((3, 1))

    np.testing.assert_array_equal(planner.next((1, 3)), np.array([2, 2]))


def test_distancexform_animate():
    # Regression test: the animate path used matplotlib.cm.get_cmap(),
    # removed in matplotlib 3.9, so plan(animate=True) raised
    # AttributeError before computing the distance map.
    floorplan = np.zeros((10, 10), dtype=int)
    floorplan[4:7, 4:7] = 1

    planner = rtb.DistanceTransformPlanner(floorplan, inflate=1)
    planner.plan((8, 8), animate=True)

    assert planner.distancemap is not None
    plt.close("all")
