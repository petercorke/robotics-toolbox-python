"""
Tests of the ``block`` argument of the plot methods.

The contract, for every method that has a ``block`` argument and shows a figure with
``plt.show``:

=========  =======================================================
``block``  what the method does about showing the figure
=========  =======================================================
``None``   nothing, no call to ``plt.show``
``False``  ``plt.show(block=False)``
``True``   ``plt.show(block=True)``
=========  =======================================================

These tests replace ``plt.show`` with a mock, so no window is opened and nothing
waits.
"""

import pathlib
import re
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from roboticstoolbox import (
    BinaryOccupancyGrid,
    DistanceTransformPlanner,
    LandmarkMap,
    jtraj,
)
from roboticstoolbox.tools.plot import apply_block, xplot

SRC = pathlib.Path(__file__).parent.parent / "src" / "roboticstoolbox"


class BlockContractTestCase(unittest.TestCase):
    """Check what ``plt.show`` is called with for each value of ``block``."""

    def setUp(self):
        patcher = mock.patch.object(plt, "show")
        self.show = patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(plt.close, "all")

    def check(self, call):
        """``call(block)`` must follow the contract for None, False and True."""
        for block, expected in (
            (None, []),
            (False, [mock.call(block=False)]),
            (True, [mock.call(block=True)]),
        ):
            self.show.reset_mock()
            call(block)
            self.assertEqual(
                self.show.call_args_list, expected, msg=f"block={block!r}"
            )
            plt.close("all")


class TestApplyBlock(BlockContractTestCase):
    def test_contract(self):
        self.check(apply_block)


class TestPlotMethods(BlockContractTestCase):
    def test_occupancy_grid(self):
        grid = BinaryOccupancyGrid(workspace=[0, 10, 0, 10], cellsize=1)
        self.check(lambda block: grid.plot(block=block))

    def test_landmark_map(self):
        landmarks = LandmarkMap(10, workspace=10, seed=0)
        self.check(lambda block: landmarks.plot(block=block))

    def test_planner_plot(self):
        # PlannerBase.plot used to ignore block=False, it tested ``if block:``
        planner = DistanceTransformPlanner(np.zeros((20, 20), dtype=int))
        planner.plan(goal=(15, 15))
        self.check(lambda block: planner.plot(block=block))

    def test_planner_plot_3d(self):
        planner = DistanceTransformPlanner(np.zeros((20, 20), dtype=int))
        planner.plan(goal=(15, 15))
        self.check(lambda block: planner.plot_3d(block=block))

    def test_trajectory_plot(self):
        # used to call plt.show(block=None) for None, which waits in a script
        traj = jtraj(np.zeros(2), np.ones(2), 20)
        self.check(lambda block: traj.plot(block=block))

    def test_xplot(self):
        self.check(lambda block: xplot(np.arange(10), np.arange(10), block=block))

    def test_defaults_are_unchanged(self):
        # the documented defaults: None for the map and planner plots, False for
        # xplot and Trajectory.plot, True for plot_3d
        grid = BinaryOccupancyGrid(workspace=[0, 10, 0, 10], cellsize=1)
        grid.plot()
        self.show.assert_not_called()
        plt.close("all")

        xplot(np.arange(10), np.arange(10))
        self.show.assert_called_once_with(block=False)
        plt.close("all")

        self.show.reset_mock()
        planner = DistanceTransformPlanner(np.zeros((20, 20), dtype=int))
        planner.plan(goal=(15, 15))
        planner.plot_3d()
        self.show.assert_called_once_with(block=True)


class TestOneImplementation(unittest.TestCase):
    """The contract lives in ``apply_block``: no other module may re-implement it."""

    def test_no_direct_plt_show_with_block(self):
        offenders = []
        for path in SRC.rglob("*.py"):
            if path.name == "plot.py" and path.parent.name == "tools":
                continue
            if "Dynamixel" in path.parts:
                continue
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                code = line.split("#")[0]
                if re.search(r"plt\.show\(block=block\)", code):
                    offenders.append(f"{path.relative_to(SRC)}:{n}")
        self.assertEqual(
            offenders, [], msg="use apply_block(block) instead of plt.show(block=block)"
        )


if __name__ == "__main__":  # pragma nocover
    unittest.main()
