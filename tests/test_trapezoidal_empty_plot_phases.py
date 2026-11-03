"""All trajectory samples remain visible when a motion phase has no samples."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest

from roboticstoolbox.tools.trajectory import Trajectory, trapezoidal

matplotlib.use("Agg")


@pytest.mark.parametrize(
    "samples,velocity,goal",
    [
        ([0.0, 0.5, 1.0], 2.0, 1.0),
        ([0.0, 0.25, 0.5, 0.75, 1.0], 1.0, 1.0),
        ([0.0, 1.0], None, 1.0),
        ([0.0, 0.5, 1.0], None, 0.0),
        ([0.0, 0.25, 0.5, 0.75, 1.0], None, 1.0),
    ],
    ids=["triangular", "no-deceleration", "sparse", "stationary", "all-phases"],
)
@pytest.mark.parametrize("offset", [0.0, 2.5])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_plot_preserves_samples_with_empty_phases(
    samples: list[float],
    velocity: float | None,
    goal: float,
    offset: float,
    direction: float,
) -> None:
    if velocity == 1.0:
        # Finite constant-velocity data isolates an empty deceleration phase.
        trajectory = Trajectory(
            "trapezoidal",
            np.asarray(samples) + offset,
            direction * np.asarray(samples),
            np.full(len(samples), direction),
            np.zeros(len(samples)),
            istime=True,
        )
        trajectory.tblend = 0.0
    else:
        trajectory = trapezoidal(
            0.0,
            direction * goal,
            np.asarray(samples) + offset,
            V=None if velocity is None else direction * velocity,
        )
    assert np.isfinite(trajectory.s).all()
    saved = [
        value.copy()
        for value in (trajectory.t, trajectory.s, trajectory.sd, trajectory.sdd)
    ]
    try:
        trajectory.plot(block=None)
        axes = plt.gcf().axes
        assert len(axes) == 3
        plotted = np.concatenate(
            [
                np.column_stack((line.get_xdata(), line.get_ydata()))
                for line in axes[0].lines
            ]
        )
        expected = np.column_stack((trajectory.t, trajectory.s))
        for point in expected:
            assert np.any(np.all(plotted == point, axis=1))
        for point in plotted:
            assert np.any(np.all(expected == point, axis=1))
        npt.assert_array_equal(axes[1].lines[0].get_xdata(), trajectory.t)
        npt.assert_array_equal(axes[1].lines[0].get_ydata(), trajectory.sd)
        npt.assert_array_equal(axes[2].lines[0].get_ydata(), trajectory.sdd)
        for old, current in zip(
            saved, (trajectory.t, trajectory.s, trajectory.sd, trajectory.sdd)
        ):
            npt.assert_array_equal(current, old)
    finally:
        plt.close("all")
