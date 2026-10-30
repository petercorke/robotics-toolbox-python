"""Quintic sampling preserves its terminal boundary conditions."""

import numpy as np
import numpy.testing as npt
import pytest

from roboticstoolbox.mobile.QuinticPolyPlanner import quintic_polynomials_planner


@pytest.mark.parametrize(
    "duration,step",
    [(1.0, 0.3), (1.1, 0.3), (1.0, 2.0), (1.0, 0.25), (3 * 0.1, 0.1)],
)
@pytest.mark.parametrize("heading", [-0.4, 0.7])
def test_terminal_boundary_is_sampled_once(
    duration: float, step: float, heading: float
) -> None:
    times, path, velocity, acceleration, jerk = quintic_polynomials_planner(
        sx=0.2,
        sy=-0.3,
        syaw=0.1,
        sv=0.6,
        sa=0.0,
        gx=1.2,
        gy=0.8,
        gyaw=heading,
        gv=0.4,
        ga=0.0,
        max_accel=np.inf,
        max_jerk=np.inf,
        dt=step,
        MIN_T=duration,
        MAX_T=2 * duration,
    )

    assert times[0] == 0.0
    assert times[-1] == duration
    assert np.count_nonzero(np.asarray(times) == duration) == 1
    assert np.all(np.diff(times) > 0)
    assert np.all(np.asarray(times) <= duration)
    assert np.all(np.diff(times) <= step + 1e-15)
    assert path.shape == (len(times), 3)
    assert len(velocity) == len(acceleration) == len(jerk) == len(times)
    npt.assert_allclose(path[0], [0.2, -0.3, 0.1], atol=1e-12)
    npt.assert_allclose(path[-1], [1.2, 0.8, heading], atol=1e-10)
    npt.assert_allclose(velocity[-1], 0.4, atol=1e-10)
    npt.assert_allclose(acceleration[-1], 0.0, atol=1e-9)


def test_exact_grid_spacing_is_preserved() -> None:
    times, *_ = quintic_polynomials_planner(
        0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 100, 100, 0.25, 1.0, 2.0
    )
    npt.assert_array_equal(times, [0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize("step", [-1.0, -0.1, 0.0, np.inf, -np.inf, np.nan])
def test_invalid_time_step_is_rejected(step: float) -> None:
    with pytest.raises(ValueError):
        quintic_polynomials_planner(
            0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 100, 100, step, 1.0, 2.0
        )
