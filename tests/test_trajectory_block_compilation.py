"""Trajectory sources support bdsim's compile-before-start lifecycle."""

import bdsim
import numpy as np
import numpy.testing as nt
import pytest
from spatialmath import SE3


def phase_at(s: float, profile: str) -> float:
    if profile == "linear":
        return s
    if profile == "quintic":
        return 10 * s**3 - 15 * s**4 + 6 * s**5
    if s <= 1 / 3:
        return 2.25 * s**2
    if s <= 2 / 3:
        return 1.5 * s - 0.25
    return 1 - 2.25 * (1 - s) ** 2


@pytest.mark.parametrize("duration", [None, 1.0])
@pytest.mark.parametrize(
    "kind,profile",
    [
        ("ctraj", "linear"),
        ("ctraj", "trapezoidal"),
        ("trapezoidal", "trapezoidal"),
        ("traj", "trapezoidal"),
        ("traj", "quintic"),
    ],
)
def test_compile_and_run_trajectory_twice(
    kind: str, profile: str, duration: float | None
) -> None:
    sim = bdsim.BDSim(graphics=False, animation=False, quiet=True)
    bd = sim.blockdiagram()
    if kind == "ctraj":
        initial = np.array([1.0, -2.0, 0.5])
        final = np.array([3.0, 1.0, -0.5])
        block = bd.CTRAJ(
            SE3(initial),
            SE3(final),
            T=duration,
            trapezoidal=profile == "trapezoidal",
        )
        position = bd.FUNCTION(lambda pose: pose.t, nin=1, nout=1)
        bd.connect(block, position)
        watched = position[0]
    elif kind == "trapezoidal":
        initial, final = np.array([-1.0]), np.array([3.0])
        block = bd.TRAPEZOIDAL(-1, 3, T=duration)
        watched = block[0]
    else:
        initial, final = np.array([-1.0, 2.0]), np.array([3.0, -2.0])
        block = bd.TRAJ(initial, final, T=duration, time=True, traj=profile)
        watched = block[0]

    # bdsim evaluates outputs here, before it supplies the run length to start().
    bd.compile()

    for run_length in (0.25, 0.5):
        result = sim.run(bd, T=run_length, dt=run_length / 8, watch=[watched])
        nt.assert_allclose(result.t[-1], run_length)
        actual = np.asarray(result.y).reshape(-1, initial.size)
        scale = run_length if duration is None else duration
        phase = np.array([phase_at(t / scale, profile) for t in result.t])
        expected = initial + phase[:, None] * (final - initial)
        nt.assert_allclose(actual, expected, atol=1e-12)
        assert block.T == duration


@pytest.mark.parametrize("profile", ["trapezoidal", "quintic"])
def test_input_driven_trajectory_compiles_and_uses_its_input(profile: str) -> None:
    sim = bdsim.BDSim(graphics=False, animation=False, quiet=True)
    bd = sim.blockdiagram()
    phase = bd.CONSTANT(0.25)
    block = bd.TRAJ([-1, 2], [3, -2], traj=profile)
    bd.connect(phase, block)

    bd.compile()
    result = sim.run(bd, T=0.2, dt=0.1, watch=[block[0]])

    expected = np.array([-1.0, 2.0]) + phase_at(0.25, profile) * np.array([4, -4])
    nt.assert_allclose(result.y, np.tile(expected, (len(result.t), 1)), atol=1e-12)
