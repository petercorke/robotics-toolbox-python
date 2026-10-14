"""Quintic boundary conditions across trajectory time scales."""

from collections.abc import Callable
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import numpy.testing as nt
import pytest
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike

from roboticstoolbox import jtraj, mtraj, quintic, quintic_func, quintic_hermite_func


@pytest.mark.parametrize("duration", [1.0, 5.0, 10.0, 49.0, -1.0, -10.0])
@pytest.mark.parametrize("moving_endpoints", [False, True])
def test_quintic_func_linear_system_control(
    duration: float, moving_endpoints: bool
) -> None:
    # Independent integer-coefficient polynomials, including nonzero velocities.
    polynomial = Polynomial(
        [1, 1, 0, -4, 7, -3] if moving_endpoints else [1, 0, 0, 10, -15, 6]
    )
    samples = np.array([0.0, 0.125, 0.5, 0.875, 1.0])
    actual = quintic_func(
        1,
        2,
        duration,
        qd0=polynomial.deriv()(0) / duration,
        qdf=polynomial.deriv()(1) / duration,
    )(samples * duration)
    for order, values in enumerate(actual):
        nt.assert_allclose(
            values * duration**order,
            polynomial.deriv(order)(samples),
            rtol=1e-10,
            atol=1e-10,
        )


@pytest.mark.parametrize("duration", [0.0, 0.001, -0.001, 1000.0, -1000.0, 1e6])
def test_quintic_func_rejects_numerically_singular_system(duration: float) -> None:
    with pytest.raises(np.linalg.LinAlgError, match="use quintic_hermite_func"):
        quintic_func(1, 2, duration)


@pytest.mark.parametrize("duration", [0.0019, 0.0021, 920.0, 970.0])
def test_quintic_func_condition_cutoff(duration: float) -> None:
    # Build the boundary operator from independently differentiated monomials,
    # using ascending powers rather than copying the production matrix.
    matrix = np.array(
        [
            [Polynomial.basis(power).deriv(order)(time) for power in range(6)]
            for order in range(3)
            for time in (0, duration)
        ]
    )
    cutoff = 1 / (6 * np.finfo(float).eps)
    condition = np.linalg.cond(matrix)
    # These samples straddle each cutoff with a margin against SVD round-off.
    nt.assert_equal(0.7 < condition / cutoff < 1.4, True)
    if duration in (0.0019, 970.0):
        nt.assert_equal(condition > cutoff, True)
        with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
            quintic_func(1, 2, duration)
    else:
        nt.assert_equal(condition < cutoff, True)
        values = quintic_func(1, 2, duration)([0, duration / 2, duration])
        nt.assert_equal(np.all(np.isfinite(values)), True)


@pytest.mark.parametrize("duration", [np.nan, np.inf, -np.inf])
def test_quintic_func_nonfinite_duration(duration: float) -> None:
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        quintic_func(1, 2, duration)
    with pytest.raises(ValueError, match="duration must be finite"):
        quintic_hermite_func(1, 2, duration)


def test_quintic_hermite_func_public_exports() -> None:
    import roboticstoolbox
    import roboticstoolbox.tools

    nt.assert_equal(
        roboticstoolbox.tools.quintic_hermite_func is quintic_hermite_func, True
    )
    for module in (roboticstoolbox, roboticstoolbox.tools):
        nt.assert_equal("quintic_hermite_func" in module.__all__, True)


@pytest.mark.parametrize("duration", [-0.001, -1.0, -1000.0, 1e-6])
def test_quintic_hermite_func_signed_and_short_duration(duration: float) -> None:
    polynomial = Polynomial([1, 1, 0, -4, 7, -3])
    samples = np.array([0, 0.25, 0.5, 0.75, 1])
    actual = quintic_hermite_func(1, 2, duration, 1 / duration, 2 / duration)(
        samples * duration
    )
    for order, values in enumerate(actual):
        nt.assert_allclose(
            values * duration**order,
            polynomial.deriv(order)(samples),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize(
    "duration",
    [
        np.float16(0.001),
        np.float32(1e-25),
        np.float32(1e-20),
        np.float32(1e20),
        np.float32(1e30),
        np.float32(-1e-25),
        np.int64(10**10),
        np.complex64(1e-25 + 1e-25j),
        np.complex64(1e20 + 1e20j),
    ],
)
@pytest.mark.parametrize("moving_endpoints", [False, True])
def test_quintic_hermite_func_promotes_duration(
    duration: complex, moving_endpoints: bool
) -> None:
    # Use the represented duration, then scale in float64/complex128. In
    # particular, a float32 square must not overflow or underflow before the
    # derivative is divided by it.
    represented_duration = (
        complex(duration) if np.iscomplexobj(duration) else float(duration)
    )
    polynomial = Polynomial(
        [1, 1, 0, -4, 7, -3] if moving_endpoints else [1, 0, 0, 10, -15, 6]
    )
    samples = np.array([0, 0.25, 0.5, 0.75, 1])
    actual = quintic_hermite_func(
        1,
        2,
        duration,
        qd0=polynomial.deriv()(0) / represented_duration,
        qdf=polynomial.deriv()(1) / represented_duration,
    )(samples * represented_duration)
    nt.assert_equal(np.all(np.isfinite(actual)), True)
    for order, values in enumerate(actual):
        nt.assert_allclose(
            values * represented_duration**order,
            polynomial.deriv(order)(samples),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("duration", [1e155, -1e155])
@pytest.mark.parametrize("moving_endpoints", [False, True])
def test_quintic_hermite_func_large_duration_acceleration(
    duration: float, moving_endpoints: bool
) -> None:
    polynomial = Polynomial(
        [1, 1, 0, -4, 7, -3] if moving_endpoints else [1, 0, 0, 10, -15, 6]
    )
    samples = np.array([0, 0.25, 0.5, 0.75, 1])
    actual = quintic_hermite_func(
        1,
        2,
        duration,
        qd0=polynomial.deriv()(0) / duration,
        qdf=polynomial.deriv()(1) / duration,
    )(samples * duration)
    nt.assert_equal(np.all(np.isfinite(actual)), True)
    for order, values in enumerate(actual):
        # Scale sequentially: duration**2 overflows although the small physical
        # acceleration and its normalized reference are both representable.
        scaled = values
        for _ in range(order):
            scaled = scaled * duration
        nt.assert_allclose(
            scaled, polynomial.deriv(order)(samples), rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize("duration", [1e-200, -1e-200])
def test_quintic_hermite_func_tiny_duration_endpoints(duration: float) -> None:
    # Interior acceleration need not fit float64 at this scale, but the exact
    # zero endpoint accelerations must not turn into 0/0 when T**2 underflows.
    actual = quintic_hermite_func(1, 2, duration, 0.2, -0.3)([0, duration])
    nt.assert_allclose(actual[0], [1, 2], atol=1e-12)
    nt.assert_allclose(actual[1], [0.2, -0.3], atol=1e-12)
    nt.assert_array_equal(actual[2], [0, 0])


@pytest.mark.parametrize("duration", [1e-200, -1e-200])
@pytest.mark.parametrize("moving_endpoints", [False, True])
def test_quintic_hermite_func_tiny_duration_small_displacement(
    duration: float, moving_endpoints: bool
) -> None:
    displacement = 1e-200
    polynomial = Polynomial(
        [0, 1, 0, -4, 7, -3] if moving_endpoints else [0, 0, 0, 10, -15, 6]
    )
    samples = np.array([0, 0.25, 0.5, 0.75, 1])
    actual = quintic_hermite_func(
        0,
        displacement,
        duration,
        qd0=polynomial.deriv()(0) * displacement / duration,
        qdf=polynomial.deriv()(1) * displacement / duration,
    )(samples * duration)
    nt.assert_equal(np.all(np.isfinite(actual)), True)
    for order, values in enumerate(actual):
        # A tiny displacement gives representable interior acceleration even
        # though duration**2 is zero in float64. Keep the oracle scaling safe.
        scaled = values
        for _ in range(order):
            scaled = scaled * duration
        nt.assert_allclose(
            scaled / displacement,
            polynomial.deriv(order)(samples),
            rtol=1e-12,
            atol=1e-12,
        )


def test_quintic_single_sample_zero_duration() -> None:
    trajectory = quintic(1, 2, 1, 0.2, -0.1)
    nt.assert_array_equal(trajectory.t, [0])
    nt.assert_allclose(trajectory.q, [1.5])
    nt.assert_allclose(trajectory.qd, [0.05])
    nt.assert_allclose(trajectory.qdd, [0])


def test_quintic_hermite_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    from roboticstoolbox.tools import trajectory

    calls = []
    hermite = trajectory.quintic_hermite_func

    def record_hermite(*args: float) -> Callable:
        calls.append(args)
        return hermite(*args)

    def unexpected_linear_system(*args: float) -> Callable:
        pytest.fail("quintic() must evaluate with the Hermite function")

    monkeypatch.setattr(trajectory, "quintic_hermite_func", record_hermite)
    monkeypatch.setattr(trajectory, "quintic_func", unexpected_linear_system)
    actual = trajectory.quintic(1, 2, [5, 505, 1005], qd0=0.2, qdf=-0.3)
    nt.assert_equal(calls, [(1, 2, 1000, 0.2, -0.3)])
    nt.assert_array_equal(actual.t, [5, 505, 1005])
    nt.assert_allclose(actual.q[[0, -1]], [1, 2])
    nt.assert_allclose(actual.qd[[0, -1]], [0.2, -0.3])
    nt.assert_allclose(actual.qdd[[0, -1]], [0, 0])


def test_bdsim_quintic_retains_linear_system() -> None:
    from roboticstoolbox.blocks.arm import Traj

    ordinary = Traj(y0=1, yf=2, T=5, time=True, traj="quintic")
    ordinary.start(SimpleNamespace(T=5))
    nt.assert_allclose(ordinary.output(2.5, [], []), [[1.5], [0.375], [0]], atol=1e-12)
    long_duration = Traj(y0=1, yf=2, T=1000, time=True, traj="quintic")
    with pytest.raises(np.linalg.LinAlgError, match="use quintic_hermite_func"):
        long_duration.start(SimpleNamespace(T=1000))


@pytest.mark.parametrize("duration", [0.001, 0.01, 1.0, 10.0, 999.0, 1000.0, 1e6])
@pytest.mark.parametrize("moving_endpoints", [False, True])
def test_quintic_hermite_func_analytic_polynomial(
    duration: float, moving_endpoints: bool
) -> None:
    # These fixed polynomials independently specify all six boundary conditions.
    # Both have p(0)=1, p(1)=2, p''(0)=p''(1)=0. The second has p'(0)=1,
    # p'(1)=2; the first has zero endpoint velocities.
    polynomial = Polynomial(
        [1, 1, 0, -4, 7, -3] if moving_endpoints else [1, 0, 0, 10, -15, 6]
    )
    velocity = polynomial.deriv()
    acceleration = velocity.deriv()
    samples = np.array([0.0, 0.125, 0.25, 0.5, 0.875, 1.0])
    function = quintic_hermite_func(
        1, 2, duration, qd0=velocity(0) / duration, qdf=velocity(1) / duration
    )
    position, speed, accel = function(samples * duration)

    nt.assert_allclose(position, polynomial(samples), rtol=1e-12, atol=1e-12)
    nt.assert_allclose(speed * duration, velocity(samples), rtol=1e-12, atol=1e-12)
    nt.assert_allclose(
        accel * duration**2, acceleration(samples), rtol=1e-12, atol=1e-12
    )
    nt.assert_allclose(position[[0, -1]], [1, 2], atol=1e-12)
    nt.assert_allclose(
        speed[[0, -1]] * duration, [velocity(0), velocity(1)], atol=1e-12
    )
    nt.assert_allclose(accel[[0, -1]] * duration**2, [0, 0], atol=1e-12)


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
def test_quintic_hermite_func_physical_endpoint_velocities(duration: float) -> None:
    # Independent Hermite basis functions, with fixed velocities in time units.
    position_basis = Polynomial([0, 0, 0, 10, -15, 6])
    initial_velocity_basis = Polynomial([0, 1, 0, -6, 8, -3])
    final_velocity_basis = Polynomial([0, 0, 0, -4, 7, -3])
    samples = np.array([0.0, 0.125, 0.5, 0.875, 1.0])
    actual = quintic_hermite_func(-2, 3, duration, qd0=0.2, qdf=-0.3)(
        samples * duration
    )
    for order, values in enumerate(actual):
        # Evaluate each integer-coefficient basis before scaling so exact zero
        # endpoint derivatives are not replaced by oracle cancellation noise.
        reference = (
            5 * position_basis.deriv(order)(samples)
            + (0.2 * duration) * initial_velocity_basis.deriv(order)(samples)
            - (0.3 * duration) * final_velocity_basis.deriv(order)(samples)
        )
        if order == 0:
            reference -= 2
        nt.assert_allclose(
            values * duration**order,
            reference,
            rtol=1e-12,
            atol=1e-12,
        )
    nt.assert_allclose(actual[0][[0, -1]], [-2, 3], atol=1e-12)
    nt.assert_allclose(actual[1][[0, -1]], [0.2, -0.3], atol=1e-12)
    nt.assert_allclose(actual[2][[0, -1]], [0, 0], atol=1e-9)


@pytest.mark.parametrize("steps", [11, 1000])
def test_quintic_integer_samples(steps: int) -> None:
    trajectory = quintic(1, 2, steps)
    samples = np.arange(steps) / (steps - 1)
    polynomial = Polynomial([1, 0, 0, 10, -15, 6])

    nt.assert_array_equal(trajectory.t, np.arange(steps))
    nt.assert_equal(not trajectory.istime, True)
    nt.assert_equal(
        trajectory.q.shape == trajectory.qd.shape == trajectory.qdd.shape == (steps,),
        True,
    )
    nt.assert_allclose(trajectory.q, polynomial(samples), rtol=1e-12, atol=1e-12)
    nt.assert_allclose(
        trajectory.qd * (steps - 1), polynomial.deriv()(samples), atol=1e-12
    )
    nt.assert_allclose(
        trajectory.qdd * (steps - 1) ** 2,
        polynomial.deriv(2)(samples),
        atol=1e-12,
    )


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
@pytest.mark.parametrize("origin", [-5.0, 0.0, 5.0])
def test_quintic_sampled_time_origin(duration: float, origin: float) -> None:
    times = origin + duration * np.array([0.0, 0.125, 0.5, 0.875, 1.0])
    actual_duration = times[-1] - times[0]
    # Use the represented elapsed times, including floating-point timestamp noise.
    samples = (times - times[0]) / actual_duration
    polynomial = Polynomial([1, 1, 0, -4, 7, -3])
    trajectory = quintic(1, 2, times, qd0=1 / actual_duration, qdf=2 / actual_duration)

    nt.assert_array_equal(trajectory.t, times)
    nt.assert_equal(trajectory.istime, True)
    for values, derivative, scale in (
        (trajectory.q, polynomial, 1),
        (trajectory.qd, polynomial.deriv(), actual_duration),
        (trajectory.qdd, polynomial.deriv(2), actual_duration**2),
    ):
        nt.assert_allclose(values * scale, derivative(samples), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
def test_quintic_multi_axis_native_control(duration: float) -> None:
    times = np.linspace(0, duration, 11)
    initial = np.array([1, -2, 3])
    final = np.array([2, 4, 3])
    actual = mtraj(quintic, initial, final, times)
    expected = jtraj(initial, final, times)

    nt.assert_equal(
        actual.q.shape == actual.qd.shape == actual.qdd.shape == (11, 3), True
    )
    nt.assert_array_equal(actual.t, times)
    nt.assert_equal(actual.istime, True)
    for values, reference, scale in (
        (actual.q, expected.q, 1),
        (actual.qd, expected.qd, duration),
        (actual.qdd, expected.qdd, duration**2),
    ):
        nt.assert_equal(np.issubdtype(values.dtype, np.floating), True)
        nt.assert_allclose(values * scale, reference * scale, rtol=1e-12, atol=1e-12)
    nt.assert_allclose(actual.q[[0, -1]], [initial, final], atol=1e-12)


@pytest.mark.parametrize("samples", [0.5, [0, 0.5, 1], [[0, 0.5], [0.75, 1]]])
def test_quintic_hermite_func_sample_shapes(samples: ArrayLike) -> None:
    function = quintic_hermite_func(np.int64(1), np.float32(2), 1)
    position, velocity, acceleration = function(samples)
    polynomial = Polynomial([1, 0, 0, 10, -15, 6])
    for actual, expected in zip(
        (position, velocity, acceleration),
        (
            polynomial(samples),
            polynomial.deriv()(samples),
            polynomial.deriv(2)(samples),
        ),
    ):
        nt.assert_equal(np.shape(actual) == np.shape(samples), True)
        nt.assert_equal(np.issubdtype(np.asarray(actual).dtype, np.floating), True)
        nt.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("duration", [0.001, 1.0, 1000.0])
@pytest.mark.parametrize(
    "samples",
    [
        np.float32(0.3),
        np.array([0, 0.125, 0.3, 0.5, 0.7, 0.875, 1], dtype=np.float32),
        np.array([[0, 0.3], [0.7, 1]], dtype=np.float32),
    ],
)
def test_quintic_hermite_func_float32_sample_precision(
    duration: float, samples: ArrayLike
) -> None:
    times = np.asarray(samples) * np.float32(duration)
    # Evaluate independent integer-coefficient polynomials at the exactly
    # represented samples, including the rounding of the physical timestamps.
    expected = []
    with localcontext() as context:
        context.prec = 70
        for time in np.asarray(times).flat:
            u = Decimal.from_float(float(time)) / Decimal.from_float(duration)
            expected.append(
                [
                    1 + 10 * u**3 - 15 * u**4 + 6 * u**5,
                    30 * u**2 - 60 * u**3 + 30 * u**4,
                    60 * u - 180 * u**2 + 120 * u**3,
                ]
            )
    reference = np.asarray(expected, dtype=float).T.reshape((3, *np.shape(times)))
    for order, actual in enumerate(quintic_hermite_func(1, 2, duration)(times)):
        nt.assert_equal(np.shape(actual) == np.shape(times), True)
        nt.assert_equal(np.asarray(actual).dtype == np.dtype(np.float64), True)
        nt.assert_allclose(
            actual * duration**order, reference[order], rtol=1e-12, atol=1e-12
        )


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        np.int64,
        np.complex64,
        np.complex128,
        np.longdouble,
        np.clongdouble,
    ],
)
def test_quintic_hermite_func_sample_dtype_compatibility(dtype: type) -> None:
    values = (
        [0.3 + 0.2j, 0.7 - 0.1j]
        if np.issubdtype(dtype, np.complexfloating)
        else [0, 0.3, 1]
    )
    samples = np.asarray(values, dtype=dtype)
    result_dtype = np.result_type(samples.dtype, np.float64)
    polynomial = Polynomial([1, 0, 0, 10, -15, 6])
    for order, actual in enumerate(quintic_hermite_func(1, 2, 1)(samples)):
        nt.assert_equal(isinstance(actual, np.ndarray), True)
        nt.assert_equal(actual.shape == samples.shape, True)
        nt.assert_equal(actual.dtype == result_dtype, True)
        nt.assert_allclose(
            actual,
            polynomial.deriv(order)(samples.astype(result_dtype)),
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("dtype", [np.longdouble, np.clongdouble])
@pytest.mark.parametrize("argument", ["q0", "T", "qd0"])
def test_quintic_hermite_func_extended_boundary_dtype_compatibility(
    dtype: type, argument: str
) -> None:
    arguments = {"q0": 1, "qf": 2, "T": 1, "qd0": 0}
    arguments[argument] = dtype(arguments[argument])
    result_dtype = np.result_type(dtype, np.float64)
    if result_dtype in (np.dtype(np.float64), np.dtype(np.complex128)):
        # Some platforms give longdouble the same precision as float64.
        nt.assert_allclose(
            quintic_hermite_func(**arguments)(0.5), [1.5, 1.875, 0], atol=1e-12
        )
    else:
        with pytest.raises(
            TypeError, match=f"array type {result_dtype} is unsupported"
        ):
            quintic_hermite_func(**arguments)


def test_quintic_zero_duration_compatibility() -> None:
    # Preserve the existing least-squares result for the degenerate duration.
    function = quintic_hermite_func(1, 2, 0, qd0=0.2, qdf=-0.1)
    position, velocity, acceleration = function(np.array([0, 0.5, 1]))
    nt.assert_allclose(position, [1.5, 1.525, 1.55], atol=1e-12)
    nt.assert_allclose(velocity, [0.05, 0.05, 0.05], atol=1e-12)
    nt.assert_allclose(acceleration, [0, 0, 0], atol=1e-12)


@pytest.mark.parametrize("container", [list, np.array])
def test_quintic_hermite_func_single_element_endpoint_arrays(
    container: Callable[[list[int]], ArrayLike],
) -> None:
    actual = quintic_hermite_func(container([1]), container([2]), 1)([0, 0.5, 1])
    nt.assert_allclose(actual[0], [1, 1.5, 2], atol=1e-12)
    nt.assert_allclose(actual[1], [0, 1.875, 0], atol=1e-12)
    nt.assert_allclose(actual[2], [0, 0, 0], atol=1e-12)


def test_quintic_hermite_func_rejects_multi_axis_endpoints() -> None:
    with pytest.raises(np.linalg.LinAlgError, match="Incompatible dimensions"):
        quintic_hermite_func([1, 2], [3, 4], 1)


def test_quintic_invalid_sampling_argument() -> None:
    with pytest.raises(TypeError, match="bad argument for time"):
        quintic(1, 2, "not time")
