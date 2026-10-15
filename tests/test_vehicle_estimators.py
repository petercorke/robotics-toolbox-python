"""
Tests for the vehicle models, range-bearing sensor, landmark map and the two
state estimators (EKF and particle filter).

The vehicle models and their Jacobians are checked against closed-form results
and finite differences.  The estimators are checked against the simulated ground
truth, using fixed seeds so that the runs are reproducible; the bounds are
loose multiples of the errors measured at the time of writing, so that they
catch a broken estimator without being brittle.
"""

import contextlib
import io
import unittest
from collections.abc import Callable
from typing import Any, TypeVar
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as nt

from roboticstoolbox import (
    EKF,
    Bicycle,
    DiffSteer,
    LandmarkMap,
    ParticleFilter,
    RandomPath,
    RangeBearingSensor,
    Unicycle,
)

T = TypeVar("T")

V = np.diag([0.02, np.radians(0.5)]) ** 2  # odometry covariance
W = np.diag([0.1, np.radians(1)]) ** 2  # sensor covariance
P0 = np.diag([0.05, 0.05, np.radians(0.5)]) ** 2  # initial vehicle covariance


def quiet(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Call ``func`` discarding anything it prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def numerical_jacobian(
    func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """Central-difference Jacobian of ``func`` at ``x``."""
    cols = [(func(x + eps * e) - func(x - eps * e)) / (2 * eps) for e in np.eye(len(x))]
    return np.column_stack(cols)


class TestBicycle(unittest.TestCase):
    def test_deriv(self) -> None:
        bike = Bicycle(L=2)
        x = np.array([0, 0, 0.3])
        d = bike.deriv(x, [1.0, 0.2])
        nt.assert_allclose(d, [np.cos(0.3), np.sin(0.3), np.tan(0.2) / 2])

    def test_steer_limit(self) -> None:
        bike = Bicycle(L=2, steer_max=0.5)
        d = bike.deriv([0, 0, 0], [1.0, 5.0])
        self.assertAlmostEqual(d[2], np.tan(0.5) / 2)

    def test_speed_limit(self) -> None:
        bike = Bicycle(speed_max=3)
        d = bike.deriv([0, 0, 0], [10.0, 0.0])
        self.assertAlmostEqual(np.linalg.norm(d[:2]), 3)

    def test_limits_can_be_disabled(self) -> None:
        bike = Bicycle(L=2, steer_max=0.5)
        d = bike.deriv([0, 0, 0], [1.0, 1.0], limits=False)
        self.assertAlmostEqual(d[2], np.tan(1.0) / 2)

    def test_turning_radius(self) -> None:
        bike = Bicycle(L=2, steer_max=0.5)
        self.assertAlmostEqual(bike.radius_min, 2 / np.tan(0.5))
        self.assertAlmostEqual(bike.curvature_max, np.tan(0.5) / 2)

    def test_step_straight(self) -> None:
        bike = Bicycle(dt=0.1)
        odo = bike.step((1.0, 0.0), animate=False)
        nt.assert_allclose(bike.x, [0.1, 0, 0])
        nt.assert_allclose(odo, [0.1, 0])
        self.assertEqual(len(bike.x_hist), 1)

    def test_run_returns_history(self) -> None:
        bike = Bicycle(dt=0.1)
        hist = bike.run(T=2, control=(1.0, 0.1), animate=False)
        self.assertEqual(np.asarray(hist).shape, (20, 3))

    def test_odometry_noise_is_seeded(self) -> None:
        def odometry(seed: int) -> np.ndarray:
            bike = Bicycle(covar=V, seed=seed)
            return np.array([bike.step((1, 0.1), animate=False) for _ in range(5)])

        nt.assert_array_equal(odometry(1), odometry(1))
        self.assertFalse(np.allclose(odometry(1), odometry(2)))


class TestUnicycleAndDiffSteer(unittest.TestCase):
    def test_unicycle_circle(self) -> None:
        # unit speed and unit turn rate for half a revolution: a semicircle of
        # radius 1, ending at (0, 2) with heading pi.  W = 1 so that this does
        # not depend on how the turn rate is scaled by the vehicle width.
        veh = Unicycle(W=1, dt=0.001)
        hist = np.asarray(veh.run(T=np.pi, control=(1.0, 1.0), animate=False))
        nt.assert_allclose(hist[-1], [0, 2, np.pi], atol=5e-3)

    def test_diffsteer_wheel_speeds(self) -> None:
        veh = DiffSteer(W=2)
        # equal wheel speeds: straight line at that speed
        nt.assert_allclose(veh.deriv([0, 0, 0], [1.0, 1.0]), [1, 0, 0])
        # opposite wheel speeds: turn on the spot at (vR - vL) / W
        nt.assert_allclose(veh.deriv([0, 0, 0], [-1.0, 1.0]), [0, 0, 1.0])


class TestStateTransition(unittest.TestCase):
    def setUp(self) -> None:
        self.veh = Bicycle()
        self.x = np.array([1.0, 2.0, 0.4])
        self.odo = np.array([0.3, 0.05])

    def test_f(self) -> None:
        x = self.veh.f([0, 0, 0.5], [1, 0.1])
        nt.assert_allclose(x, [np.cos(0.5), np.sin(0.5), 0.6])

    def test_f_vectorized_matches_rows(self) -> None:
        xs = np.array([[0, 0, 0.0], [1, 2, 0.4], [-3, 1, -1.0]])
        batch = self.veh.f(xs, self.odo)
        for row, x in zip(batch, xs):
            nt.assert_allclose(row, self.veh.f(x, self.odo))

    def test_Fx_matches_finite_difference(self) -> None:
        J = self.veh.Fx(self.x, self.odo)
        Jn = numerical_jacobian(lambda x: self.veh.f(x, self.odo), self.x)
        nt.assert_allclose(J, Jn, atol=1e-6)

    def test_Fv_matches_finite_difference(self) -> None:
        J = self.veh.Fv(self.x, self.odo)
        Jn = numerical_jacobian(
            lambda v: self.veh.f(self.x, self.odo, v=v), np.zeros(2)
        )
        self.assertEqual(J.shape, (3, 2))
        nt.assert_allclose(J, Jn, atol=1e-6)


class TestLandmarkMap(unittest.TestCase):
    def test_random_map_is_seeded(self) -> None:
        a = LandmarkMap(10, workspace=10, seed=3)
        b = LandmarkMap(10, workspace=10, seed=3)
        c = LandmarkMap(10, workspace=10, seed=4)
        self.assertEqual(len(a), 10)
        nt.assert_array_equal(np.array(list(a)), np.array(list(b)))
        self.assertFalse(np.allclose(np.array(list(a)), np.array(list(c))))

    def test_landmarks_inside_workspace(self) -> None:
        m = LandmarkMap(50, workspace=[-3, 5, 1, 4])
        xy = np.array(list(m))
        self.assertEqual(xy.shape, (50, 2))
        self.assertTrue(np.all((xy[:, 0] >= -3) & (xy[:, 0] <= 5)))
        self.assertTrue(np.all((xy[:, 1] >= 1) & (xy[:, 1] <= 4)))

    def test_explicit_map(self) -> None:
        coords = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = LandmarkMap(coords, workspace=10)
        self.assertEqual(len(m), 3)
        nt.assert_array_equal(m[1], [2, 5])
        nt.assert_array_equal(np.array(list(m)), coords.T)

    def test_bad_map(self) -> None:
        with self.assertRaises(ValueError):
            LandmarkMap("twenty")


class TestRangeBearingSensor(unittest.TestCase):
    def setUp(self) -> None:
        self.robot = Bicycle(x0=[1, 2, 0.5])
        self.map = LandmarkMap(5, workspace=10, seed=0)

    def test_h_matches_geometry(self) -> None:
        sensor = RangeBearingSensor(self.robot, self.map, covar=W)
        for id in range(5):
            with self.subTest(landmark=id):
                dx, dy = self.map[id] - np.array([1, 2])
                bearing = np.arctan2(dy, dx) - 0.5
                nt.assert_allclose(
                    sensor.h([1, 2, 0.5], id),
                    [np.hypot(dx, dy), np.arctan2(np.sin(bearing), np.cos(bearing))],
                )

    def test_noise_free_reading_equals_h(self) -> None:
        sensor = RangeBearingSensor(self.robot, self.map)  # covar=None
        z, id = sensor.reading()
        nt.assert_allclose(z, sensor.h(self.robot.x, id))

    def test_range_limit(self) -> None:
        sensor = RangeBearingSensor(self.robot, self.map, range=3)
        for id in range(5):
            with self.subTest(landmark=id):
                inside = sensor.h(self.robot.x, id)[0] <= 3
                self.assertEqual(bool(sensor.isvisible(id)), bool(inside))

    def test_nothing_visible_gives_no_reading(self) -> None:
        sensor = RangeBearingSensor(self.robot, self.map, range=0.01)
        z, id = sensor.reading()
        self.assertIsNone(z)


class TestEKF(unittest.TestCase):
    T = 20
    maxdiff = 1e-9

    @staticmethod
    def robot(seed: int = 0, covar: np.ndarray | None = V) -> Bicycle:
        robot = Bicycle(covar=covar, seed=seed)
        robot.control = RandomPath(workspace=10, seed=seed)
        return robot

    @staticmethod
    def map() -> LandmarkMap:
        return LandmarkMap(20, workspace=10, seed=0)

    def sensor(self, robot: Bicycle, map: LandmarkMap) -> RangeBearingSensor:
        return RangeBearingSensor(
            robot, map, covar=W, angle=[-np.pi / 2, np.pi / 2], range=4
        )

    def error(self, ekf: EKF, robot: Bicycle) -> float:
        return float(np.linalg.norm(ekf.get_xyt()[-1, :2] - robot.x[:2]))

    def dead_reckoning(self, T: float | None = None) -> tuple[EKF, Bicycle]:
        robot = self.robot()
        ekf = EKF(robot=(robot, V), P0=P0, animate=False)
        quiet(ekf.run, T=T or self.T)
        return ekf, robot

    def test_dead_reckoning(self) -> None:
        ekf, robot = self.dead_reckoning()
        self.assertEqual(len(ekf.history), round(self.T / robot.dt))
        self.assertEqual(ekf.x_est.shape, (3,))
        self.assertLess(self.error(ekf, robot), 2.0)

    def test_dead_reckoning_uncertainty_grows(self) -> None:
        ekf, _ = self.dead_reckoning()
        pn = ekf.get_Pnorm()
        self.assertTrue(np.all(np.diff(pn) >= -1e-12))
        self.assertGreater(pn[-1], 100 * pn[0])

    def test_covariance_is_symmetric_positive_definite(self) -> None:
        ekf, _ = self.dead_reckoning(T=10)
        for P in ekf.get_P()[::10]:
            nt.assert_allclose(P, P.T, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(P).min(), -1e-12)

    def test_localization_beats_dead_reckoning(self) -> None:
        dr, dr_robot = self.dead_reckoning()

        robot = self.robot()
        map = self.map()
        ekf = EKF(
            robot=(robot, V),
            P0=P0,
            map=map,
            sensor=(self.sensor(robot, map), W),
            animate=False,
        )
        quiet(ekf.run, T=self.T)

        self.assertEqual(ekf.x_est.shape, (3,))
        self.assertLess(self.error(ekf, robot), 0.5)
        self.assertLess(self.error(ekf, robot), self.error(dr, dr_robot))
        # the map makes the filter more certain, not less
        self.assertLess(ekf.get_Pnorm()[-1], 0.01 * dr.get_Pnorm()[-1])

    def test_map_creation(self) -> None:
        robot = self.robot(covar=None)  # perfect odometry
        map = self.map()
        ekf = EKF(robot=(robot, None), sensor=(self.sensor(robot, map), W), animate=False)
        quiet(ekf.run, T=40)

        seen = list(ekf.landmarks)
        self.assertGreater(len(seen), 5)
        # state is the landmark coordinates only
        self.assertEqual(ekf.x_est.shape, (2 * len(seen),))
        for id in seen:
            with self.subTest(landmark=id):
                self.assertLess(np.linalg.norm(ekf.landmark_x(id) - map[id]), 0.3)

    def test_slam(self) -> None:
        robot = self.robot()
        map = self.map()
        ekf = EKF(robot=(robot, V), P0=P0, sensor=(self.sensor(robot, map), W), animate=False)
        quiet(ekf.run, T=40)

        seen = list(ekf.landmarks)
        self.assertGreater(len(seen), 5)
        # state is the vehicle pose followed by landmark coordinates
        self.assertEqual(ekf.x_est.shape, (3 + 2 * len(seen),))
        self.assertLess(self.error(ekf, robot), 1.0)
        for id in seen:
            with self.subTest(landmark=id):
                self.assertLess(np.linalg.norm(ekf.landmark_x(id) - map[id]), 1.5)

    def test_landmark_bookkeeping(self) -> None:
        robot = self.robot(covar=None)
        map = self.map()
        ekf = EKF(robot=(robot, None), sensor=(self.sensor(robot, map), W), animate=False)
        quiet(ekf.run, T=20)

        orders = sorted(order for order, _ in ekf.landmarks.values())
        self.assertEqual(orders, list(range(len(ekf.landmarks))))
        id = next(iter(ekf.landmarks))
        order, count = ekf.landmark(id)
        self.assertEqual(ekf.landmark_index(id), 2 * order)
        self.assertGreaterEqual(count, 1)
        with self.assertRaises(ValueError):
            ekf.landmark(10**6)

    def test_joseph_and_standard_forms_agree(self) -> None:
        def final(joseph: bool) -> np.ndarray:
            robot = self.robot()
            map = self.map()
            ekf = EKF(
                robot=(robot, V),
                P0=P0,
                map=map,
                sensor=(self.sensor(robot, map), W),
                joseph=joseph,
                animate=False,
            )
            quiet(ekf.run, T=10)
            return ekf.x_est

        nt.assert_allclose(final(True), final(False), atol=1e-3)

    def test_map_must_be_landmarkmap(self) -> None:
        robot = self.robot()
        sensor = self.sensor(robot, self.map())
        with self.assertRaises(TypeError):
            EKF(robot=(robot, V), P0=P0, map="not a map", sensor=(sensor, W))

    def test_history_is_reset_by_init(self) -> None:
        ekf, _ = self.dead_reckoning(T=2)
        self.assertGreater(len(ekf.history), 0)
        ekf.init()
        self.assertEqual(len(ekf.history), 0)

    def test_plots(self) -> None:
        robot = self.robot()
        map = self.map()
        ekf = EKF(robot=(robot, V), P0=P0, sensor=(self.sensor(robot, map), W), animate=False)
        quiet(ekf.run, T=5)
        ekf.plot_xy()
        ekf.plot_ellipse()
        ekf.plot_map()
        plt.close("all")

    def test_plot_error_dead_reckoning(self) -> None:
        ekf, _ = self.dead_reckoning(T=5)
        ekf.plot_error()
        plt.close("all")

    def slam(self, T: float = 5) -> tuple[EKF, Bicycle]:
        robot = self.robot()
        map = self.map()
        ekf = EKF(
            robot=(robot, V), P0=P0, sensor=(self.sensor(robot, map), W), animate=False
        )
        quiet(ekf.run, T=T)
        return ekf, robot

    def test_plot_error_slam(self) -> None:
        # the state vector includes the landmarks, only the pose is compared
        ekf, _ = self.slam()
        ekf.plot_error()
        plt.close("all")

    def test_plot_error_is_against_time(self) -> None:
        ekf, _ = self.dead_reckoning(T=5)
        ekf.plot_error()
        axes = plt.gcf().axes
        self.assertEqual(len(axes), 3)
        for ax in axes:
            line = ax.get_lines()[0]
            nt.assert_allclose(line.get_xdata(), ekf.get_t())
            self.assertEqual(len(line.get_ydata()), len(ekf.history))
        plt.close("all")

    def test_plot_error_error_and_bounds(self) -> None:
        from scipy.stats import norm

        ekf, robot = self.slam()
        ekf.plot_error(confidence=0.95)
        ax = plt.gcf().axes[0]
        # error is truth - estimate, for the x coordinate
        expected = np.array([robot.x_hist[k, 0] - h.xest[0] for k, h in enumerate(ekf.history)])
        nt.assert_allclose(ax.get_lines()[0].get_ydata(), expected)
        # a 95% interval on a scalar quantity is 1.96 sigma
        sigma = np.array([np.sqrt(h.P[0, 0]) for h in ekf.history])
        vertices = ax.patches[0].get_xy()
        upper = vertices[: len(ekf.history), 1]
        nt.assert_allclose(upper, norm.ppf(0.975) * sigma, rtol=1e-9)
        plt.close("all")

    def test_plot_error_bgcolor(self) -> None:
        from matplotlib.colors import to_rgb

        ekf, _ = self.dead_reckoning(T=2)
        ekf.plot_error(bgcolor="g")
        for ax in plt.gcf().axes:
            nt.assert_allclose(ax.patches[0].get_facecolor()[:3], to_rgb("g"))
        plt.close("all")

    def test_plot_error_without_bounds(self) -> None:
        ekf, _ = self.dead_reckoning(T=2)
        ekf.plot_error(confidence=None)
        for ax in plt.gcf().axes:
            self.assertEqual(len(ax.patches), 0)
        plt.close("all")

    def test_plot_error_needs_a_vehicle_estimate(self) -> None:
        robot = self.robot(covar=None)
        map = self.map()
        ekf = EKF(robot=(robot, None), sensor=(self.sensor(robot, map), W), animate=False)
        quiet(ekf.run, T=2)
        with self.assertRaises(ValueError):
            ekf.plot_error()


class TestParticleFilter(unittest.TestCase):
    R = np.diag([0.1, 0.1, np.radians(1)]) ** 2
    L = np.diag([0.1, 0.1])

    def run_filter(self, nparticles: int = 500, T: float = 6, seed: int = 0):
        robot = Bicycle(covar=V, seed=seed)
        robot.control = RandomPath(workspace=10, seed=seed)
        map = LandmarkMap(20, workspace=10, seed=0)
        sensor = RangeBearingSensor(robot, map, covar=W)
        pf = ParticleFilter(
            robot, sensor, self.R, self.L, nparticles=nparticles, seed=seed, animate=False
        )
        quiet(pf.run, T=T)
        return pf, robot

    def test_no_pause_unless_animating(self) -> None:
        # a non-animated run must not sleep, it used to pause 0.2 s per step
        with mock.patch.object(plt, "pause") as pause:
            self.run_filter(nparticles=50, T=1)
        pause.assert_not_called()

    def test_converges_to_the_truth(self) -> None:
        pf, robot = self.run_filter()
        error = np.linalg.norm(pf.get_xyt()[-1, :2] - robot.x[:2])
        self.assertLess(error, 1.0)

    def test_uncertainty_shrinks(self) -> None:
        pf, _ = self.run_filter()
        std = pf.get_std()[:, :2]
        self.assertLess(std[-1].mean(), 0.2 * std[0].mean())

    def test_result_shapes(self) -> None:
        pf, robot = self.run_filter(nparticles=100, T=2)
        steps = round(2 / robot.dt)
        self.assertEqual(pf.get_t().shape, (steps,))
        self.assertEqual(pf.get_xyt().shape, (steps, 3))
        self.assertEqual(pf.get_std().shape, (steps, 3))
        self.assertEqual(pf.x.shape, (100, 3))

    def test_xyt_includes_heading(self) -> None:
        pf, _ = self.run_filter(nparticles=100, T=2)
        xyt = pf.get_xyt()
        # heading is the estimate in the history, within the wrapped range
        nt.assert_array_equal(xyt[:, 2], [h.xest[2] for h in pf.history])
        self.assertTrue(np.all(np.abs(xyt[:, 2]) <= np.pi + 1e-9))

    def test_same_seed_same_estimate(self) -> None:
        a, _ = self.run_filter(nparticles=100, T=2, seed=5)
        b, _ = self.run_filter(nparticles=100, T=2, seed=5)
        nt.assert_array_equal(a.get_xyt(), b.get_xyt())

    def test_particles_stay_in_range(self) -> None:
        pf, _ = self.run_filter(nparticles=100, T=2)
        self.assertTrue(np.all(np.abs(pf.x[:, 2]) <= np.pi + 1e-9))


if __name__ == "__main__":  # pragma nocover
    unittest.main()
