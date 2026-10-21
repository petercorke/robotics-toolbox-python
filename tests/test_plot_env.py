"""
Plot a robot into a scene that an earlier plot made, in another pose.

These use a real Matplotlib (the Agg backend) and read back the line data that was drawn,
so they check where each drawing of the robot is, not just that something was called.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.testing as nt  # noqa: E402
import unittest  # noqa: E402

import roboticstoolbox as rtb  # noqa: E402
from roboticstoolbox import ET2  # noqa: E402
from roboticstoolbox.backends.PyPlot import PyPlot  # noqa: E402
from roboticstoolbox.backends.PyPlot.RobotPlot import RobotPlot  # noqa: E402


def tip(handle) -> np.ndarray:
    """Where the drawing puts the end of its (only) line, and the base is the start."""
    line = handle.links[0]
    return np.array(line.get_data_3d()).T[-1]


def base(handle) -> np.ndarray:
    return np.array(handle.links[0].get_data_3d()).T[0]


class TestSharedScene(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        self.puma = rtb.models.DH.Puma560()
        self.q0 = self.puma.qr
        self.q1 = self.puma.qn

    def tearDown(self):
        plt.close("all")

    def two_plots(self, **kwargs):
        env = self.puma.plot(self.q0, backend="pyplot", block=False, **kwargs)
        env2 = self.puma.plot(
            self.q1, backend="pyplot", env=env, block=False, color="red"
        )
        return env, env2

    def test_the_second_plot_is_added_to_the_scene(self):
        env, env2 = self.two_plots()
        self.assertIs(env2, env)
        self.assertEqual(len(env.robots), 2)
        self.assertTrue(all(isinstance(r, RobotPlot) for r in env.robots))
        # one figure, one set of axes: the first plot is not covered
        self.assertEqual(plt.get_fignums(), [env.fig.number])
        self.assertEqual(len(env.fig.axes), 1)

    def test_without_env_each_plot_is_a_new_scene(self):
        env = self.puma.plot(self.q0, backend="pyplot", block=False)
        env2 = self.puma.plot(self.q1, backend="pyplot", block=False)
        self.assertIsNot(env2, env)
        self.assertEqual(len(env.robots), 1)

    def test_each_drawing_is_at_its_own_pose(self):
        env, _ = self.two_plots()
        first, second = env.robots
        nt.assert_array_almost_equal(tip(first), self.puma.fkine(self.q0).t)
        nt.assert_array_almost_equal(tip(second), self.puma.fkine(self.q1).t)
        nt.assert_array_almost_equal(first.q, self.q0)
        nt.assert_array_almost_equal(second.q, self.q1)
        # they really are different poses
        self.assertGreater(np.linalg.norm(tip(first) - tip(second)), 0.1)

    def test_the_robot_moving_does_not_move_a_drawing(self):
        env, _ = self.two_plots()
        before = [tip(h).copy() for h in env.robots]
        self.puma.q = self.puma.qz
        env.step(0.001)
        for h, b in zip(env.robots, before):
            nt.assert_array_almost_equal(tip(h), b)

    def test_setting_q_of_a_drawing_moves_only_that_drawing(self):
        env, _ = self.two_plots()
        first, second = env.robots
        before_second = tip(second).copy()
        q2 = self.puma.qz
        first.q = q2
        env.step(0.001)
        nt.assert_array_almost_equal(tip(first), self.puma.fkine(q2).t)
        nt.assert_array_almost_equal(tip(second), before_second)
        nt.assert_array_almost_equal(second.q, self.q1)

    def test_the_other_drawing_moves_when_it_is_told(self):
        env, _ = self.two_plots()
        first, second = env.robots
        before_first = tip(first).copy()
        second.q = self.puma.qz
        env.step(0.001)
        nt.assert_array_almost_equal(tip(second), self.puma.fkine(self.puma.qz).t)
        nt.assert_array_almost_equal(tip(first), before_first)

    def test_a_trajectory_is_drawn_over_a_static_plot(self):
        env = self.puma.plot(self.q0, backend="pyplot", block=False)
        first = env.robots[0]
        qt = np.array([self.q0 + (self.q1 - self.q0) * s for s in np.linspace(0, 1, 5)])

        seen = []
        step = env.step
        env.step = lambda dt=0.05: (step(0.001), seen.append(tip(first).copy()))[0]
        env2 = self.puma.plot(qt, env=env, backend="pyplot", block=False, color="red")
        env.step = step

        self.assertIs(env2, env)
        second = env.robots[1]
        nt.assert_array_almost_equal(second.q, self.q1)  # the last pose
        nt.assert_array_almost_equal(tip(second), self.puma.fkine(self.q1).t)
        # the static drawing did not move at any step
        self.assertEqual(len(seen), 5)
        for t in seen:
            nt.assert_array_almost_equal(t, self.puma.fkine(self.q0).t)

    def test_the_robot_is_left_in_the_last_pose(self):
        self.two_plots()
        nt.assert_array_almost_equal(self.puma.q, self.q1)

    def test_a_different_robot_in_the_same_scene(self):
        panda = rtb.models.ETS.Panda()
        env = self.puma.plot(self.q0, backend="pyplot", block=False)
        panda.plot(panda.qr, backend="pyplot", env=env, block=False)
        self.assertEqual(len(env.robots), 2)
        self.assertIs(env.robots[0].robot, self.puma)
        self.assertIs(env.robots[1].robot, panda)
        nt.assert_array_almost_equal(tip(env.robots[0]), self.puma.fkine(self.q0).t)

    def test_the_view_shows_both_robots(self):
        env, _ = self.two_plots()
        lims = [env.ax.get_xlim3d(), env.ax.get_ylim3d(), env.ax.get_zlim3d()]
        for handle in env.robots:
            for line in handle.links:
                for pt in np.array(line.get_data_3d()).T:
                    for value, (lo, hi) in zip(pt, lims):
                        self.assertGreaterEqual(value, lo - 1e-9)
                        self.assertLessEqual(value, hi + 1e-9)

    def test_the_view_is_the_same_scale_on_every_axis(self):
        env, _ = self.two_plots()
        spans = [
            hi - lo
            for lo, hi in (
                env.ax.get_xlim3d(),
                env.ax.get_ylim3d(),
                env.ax.get_zlim3d(),
            )
        ]
        nt.assert_array_almost_equal(spans, [spans[0]] * 3)


class TestColor(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        self.puma = rtb.models.DH.Puma560()

    def tearDown(self):
        plt.close("all")

    def test_color_of_each_robot(self):
        env = self.puma.plot(self.puma.qr, backend="pyplot", block=False, color="gray")
        self.puma.plot(self.puma.qn, backend="pyplot", env=env, block=False, color="red")
        first, second = env.robots
        self.assertEqual(first.links[0].get_color(), "gray")
        self.assertEqual(second.links[0].get_color(), "red")

    def test_default_color_is_unchanged(self):
        env = self.puma.plot(self.puma.qr, backend="pyplot", block=False)
        self.assertEqual(env.robots[0].links[0].get_color(), "#E16F6D")

    def test_color_overrides_options_and_keeps_the_rest(self):
        env = self.puma.plot(
            self.puma.qr,
            backend="pyplot",
            block=False,
            color="blue",
            options={"robot": {"color": "green", "linewidth": 2}},
        )
        line = env.robots[0].links[0]
        self.assertEqual(line.get_color(), "blue")
        self.assertEqual(line.get_linewidth(), 2)

    def test_add_takes_color(self):
        env = PyPlot()
        env.launch()
        handle = env.add(self.puma, readonly=True, color="purple")
        self.assertEqual(handle.links[0].get_color(), "purple")


class TestAddAndHandle(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        self.puma = rtb.models.DH.Puma560()
        self.env = PyPlot()
        self.env.launch()

    def tearDown(self):
        plt.close("all")

    def test_add_returns_the_drawing(self):
        handle = self.env.add(self.puma)
        self.assertIsInstance(handle, RobotPlot)
        self.assertIs(handle.robot, self.puma)

    def test_without_q_the_drawing_follows_the_robot(self):
        handle = self.env.add(self.puma)
        self.puma.q = self.puma.qn
        self.env.step(0.001)
        nt.assert_array_almost_equal(tip(handle), self.puma.fkine(self.puma.qn).t)
        nt.assert_array_almost_equal(handle.q, self.puma.qn)

    def test_with_q_the_drawing_is_pinned(self):
        handle = self.env.add(self.puma, q=self.puma.qr)
        self.puma.q = self.puma.qn
        self.env.step(0.001)
        nt.assert_array_almost_equal(tip(handle), self.puma.fkine(self.puma.qr).t)

    def test_setting_q_pins_the_drawing(self):
        handle = self.env.add(self.puma)
        handle.q = self.puma.qr
        self.puma.q = self.puma.qn
        self.env.step(0.001)
        nt.assert_array_almost_equal(tip(handle), self.puma.fkine(self.puma.qr).t)

    def test_the_pose_is_copied(self):
        q = self.puma.qr.copy()
        handle = self.env.add(self.puma, q=q)
        q[:] = 0
        nt.assert_array_almost_equal(handle.q, self.puma.qr)

    def test_the_same_robot_twice_in_two_poses(self):
        a = self.env.add(self.puma, q=self.puma.qr)
        b = self.env.add(self.puma, q=self.puma.qn)
        self.env.step(0.001)
        nt.assert_array_almost_equal(tip(a), self.puma.fkine(self.puma.qr).t)
        nt.assert_array_almost_equal(tip(b), self.puma.fkine(self.puma.qn).t)

    def test_wrong_length_q(self):
        with self.assertRaises(ValueError):
            self.env.add(self.puma, q=[0, 0, 0])


class TestAxes(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        self.puma = rtb.models.DH.Puma560()

    def tearDown(self):
        plt.close("all")

    def test_plot_on_given_axes(self):
        fig = plt.figure("mine")
        ax = fig.add_subplot(111, projection="3d")
        env = self.puma.plot(self.puma.qr, backend="pyplot", ax=ax, block=False)
        self.assertIs(env.ax, ax)
        self.assertEqual(plt.get_fignums(), [fig.number])
        self.assertEqual(len(fig.axes), 1)
        self.assertEqual(len(ax.lines), 2 + 0)  # the noodle and its shadow

    def test_ax_and_env_are_exclusive(self):
        env = self.puma.plot(self.puma.qr, backend="pyplot", block=False)
        with self.assertRaisesRegex(ValueError, "ax and env"):
            self.puma.plot(self.puma.qn, env=env, ax=env.ax, block=False)

    def test_things_set_when_the_scene_is_made_cannot_be_given_with_env(self):
        env = self.puma.plot(self.puma.qr, backend="pyplot", block=False)
        for kwargs in (
            {"limits": [-1, 1, -1, 1, 0, 1]},
            {"fig": env.fig},
            {"render_mode": "window"},
            {"notebook": True},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, "cannot be used with env"):
                    self.puma.plot(self.puma.qn, env=env, block=False, **kwargs)
        self.assertEqual(len(env.robots), 1)  # nothing was added

    def test_planar_backend_has_no_ax(self):
        planar = rtb.Robot2(ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1), name="planar")
        fig = plt.figure("p")
        ax = fig.add_subplot(111)
        with self.assertRaisesRegex(ValueError, "cannot plot on ax"):
            planar.plot([0.1, 0.2], backend="pyplot2", ax=ax, block=False)


class TestPlanar(unittest.TestCase):
    def setUp(self):
        plt.close("all")
        self.planar = rtb.Robot2(
            ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1), name="planar"
        )

    def tearDown(self):
        plt.close("all")

    def test_two_poses_in_one_scene(self):
        env = self.planar.plot([0.1, 0.2], backend="pyplot2", block=False)
        env2 = self.planar.plot(
            [1.0, -0.5], backend="pyplot2", env=env, block=False
        )
        self.assertIs(env2, env)
        first, second = env.robots
        nt.assert_array_almost_equal(first.q, [0.1, 0.2])
        nt.assert_array_almost_equal(second.q, [1.0, -0.5])
        end1 = np.array(first.links[0].get_data()).T[-1]
        end2 = np.array(second.links[0].get_data()).T[-1]
        nt.assert_array_almost_equal(end1, self.planar.fkine([0.1, 0.2]).t)
        nt.assert_array_almost_equal(end2, self.planar.fkine([1.0, -0.5]).t)
        self.assertEqual(len(env.fig.axes), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
