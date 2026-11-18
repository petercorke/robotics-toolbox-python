#!/usr/bin/env python3
"""
Block parameters of the RTB blocks: ``Block.set_param``, ``Block.parameters`` and the
``--set`` option of bdsim.

Every ``__init__`` argument of a block is registered with ``self.add_param(name)``, and
is one of

* ``when="anytime"`` (the default): settable at any time, between evaluations
* ``when="setup"``: settable before the run starts, refused while it runs, since
  ``start()`` has used the value to build something
* ``when="never"``: fixed when the block is created (``robot``, ``model``)

A parameter whose value is not a plain attribute, because the block derived state from
it, is a Python ``property``: the setter does the work and the registration is the same.
"""

import ast
import unittest
import unittest.mock as mock
from pathlib import Path

import bdsim
import numpy as np
import numpy.testing as nt

import roboticstoolbox as rtb
from roboticstoolbox.blocks.quad_model import quadrotor

BLOCKS = Path(rtb.__file__).parent / "blocks"

# Arguments that cannot be block parameters, and why.
NOT_PARAMETERS: dict[str, str] = {
    # a block has a shape of its own, its icon in the diagram
    "VehiclePlot.shape": "the name is taken by Block.shape",
}


def block_parameters() -> dict[str, str]:
    """{Class.param: registered or not} from the source of every block module."""
    found = {}
    for path in sorted(BLOCKS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            init = next(
                (
                    n
                    for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name == "__init__"
                ),
                None,
            )
            if init is None:
                continue
            a = init.args
            params = [
                x.arg
                for x in a.posonlyargs + a.args + a.kwonlyargs
                if x.arg not in ("self", "blockargs", "kwargs")
            ]
            registered = {
                node.args[0].value
                for node in ast.walk(init)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_param"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            }
            # parameters registered by a helper method of the class
            for node in cls.body:
                if isinstance(node, ast.FunctionDef) and node.name != "__init__":
                    for call in ast.walk(node):
                        if (
                            isinstance(call, ast.Call)
                            and isinstance(call.func, ast.Attribute)
                            and call.func.attr == "add_param"
                            and call.args
                            and isinstance(call.args[0], ast.Constant)
                        ):
                            registered.add(call.args[0].value)
            for p in params:
                found[f"{cls.name}.{p}"] = "registered" if p in registered else "none"
    return found


class EveryParameterIsAccountedForTest(unittest.TestCase):
    def test_no_parameter_is_left_out(self):
        left_out = sorted(
            k
            for k, how in block_parameters().items()
            if how == "none" and k not in NOT_PARAMETERS
        )
        self.assertEqual(
            left_out,
            [],
            'register with self.add_param(name) (when="never" if it is fixed)',
        )

    def test_not_parameters_are_real_arguments_that_are_not_registered(self):
        found = block_parameters()
        self.assertEqual(
            sorted(k for k in NOT_PARAMETERS if found.get(k) != "none"),
            [],
            "remove from NOT_PARAMETERS",
        )


class ParamTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = bdsim.BDSim(graphics=None, progress=False, banner=False, quiet=True)
        cls.robot = rtb.models.DH.Puma560()

    def bd(self):
        return self.sim.blockdiagram()


class ParametersTest(ParamTestCase):
    def test_robot_is_fixed(self):
        block = self.bd().FKINE(self.robot)
        with self.assertRaises(ValueError):
            block.set_param("robot", self.robot)
        self.assertNotIn("robot", block.parameters())

    def test_fdyn_q0_is_the_first_half_of_the_initial_state(self):
        block = self.bd().FDYN(self.robot, self.robot.qn)
        nt.assert_array_equal(block.q0, self.robot.qn)
        block.set_param("q0", self.robot.qr)
        nt.assert_array_equal(block.getstate0(), np.r_[self.robot.qr, np.zeros(6)])
        with self.assertRaises(ValueError):
            block.set_param("q0", [1, 2, 3])

    def test_jacobian_frame_and_representation_rebuild_the_function(self):
        block = self.bd().JACOBIAN(self.robot)
        q = self.robot.qn
        nt.assert_array_almost_equal(block.jfunc(q), self.robot.jacob0(q))
        block.set_param("frame", "e")
        nt.assert_array_almost_equal(block.jfunc(q), self.robot.jacobe(q))
        with self.assertRaises(ValueError):
            block.set_param("representation", "rpy/xyz")  # not in the EE frame
        self.assertEqual(block.frame, "e")

    def test_jacobian_inverse_and_pinv_exclude_each_other(self):
        block = self.bd().JACOBIAN(self.robot, inverse=True)
        with self.assertRaises(ValueError):
            block.set_param("pinv", True)
        block.set_param("inverse", False)
        block.set_param("pinv", True)
        self.assertTrue(block.pinv)

    def test_gravity_is_checked_and_can_be_none(self):
        block = self.bd().GRAVLOAD(self.robot)
        self.assertIsNone(block.gravity)
        block.set_param("gravity", [0, 0, -1])
        nt.assert_array_equal(block.gravity, [0, 0, -1])
        with self.assertRaises(ValueError):
            block.set_param("gravity", [1, 2])
        block.set_param("gravity", None)
        self.assertIsNone(block.gravity)

    def test_circlepath_frequency_and_phase_are_in_the_unit_given(self):
        block = self.bd().CIRCLEPATH(frequency=1, phase=0.25)
        self.assertAlmostEqual(block.omega, 2 * np.pi)
        self.assertAlmostEqual(block._phase_rad, np.pi / 2)
        block.set_param("frequency", 2.0)
        self.assertAlmostEqual(block.omega, 4 * np.pi)
        with self.assertRaises(ValueError):
            block.set_param("unit", "rad")  # what the numbers mean is fixed
        radians = self.bd().CIRCLEPATH(frequency=3, unit="rad")
        self.assertAlmostEqual(radians.omega, 3)

    def test_point2tr_T_is_the_pose(self):
        from spatialmath import SE3

        block = self.bd().POINT2TR()
        pose = SE3.Rx(0.3)
        block.set_param("T", pose)
        self.assertIs(block.pose, pose)

    def test_vehicle_limits_reach_the_model(self):
        block = self.bd().BICYCLE(L=2, speed_max=3)
        self.assertEqual(block.L, 2)
        block.set_param("L", 3.0)
        block.set_param("speed_max", 5.0)
        block.set_param("steer_max", 0.3)
        self.assertEqual(block.vehicle.l, 3)
        self.assertEqual(block.vehicle.speed_max, 5)
        self.assertEqual(block.vehicle.steer_max, 0.3)

        block = self.bd().UNICYCLE(w=2)
        block.set_param("w", 4.0)
        self.assertEqual(block.vehicle._W, 4)

    def test_multirotor_x0_accepts_the_short_forms(self):
        block = self.bd().MULTIROTOR(quadrotor)
        block.set_param("x0", [1, 2, 3])
        nt.assert_array_equal(block.getstate0(), np.r_[1, 2, 3, np.zeros(9)])
        block.set_param("x0", np.arange(12.0))
        nt.assert_array_equal(block.getstate0(), np.arange(12.0))
        with self.assertRaises(ValueError):
            block.set_param("x0", [1, 2])

    def test_multirotormixer_speed_limits(self):
        block = self.bd().MULTIROTORMIXER(quadrotor, wmax=800, wmin=10)
        self.assertAlmostEqual(block.wmax, 800)
        block.set_param("wmin", 20.0)
        self.assertAlmostEqual(block.minw, 400)

    def test_vehicleplot_limits_are_normalised(self):
        block = self.bd().VEHICLEPLOT()
        block.set_param("limits", [-5, 5])
        self.assertEqual(block.limits, [-5, 5, -5, 5])
        block.set_param("limits", "auto")
        self.assertEqual(block.limits, "auto")


class RunsTest(ParamTestCase):
    """Parameters that start() reads are used by the next run, not by the first."""

    def test_jtraj_with_no_T_follows_the_length_of_each_run(self):
        bd = self.bd()
        traj = bd.JTRAJ(self.robot.qz, self.robot.qr)
        bd.compile()
        for T in (1.0, 2.0):
            out = self.sim.run(bd, T=T, dt=0.1, watch=[traj[0]])
            nt.assert_array_almost_equal(out.y[-1], self.robot.qr, decimal=3)

    def test_start_resolves_no_T_without_changing_the_parameter(self):
        # a second run, longer than the first, must not keep the first run's length.
        # Exercise parameter resolution directly; compilation and complete runs
        # are covered by test_trajectory_block_compilation.py.
        from types import SimpleNamespace

        from spatialmath import SE3

        cases = {
            "trapezoidal": (self.bd().TRAPEZOIDAL(0, 1), np.array([1.0])),
            "traj": (self.bd().TRAJ(0, 1, time=True), np.array([1.0])),
        }
        for name, (block, final) in cases.items():
            with self.subTest(block=name):
                for T in (1.0, 2.0):
                    block.start(SimpleNamespace(T=T, tf=T))
                    nt.assert_array_almost_equal(
                        np.atleast_1d(block.output(T, [], None)[0]), final, decimal=3
                    )
                    self.assertIsNone(block.T)

        block = self.bd().CTRAJ(SE3(), SE3.Tx(1), None)
        for T in (1.0, 2.0):
            block.start(SimpleNamespace(T=T, tf=T))
            nt.assert_array_almost_equal(block.output(T, [], None)[0].t, [1, 0, 0])
            self.assertIsNone(block.T)

    def test_ikine_runs(self):
        # start() used to take no simstate, which every run passes
        robot = self.robot
        bd = self.bd()
        pose = bd.CONSTANT(robot.fkine(robot.qn))
        ik = bd.IKINE(robot, q0=robot.qn)
        bd.connect(pose, ik)
        bd.compile()
        out = self.sim.run(bd, T=0.2, dt=0.1, watch=[ik[0]])
        nt.assert_array_almost_equal(
            robot.fkine(out.y[-1]).A, robot.fkine(robot.qn).A, decimal=4
        )

    def test_armplot_q0_is_a_setup_parameter(self):
        block = self.bd().ARMPLOT(self.robot)
        block.set_param("q0", self.robot.qn)
        nt.assert_array_equal(block.q0, self.robot.qn)


class TestBdsimVersion(unittest.TestCase):
    """The blocks say so in plain words when bdsim is missing or too old."""

    @staticmethod
    def check(version):
        from roboticstoolbox.blocks import _check_bdsim_version

        with mock.patch("importlib.metadata.version", return_value=version):
            _check_bdsim_version()

    def test_new_enough(self):
        for version in ("1.5.0", "1.5.1", "1.10.0", "2.0", "1.5.0rc1"):
            with self.subTest(version=version):
                self.check(version)

    def test_too_old(self):
        for version in ("1.4.1", "1.0", "0.9.2"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ImportError, rf"1\.5\.0.*{version}"):
                    self.check(version)

    def test_not_installed(self):
        import importlib.metadata

        from roboticstoolbox.blocks import _check_bdsim_version

        with mock.patch(
            "importlib.metadata.version",
            side_effect=importlib.metadata.PackageNotFoundError("bdsim"),
        ):
            with self.assertRaisesRegex(ImportError, r"\[blocks\]"):
                _check_bdsim_version()


class TestPortLabels(unittest.TestCase):
    """bdsim types ``inlabels`` and ``outlabels`` as a list or tuple of names.

    A bare string passes ``len()`` for a one-character name, so it works by accident,
    and would become one port per letter for a longer one."""

    def test_labels_are_sequences_of_names(self):
        import inspect

        import roboticstoolbox.blocks as blocks
        from bdsim.components import Block

        checked = 0
        for module in (blocks.arm, blocks.mobile, blocks.spatial, blocks.uav):
            for name, cls in inspect.getmembers(module, inspect.isclass):
                try:
                    is_block = issubclass(cls, Block)
                except TypeError:
                    # Python 3.10 reports a generic alias such as list[str] as a class
                    continue
                if not is_block or cls.__module__ != module.__name__:
                    continue
                for attr in ("inlabels", "outlabels"):
                    labels = getattr(cls, attr, None)
                    if labels is None:
                        continue
                    checked += 1
                    with self.subTest(block=name, attr=attr):
                        self.assertIsInstance(labels, (list, tuple))
                        self.assertTrue(all(isinstance(s, str) for s in labels))
        self.assertGreater(checked, 20)


if __name__ == "__main__":
    unittest.main()
