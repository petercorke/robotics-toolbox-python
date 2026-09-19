"""
Regression tests for URDFRobot's robot_descriptions loading guards.
"""

import sys
import unittest
from unittest.mock import patch

from roboticstoolbox.models.URDF.URDFRobot import URDFRobot, _load_rd_module


class TestURDFRobotGripperLinkName(unittest.TestCase):
    # Regression tests for #578: a raw positional gripper_link_index broke
    # silently when an upstream robot_descriptions update reordered UR5's
    # parsed links. gripper_link_name resolves by link name instead, which
    # survives that kind of reordering.

    def test_resolves_by_name(self):
        robot = URDFRobot(
            "ur5",
            manufacturer="Universal Robotics",
            gripper_link_name=["tool0", "ee_link", "flange"],
        )
        self.assertEqual(robot.n, 6)
        self.assertEqual(len(robot.grippers), 1)
        self.assertIn(robot.grippers[0].name, ("tool0", "ee_link", "flange"))

    def test_falls_through_candidate_list_in_order(self):
        # "tool0" exists on the UR5 URDF; a bogus first candidate should be
        # skipped in favour of it, not raise.
        robot = URDFRobot(
            "ur5",
            manufacturer="Universal Robotics",
            gripper_link_name=["not_a_real_link_name", "tool0"],
        )
        self.assertEqual(robot.grippers[0].name, "tool0")

    def test_raises_when_no_candidate_matches(self):
        with self.assertRaises(ValueError) as cm:
            URDFRobot(
                "ur5",
                manufacturer="Universal Robotics",
                gripper_link_name=["definitely_not_a_real_link_name"],
            )
        self.assertIn("definitely_not_a_real_link_name", str(cm.exception))


class TestURDFRobotEnvironmentGuards(unittest.TestCase):
    def test_pyodide_raises_actionable_error(self):
        # On real Pyodide, robot_descriptions' GitPython-backed clone fails
        # with a plain ImportError ("emscripten does not support
        # processes"), not some other exception type -- simulate that here.
        # A prior version of this guard only checked sys.platform inside
        # `except Exception`, not `except ImportError`, so the loop's
        # `except ImportError: continue` swallowed it and fell through to a
        # misleading "model not found"/"renamed" error instead of this one.
        def fake_import_module(name):
            raise ImportError(f"emscripten does not support processes: {name}")

        with patch.object(sys, "platform", "emscripten"), patch(
            "roboticstoolbox.models.URDF.URDFRobot.importlib.import_module",
            side_effect=fake_import_module,
        ):
            with self.assertRaises(ValueError) as cm:
                _load_rd_module("panda")

        self.assertIn("browser", str(cm.exception))
        self.assertNotIn("is now named", str(cm.exception))
        self.assertNotIn("can not be found", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
