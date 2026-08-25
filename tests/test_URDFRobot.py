"""
Regression tests for URDFRobot's robot_descriptions loading guards.
"""

import sys
import unittest
from unittest.mock import patch

from roboticstoolbox.models.URDF.URDFRobot import _load_rd_module


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
