"""
Tests for Phases B-D of the graphics/backend decoupling refactor:
  B - MPL mixin (lazy imports, methods on robot)
  C - capability flags (supports_teach, supports_ellipse) on Connector subclasses
  D - websockets pin relaxed (no <11 constraint in pyproject.toml)
"""

import unittest
import importlib
import sys


class TestConnectorCapabilityDefaults(unittest.TestCase):
    """Connector ABC defines safe defaults for capability flags."""

    def test_connector_supports_teach_default(self):
        from roboticstoolbox.backends.Connector import Connector
        self.assertTrue(Connector.supports_teach)

    def test_connector_supports_ellipse_default(self):
        from roboticstoolbox.backends.Connector import Connector
        self.assertFalse(Connector.supports_ellipse)


class TestPyPlotCapabilities(unittest.TestCase):
    """PyPlot and PyPlot2 opt in to both capabilities."""

    def test_pyplot_supports_teach(self):
        from roboticstoolbox.backends.PyPlot import PyPlot
        self.assertTrue(PyPlot.supports_teach)

    def test_pyplot_supports_ellipse(self):
        from roboticstoolbox.backends.PyPlot import PyPlot
        self.assertTrue(PyPlot.supports_ellipse)

    def test_pyplot2_supports_teach(self):
        from roboticstoolbox.backends.PyPlot import PyPlot2
        self.assertTrue(PyPlot2.supports_teach)

    def test_pyplot2_supports_ellipse(self):
        from roboticstoolbox.backends.PyPlot import PyPlot2
        self.assertTrue(PyPlot2.supports_ellipse)

    def test_pyplot_instance_supports_teach(self):
        from roboticstoolbox.backends.PyPlot import PyPlot
        env = PyPlot.__new__(PyPlot)
        self.assertTrue(env.supports_teach)

    def test_pyplot_instance_supports_ellipse(self):
        from roboticstoolbox.backends.PyPlot import PyPlot
        env = PyPlot.__new__(PyPlot)
        self.assertTrue(env.supports_ellipse)


class TestSwiftCapabilities(unittest.TestCase):
    """RTB Swift wrapper opts out of teach and ellipse."""

    def setUp(self):
        # Skip if swift-sim is not installed
        try:
            from roboticstoolbox.backends.swift import Swift  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            self.skipTest("swift-sim not installed")

    def test_swift_class_supports_teach_false(self):
        from roboticstoolbox.backends.swift import Swift
        self.assertFalse(Swift.supports_teach)

    def test_swift_class_supports_ellipse_false(self):
        from roboticstoolbox.backends.swift import Swift
        self.assertFalse(Swift.supports_ellipse)

    def test_load_backend_swift_supports_teach_false(self):
        from roboticstoolbox.backends import load_backend
        env = load_backend("swift")
        self.assertFalse(env.supports_teach)

    def test_load_backend_swift_supports_ellipse_false(self):
        from roboticstoolbox.backends import load_backend
        env = load_backend("swift")
        self.assertFalse(env.supports_ellipse)


class TestMPLMixinOnRobot(unittest.TestCase):
    """RobotPlottingMPLMixin methods are available on robot instances."""

    def _make_robot(self):
        import roboticstoolbox as rtb
        return rtb.models.DH.Puma560()

    def test_robot_has_fellipse(self):
        robot = self._make_robot()
        self.assertTrue(hasattr(robot, "fellipse"))

    def test_robot_has_vellipse(self):
        robot = self._make_robot()
        self.assertTrue(hasattr(robot, "vellipse"))

    def test_robot_has_plot_ellipse(self):
        robot = self._make_robot()
        self.assertTrue(hasattr(robot, "plot_ellipse"))

    def test_robot_has_plot_fellipse(self):
        robot = self._make_robot()
        self.assertTrue(hasattr(robot, "plot_fellipse"))

    def test_robot_has_plot_vellipse(self):
        robot = self._make_robot()
        self.assertTrue(hasattr(robot, "plot_vellipse"))

    def test_robot_has_linkcolormap(self):
        robot = self._make_robot()
        self.assertTrue(hasattr(robot, "linkcolormap"))

    def test_mixin_present_in_mro(self):
        import roboticstoolbox as rtb
        from roboticstoolbox.robot.RobotPlottingMPL import RobotPlottingMPLMixin
        mro_names = [c.__name__ for c in type(rtb.models.DH.Puma560()).__mro__]
        self.assertIn("RobotPlottingMPLMixin", mro_names)


class TestMaybeAddEllipseCapabilityGate(unittest.TestCase):
    """_maybe_add_ellipse_to_active_env skips backends without supports_ellipse."""

    def _make_robot(self):
        import roboticstoolbox as rtb
        return rtb.models.DH.Puma560()

    def test_no_active_env_is_noop(self):
        robot = self._make_robot()
        robot._active_plot_env = None
        robot._maybe_add_ellipse_to_active_env(object())  # should not raise

    def test_non_supporting_env_skipped(self):
        robot = self._make_robot()

        class FakeEnv:
            supports_ellipse = False
            add_called = False

            def add(self, _):
                self.add_called = True

        env = FakeEnv()
        robot._active_plot_env = env
        robot._maybe_add_ellipse_to_active_env(object())
        self.assertFalse(env.add_called)

    def test_supporting_env_receives_ellipse(self):
        robot = self._make_robot()

        class FakeEnv:
            supports_ellipse = True
            received = None

            def add(self, obj):
                self.received = obj

        env = FakeEnv()
        robot._active_plot_env = env
        sentinel = object()
        robot._maybe_add_ellipse_to_active_env(sentinel)
        self.assertIs(env.received, sentinel)


class TestBaseRobotNoModuleLevelMPLImport(unittest.TestCase):
    """BaseRobot module must not import matplotlib or PyPlot at module level."""

    def test_baserobot_module_has_no_pyplot_import(self):
        import ast
        import os

        base_robot_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "roboticstoolbox",
            "robot",
            "BaseRobot.py",
        )
        with open(os.path.normpath(base_robot_path)) as f:
            source = f.read()

        tree = ast.parse(source)
        top_level_imports = [
            node for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and not isinstance(
                next(
                    (
                        p for p in ast.walk(tree)
                        if isinstance(p, ast.FunctionDef) and any(
                            isinstance(c, (ast.Import, ast.ImportFrom))
                            and c is node
                            for c in ast.walk(p)
                        )
                    ),
                    None,
                ),
                ast.FunctionDef,
            )
        ]
        # Simpler check: no "from matplotlib" at module level
        module_body_imports = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        pyplot_imports = [
            node for node in module_body_imports
            if isinstance(node, ast.ImportFrom)
            and "PyPlot" in (node.module or "")
            and "EllipsePlot" not in (node.module or "")
        ]
        # We should only import RobotPlottingMPLMixin, not PyPlot/PyPlot2 directly
        self.assertEqual(
            pyplot_imports, [],
            "BaseRobot.py should not import PyPlot at module level",
        )

    def test_baserobot_module_no_matplotlib_top_level(self):
        import ast
        import os

        base_robot_path = os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "src",
                "roboticstoolbox",
                "robot",
                "BaseRobot.py",
            )
        )
        with open(base_robot_path) as f:
            source = f.read()

        tree = ast.parse(source)
        mpl_top_imports = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            and "matplotlib" in (
                getattr(node, "module", None) or
                " ".join(a.name for a in getattr(node, "names", []))
            )
        ]
        self.assertEqual(
            mpl_top_imports, [],
            "BaseRobot.py should not import matplotlib at module level",
        )


class TestWebsocketsPin(unittest.TestCase):
    """Phase D: pyproject.toml must not pin websockets to <11."""

    def test_websockets_pin_relaxed(self):
        import os
        toml_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        )
        with open(toml_path) as f:
            content = f.read()
        self.assertNotIn(
            "websockets<11",
            content,
            "websockets must not be pinned to <11; should be unpinned",
        )


if __name__ == "__main__":
    unittest.main()
