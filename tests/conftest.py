"""
Pytest configuration for the RTB test suite.

- Forces matplotlib into non-interactive Agg backend so no windows pop up
  during a local run (CI sets this via the MPLBACKEND env var instead).
- Registers the skip_no_collision_checking/skip_no_qp marks (defined in
  tests/__init__.py, imported as `from tests import ...`) so pytest
  recognizes them instead of warning PytestUnknownMarkWarning.
- Records browser limitations and known failures when running in real Pyodide.
"""

import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # must be set before any pyplot import


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "skip_no_collision_checking: skip test if coal/trimesh are not installed",
    )
    config.addinivalue_line(
        "markers",
        "skip_no_qp: skip test if qpsolvers/quadprog are not installed",
    )


# Keep these rules together so a browser-only limitation has one stated reason,
# and desktop runs collect and execute exactly the same tests as before.
_BDSIM_REASON = "bdsim is not installed by the notebook bootstrap"
_PYODIDE_SKIPS = [
    (
        "Pyodide cannot start subprocesses",
        (
            "tests/test_bin.py",
            "tests/test_demo.py",
            "tests/test_Robot.py::TestRobot::test_showgraph",
        ),
    ),
    (
        "robot_descriptions URDF assets need git, unavailable in Pyodide",
        (
            "tests/test_BaseRobot.py::TestBaseRobot::test_ets_gripper2",
            "tests/test_BaseRobot.py::TestBaseRobot::test_init",
            "tests/test_BaseRobot.py::TestBaseRobot::test_init9",
            "tests/test_BaseRobot.py::TestBaseRobot::test_limits",
            "tests/test_BaseRobot.py::TestBaseRobot::test_segments2",
            "tests/test_BaseRobot.py::TestBaseRobot::test_str",
            "tests/test_BaseRobot.py::TestBaseRobot::test_yumi_gripper_parents",
            "tests/test_CustomXacro.py::TestCustomXacro::test_custom",
            "tests/test_IK.py::TestIK::test_IK_GN3",
            "tests/test_IK.py::TestIK::test_ik_lm_failure_returns_compact_q",
            "tests/test_Robot.py::TestRobot::test_fkine_all2",
            "tests/test_Robot.py::TestRobot::test_fkine_compact_q_branched_robot",
            "tests/test_URDFRobot.py::TestURDFRobotGripperLinkName",
            "tests/test_fknm_fallback.py::TestRobotLinkTReference::test_gripper_without_joints",
            "tests/test_models.py::TestModels::test_Jaco",
            "tests/test_models.py::TestModels::test_PandaURDF_robot_descriptions",
            "tests/test_models.py::TestModels::test_UR10",
            "tests/test_models.py::TestModels::test_UR3",
            "tests/test_models.py::TestModels::test_UR5",
            "tests/test_models.py::TestModels::test_pr2",
            "tests/test_models.py::TestModelSmoke::test_all_models_construct",
            "tests/test_urdf_model.py::TestURDF::test_urdf_load",
        ),
    ),
    (
        "the teach panel refuses to run under Jupyter/Pyodide",
        (
            "tests/test_BaseRobot.py::TestBaseRobot::test_teach",
            "tests/test_DHRobot.py::TestDHRobot::test_teach",
            "tests/test_DHRobot.py::TestDHRobot::test_teach_basic",
            "tests/test_DHRobot.py::TestDHRobot::test_teach_withq",
            "tests/test_ETS.py::TestETS::test_teach",
            "tests/test_plot_view.py::TestTeach",
        ),
    ),
    (
        _BDSIM_REASON,
        ("tests/test_quintic_conditioning.py::test_bdsim_quintic_block_long_duration",),
    ),
    (
        "window-mode expectation: Pyodide uses notebook-inline rendering",
        (
            "tests/test_plot_env.py::TestSharedScene::test_the_second_plot_is_added_to_the_scene",
            "tests/test_plot_env.py::TestAxes::test_plot_on_given_axes",
            "tests/test_render_mode.py::test_other_environments_update_one_inline_display",
            "tests/test_step_dt.py::test_3d_step_needs_a_positive_dt[0]",
            "tests/test_step_dt.py::test_3d_step_needs_a_positive_dt[0.0]",
            "tests/test_step_dt.py::test_2d_step_needs_a_positive_dt[0]",
            "tests/test_step_dt.py::test_2d_step_needs_a_positive_dt[0.0]",
            "tests/test_step_dt.py::test_plot_with_a_zero_dt_raises_and_does_not_hang",
        ),
    ),
    (
        "collision checking is disabled in Pyodide before checking for coal",
        (
            "tests/test_collision.py::TestEnvironmentGuards::test_missing_coal_raises_import_error",
        ),
    ),
]

# These are real findings, not unsupported features.  Keep running them so a
# separate fix shows up as XPASS without making this advisory job fail.
_PYODIDE_XFAILS = [
    (
        "movie getframe draws the wrong canvas before reading its renderer",
        ("tests/test_PyPlot.py::TestPyPlot::test_plot_movie",),
    ),
    (
        "path byte hashes assume int64 but Pyodide paths use int32",
        (
            "tests/test_planner_cleanup.py::TestDistanceTransformQuery::test_book_paths_are_unchanged",
        ),
    ),
    (
        "notebook-inline step accepts a negative dt without validation",
        (
            "tests/test_step_dt.py::test_3d_step_needs_a_positive_dt[-0.05]",
            "tests/test_step_dt.py::test_2d_step_needs_a_positive_dt[-0.05]",
        ),
    ),
    (
        "emscripten backend selection overrides the configured default backend",
        (
            "tests/test_default_backend.py::test_a_default_given_in_upper_case_selects_the_backend",
            "tests/test_default_backend.py::test_a_registered_backend_can_be_the_default",
        ),
    ),
]


class _PyodideBdsimModule(pytest.Module):
    def collect(self) -> list[pytest.Item | pytest.Collector]:
        # These modules import bdsim at module scope, before item marks can help.
        # Skipping the collector reports a skip, rather than hiding the module.
        pytest.skip(_BDSIM_REASON)


def pytest_pycollect_makemodule(
    module_path: Path, parent: pytest.Collector
) -> pytest.Module | None:
    if sys.platform == "emscripten" and module_path.name in (
        "test_block_parameters.py",
        "test_trapezoidal_profile.py",
    ):
        return _PyodideBdsimModule.from_parent(parent, path=module_path)
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if sys.platform != "emscripten":
        return

    for rules, mark in (
        (_PYODIDE_SKIPS, pytest.mark.skip),
        (_PYODIDE_XFAILS, pytest.mark.xfail(strict=False)),
    ):
        for reason, prefixes in rules:
            for item in items:
                # Match a module, class, test or parameter case without also
                # matching similarly named tests (test_init versus test_init2).
                if any(
                    item.nodeid == prefix
                    or item.nodeid.startswith(prefix + "::")
                    or item.nodeid.startswith(prefix + "[")
                    for prefix in prefixes
                ):
                    item.add_marker(mark(reason=reason))
