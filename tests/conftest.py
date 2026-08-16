"""
Pytest configuration for the RTB test suite.

- Forces matplotlib into non-interactive Agg backend so no windows pop up
  during a local run (CI sets this via the MPLBACKEND env var instead).
- Registers the skip_no_collision_checking/skip_no_qp marks (defined in
  tests/__init__.py, imported as `from tests import ...`) so pytest
  recognizes them instead of warning PytestUnknownMarkWarning.
"""

import matplotlib

matplotlib.use("Agg")  # must be set before any pyplot import


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_no_collision_checking: skip test if coal/trimesh are not installed",
    )
    config.addinivalue_line(
        "markers",
        "skip_no_qp: skip test if qpsolvers/quadprog are not installed",
    )
