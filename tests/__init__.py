import importlib.util
import sys

import pytest


def _available(*packages):
    return all(importlib.util.find_spec(p) is not None for p in packages)


skip_no_collision_checking = pytest.mark.skipif(
    not _available("coal", "trimesh"),
    reason="coal not installed (pip install '.[collision]')",
)

skip_no_qp = pytest.mark.skipif(
    not _available("qpsolvers"),
    reason="qpsolvers not installed (see 'qp' extra: pip install .[qp])",
)

# Pyodide/JupyterLite is a real, supported target (see rne.md /
# tech-debt.md's pure-wheel work) but is a genuine sandbox: no subprocess
# execution, no git binary, no display/GUI backend. Tests that inherently
# need one of those aren't testing a bug -- they're testing something this
# environment cannot do by construction. Matches the same sys.platform ==
# "emscripten" check already used in product code (URDFRobot.py,
# CollisionShape.py) for the identical reason.
skip_on_pyodide = pytest.mark.skipif(
    sys.platform == "emscripten",
    reason="not supported in the Pyodide/JupyterLite sandbox (no subprocess/"
    "git/display) -- an environment limitation, not a bug",
)
