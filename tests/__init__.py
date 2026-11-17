import importlib.util

import pytest

from roboticstoolbox.ets.fknm import _C_AVAILABLE as fknm_compiled
from roboticstoolbox.robot.frne import _C_AVAILABLE as frne_compiled


def _available(*packages):
    return all(importlib.util.find_spec(p) is not None for p in packages)


skip_no_collision_checking = pytest.mark.skipif(
    not _available("coal", "trimesh"),
    reason="coal not installed (pip install '.[collision]')",
)

# The C++ solvers (ets.ik_LM() and friends), the compiled handle on an ET and
# the rtbtool self-test exist only in a compiled build, not in the pure-Python
# wheel that JupyterLite and Pyodide use.
requires_compiled = pytest.mark.skipif(
    not (fknm_compiled and frne_compiled),
    reason="needs the compiled extensions (this is a pure-Python build)",
)

skip_no_qp = pytest.mark.skipif(
    not _available("qpsolvers"),
    reason="qpsolvers not installed (see 'qp' extra: pip install .[qp])",
)
