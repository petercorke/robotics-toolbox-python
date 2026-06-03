"""
Shared pytest skip-marks for optional dependencies.

Import in test files as:
    from tests.marks import skip_no_pybullet, skip_no_qp
"""

import pytest
from importlib import import_module


def _available(*packages):
    for pkg in packages:
        try:
            import_module(pkg)
        except ImportError:
            return False
    return True


skip_no_pybullet = pytest.mark.skipif(
    not _available("pybullet"),
    reason="pybullet not installed (see 'collision' extra; Apple Silicon: make install-collision)",
)

skip_no_qp = pytest.mark.skipif(
    not _available("qpsolvers"),
    reason="qpsolvers not installed (see 'qp' extra: pip install .[qp])",
)
