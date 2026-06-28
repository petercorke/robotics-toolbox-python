import importlib.util

import pytest


def _available(*packages):
    return all(importlib.util.find_spec(p) is not None for p in packages)


skip_no_pybullet = pytest.mark.skipif(
    not _available("pybullet"),
    reason="pybullet not installed (see 'collision' extra; Apple Silicon: make install-collision)",
)

skip_no_qp = pytest.mark.skipif(
    not _available("qpsolvers"),
    reason="qpsolvers not installed (see 'qp' extra: pip install .[qp])",
)
