"""
Smoke tests: syntax-check all example scripts.

py_compile catches syntax errors and bad top-level name bindings without
executing any GUI or simulation code.

Notebook tests are handled separately via `make test-notebooks`, which runs
pytest --nbmake directly on docs/notebooks/ (excluding Untitled* scratch files).
"""

import pathlib
import py_compile

import pytest

# ---------------------------------------------------------------------------
# Examples — syntax check only (many examples open visualisers when run)
# ---------------------------------------------------------------------------

_EXAMPLES_DIR = pathlib.Path(__file__).parent.parent / "examples"
_example_files = sorted(_EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("path", _example_files, ids=[p.name for p in _example_files])
def test_example_syntax(path):
    """All example scripts must at least parse without error."""
    py_compile.compile(str(path), doraise=True)
