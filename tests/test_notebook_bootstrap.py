"""Tests for the notebook bootstrap-cell generator and output-clearing scripts.

These exercise the pure notebook-JSON manipulation logic in
docs/notebooks/sync_bootstrap.py and docs/notebooks/clear_outputs.py directly, not
via subprocess -- what can't be tested this way (whether the generated cell's
*content* actually installs correctly in Colab/JupyterLite/locally) needs a real
run in each of those environments, which is a separate, non-pytest concern.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "docs" / "notebooks"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sync_bootstrap = _load_module("_sync_bootstrap", NOTEBOOKS_DIR / "sync_bootstrap.py")
clear_outputs = _load_module("_clear_outputs", NOTEBOOKS_DIR / "clear_outputs.py")


def _write_notebook(path: Path, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")


_ids = iter(f"{n:08x}" for n in range(10**6))


def _code_cell(source: str, outputs=None, execution_count=None) -> dict:
    return {
        "id": next(_ids),
        "cell_type": "code",
        "metadata": {},
        "execution_count": execution_count,
        "outputs": outputs or [],
        "source": source.splitlines(keepends=True),
    }


# --- sync_bootstrap.py -------------------------------------------------------


def test_is_bootstrap_cell_detects_marker():
    marked = _code_cell(f"{sync_bootstrap.MARKER}\nprint('hi')\n")
    unmarked = _code_cell("print('hi')\n")
    markdown = {"cell_type": "markdown", "source": [f"{sync_bootstrap.MARKER}\n"]}

    assert sync_bootstrap.is_bootstrap_cell(marked)
    assert not sync_bootstrap.is_bootstrap_cell(unmarked)
    assert not sync_bootstrap.is_bootstrap_cell(markdown)


def test_generated_source_embeds_template_and_call():
    template = "async def ensure_installed():\n    return False\n"
    lines = sync_bootstrap.generated_source(template)
    text = "".join(lines)

    assert text.startswith(sync_bootstrap.MARKER)
    assert "async def ensure_installed():" in text
    assert text.rstrip().endswith("COLAB = await ensure_installed()")


def test_process_notebook_check_mode_reports_without_writing(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    stale_cell = _code_cell(f"{sync_bootstrap.MARKER}\n# stale content\n")
    _write_notebook(nb_path, [stale_cell])
    before = nb_path.read_text(encoding="utf-8")
    template = "async def ensure_installed():\n    return False\n"

    changed = sync_bootstrap.process_notebook(nb_path, template=template, fix=False)

    assert changed is True
    assert nb_path.read_text(encoding="utf-8") == before  # untouched


def test_process_notebook_fix_mode_rewrites_and_is_idempotent(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    stale_cell = _code_cell(f"{sync_bootstrap.MARKER}\n# stale content\n")
    _write_notebook(nb_path, [stale_cell])
    template = "async def ensure_installed():\n    return False\n"

    changed = sync_bootstrap.process_notebook(nb_path, template=template, fix=True)
    assert changed is True

    # A second pass over the now-regenerated file should find nothing to do.
    changed_again = sync_bootstrap.process_notebook(nb_path, template=template, fix=True)
    assert changed_again is False

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell_text = "".join(nb["cells"][0]["source"])
    assert "async def ensure_installed():" in cell_text


def test_process_notebook_without_marker_cell_is_a_no_op(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    _write_notebook(nb_path, [_code_cell("import numpy as np\n")])

    changed = sync_bootstrap.process_notebook(nb_path, template="whatever", fix=True)

    assert changed is False


# --- clear_outputs.py ---------------------------------------------------------


def test_clear_cell_strips_output_and_execution_count():
    cell = _code_cell("print(1)\n", outputs=[{"output_type": "stream", "text": "1\n"}], execution_count=3)

    changed = clear_outputs.clear_cell(cell)

    assert changed is True
    assert cell["outputs"] == []
    assert cell["execution_count"] is None


def test_clear_cell_leaves_clean_cell_untouched():
    cell = _code_cell("print(1)\n")

    changed = clear_outputs.clear_cell(cell)

    assert changed is False


def test_clear_cell_ignores_markdown_cells():
    cell = {"cell_type": "markdown", "source": ["# Title\n"]}

    assert clear_outputs.clear_cell(cell) is False


def test_process_notebook_fix_mode_clears_and_is_idempotent(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    dirty_cell = _code_cell("print(1)\n", outputs=[{"output_type": "stream", "text": "1\n"}], execution_count=3)
    _write_notebook(nb_path, [dirty_cell])

    changed = clear_outputs.process_notebook(nb_path, fix=True)
    assert changed is True

    changed_again = clear_outputs.process_notebook(nb_path, fix=True)
    assert changed_again is False

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["cells"][0]["outputs"] == []
    assert nb["cells"][0]["execution_count"] is None


def test_process_notebook_check_mode_does_not_write(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    dirty_cell = _code_cell("print(1)\n", outputs=[{"output_type": "stream", "text": "1\n"}], execution_count=3)
    _write_notebook(nb_path, [dirty_cell])
    before = nb_path.read_text(encoding="utf-8")

    changed = clear_outputs.process_notebook(nb_path, fix=False)

    assert changed is True
    assert nb_path.read_text(encoding="utf-8") == before


def test_process_notebook_clears_every_dirty_cell(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    dirty = [
        _code_cell("print(1)\n", outputs=[{"output_type": "stream", "text": "1\n"}], execution_count=n)
        for n in (1, 2, 3)
    ]
    _write_notebook(nb_path, dirty)

    clear_outputs.process_notebook(nb_path, fix=True)

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in nb["cells"])


def test_bootstrap_cell_is_found_wherever_it_is(tmp_path):
    # the cell is located by its marker line, not by its index in the notebook
    template = "async def ensure_installed():\n    return False\n"
    for position in (0, 1, 3):
        cells = [_code_cell(f"x = {i}\n") for i in range(4)]
        cells.insert(position, _code_cell(f"{sync_bootstrap.MARKER}\n# stale\n"))
        nb_path = tmp_path / f"demo{position}.ipynb"
        _write_notebook(nb_path, cells)

        assert sync_bootstrap.process_notebook(nb_path, template=template, fix=True)

        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        text = "".join(nb["cells"][position]["source"])
        assert "async def ensure_installed():" in text
        # the other cells are untouched
        assert sum(sync_bootstrap.is_bootstrap_cell(c) for c in nb["cells"]) == 1


def test_extras_come_from_the_notebook_metadata(tmp_path):
    template = "async def ensure_installed(extras=()):\n    return False\n"
    nb_path = tmp_path / "demo.ipynb"
    nb = {
        "cells": [_code_cell(f"{sync_bootstrap.MARKER}\n# stale\n")],
        "metadata": {"rtb_bootstrap": {"extras": ["ik"]}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path.write_text(json.dumps(nb), encoding="utf-8")

    assert sync_bootstrap.process_notebook(nb_path, template=template, fix=True)

    text = "".join(json.loads(nb_path.read_text(encoding="utf-8"))["cells"][0]["source"])
    assert text.rstrip().endswith("COLAB = await ensure_installed(extras=['ik'])")
    # a second pass finds nothing to do, and removing the metadata is noticed
    assert not sync_bootstrap.process_notebook(nb_path, template=template, fix=True)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    del nb["metadata"]["rtb_bootstrap"]
    nb_path.write_text(json.dumps(nb), encoding="utf-8")
    assert sync_bootstrap.process_notebook(nb_path, template=template, fix=False)


def test_no_extras_gives_the_plain_call():
    lines = sync_bootstrap.generated_source("async def ensure_installed(): ...\n")
    assert "".join(lines).rstrip().endswith("COLAB = await ensure_installed()")


def test_cells_without_ids_get_unique_ids_and_nbformat_5(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    cells = [_code_cell("x = 1\n"), _code_cell("y = 2\n"), _code_cell("z = 3\n")]
    for cell in cells[:2]:
        del cell["id"]
    nb = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
    nb_path.write_text(json.dumps(nb), encoding="utf-8")
    kept = cells[2]["id"]

    # check mode reports the missing ids and does not write
    assert sync_bootstrap.process_notebook(nb_path, template="", fix=False)
    assert "id" not in json.loads(nb_path.read_text(encoding="utf-8"))["cells"][0]

    assert sync_bootstrap.process_notebook(nb_path, template="", fix=True)
    out = json.loads(nb_path.read_text(encoding="utf-8"))
    ids = [c["id"] for c in out["cells"]]
    assert len(set(ids)) == 3 and ids[2] == kept  # an existing id is kept
    assert out["nbformat_minor"] == 5

    # nothing more to do, and the check passes
    assert not sync_bootstrap.process_notebook(nb_path, template="", fix=False)


def test_markdown_cells_get_ids_too(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    nb = {
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# Title\n"]}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path.write_text(json.dumps(nb), encoding="utf-8")

    assert sync_bootstrap.process_notebook(nb_path, template="", fix=True)

    assert "id" in json.loads(nb_path.read_text(encoding="utf-8"))["cells"][0]


def test_editor_metadata_is_normalised(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    nb = {
        "cells": [_code_cell("x = 1\n")],
        "metadata": {
            "kernelspec": {"display_name": "dev", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.14"},
            "vscode": {"interpreter": {"hash": "abc"}},
            "varInspector": {"window_display": False},
            "rtb_bootstrap": {"extras": ["ik"]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path.write_text(json.dumps(nb), encoding="utf-8")

    assert clear_outputs.process_notebook(nb_path, fix=False)  # reported, not written
    assert clear_outputs.process_notebook(nb_path, fix=True)

    out = json.loads(nb_path.read_text(encoding="utf-8"))
    assert out["metadata"] == {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "rtb_bootstrap": {"extras": ["ik"]},  # ours, kept
    }
    assert not clear_outputs.process_notebook(nb_path, fix=True)


def test_notebooks_are_written_as_nbformat_writes_them(tmp_path):
    # so that opening and saving a notebook in VS Code or Jupyter changes nothing
    nbformat = pytest.importorskip("nbformat")

    nb_path = tmp_path / "demo.ipynb"
    _write_notebook(nb_path, [_code_cell("x = 1\n")])
    clear_outputs.process_notebook(nb_path, fix=True)

    text = nb_path.read_text(encoding="utf-8")
    assert text == nbformat.writes(nbformat.reads(text, as_version=4)) + "\n"


# --- pre_commit.py ------------------------------------------------------------

pre_commit = _load_module("_pre_commit", NOTEBOOKS_DIR / "pre_commit.py")


def test_hook_tidies_a_notebook(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    dirty = _code_cell("print(1)\n", outputs=[{"output_type": "stream", "text": "1\n"}], execution_count=3)
    _write_notebook(nb_path, [dirty])

    assert pre_commit.tidy([nb_path]) == [nb_path]

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["cells"][0]["outputs"] == []
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert pre_commit.tidy([nb_path]) == []  # nothing left to do


def test_hook_refuses_a_notebook_with_conflict_markers(tmp_path):
    nb_path = tmp_path / "demo.ipynb"
    nb_path.write_text('{\n<<<<<<< HEAD\n}\n=======\n}\n>>>>>>> main\n', encoding="utf-8")

    with pytest.raises(SystemExit, match="merge conflict"):
        pre_commit.tidy([nb_path])


def test_install_writes_the_hook_and_leaves_another_alone(tmp_path):
    hooks = tmp_path / "hooks"

    assert "installed" in pre_commit.install(hooks)
    assert (hooks / "pre-commit").read_text(encoding="utf-8") == pre_commit.STUB
    assert "installed" in pre_commit.install(hooks)  # again is fine

    (hooks / "pre-commit").write_text("#!/bin/sh\necho mine\n", encoding="utf-8")
    assert "left alone" in pre_commit.install(hooks)
    assert "mine" in (hooks / "pre-commit").read_text(encoding="utf-8")


def test_the_browser_install_goes_through_piplite():
    # micropip cannot see a JupyterLite site's own package index, and a /pypi folder
    # does not exist in the kernel, so the wheel built with the site would never be used
    template = (NOTEBOOKS_DIR / "_rtb_nb_bootstrap.py").read_text(encoding="utf-8")

    assert "import piplite" in template
    assert '"/pypi"' not in template
