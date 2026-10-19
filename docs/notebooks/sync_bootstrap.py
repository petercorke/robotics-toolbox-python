#!/usr/bin/env python
"""Regenerate the bootstrap cell in every notebook from _rtb_nb_bootstrap.py.

Every notebook in this folder carries its own copy of the environment-bootstrap
logic, since Google Colab only fetches the single .ipynb file it's opened from and
can't see sibling files. That copy is machine-generated from _rtb_nb_bootstrap.py,
the single source of truth -- see README.md for the full explanation.

Usage:
    python sync_bootstrap.py            regenerate any notebook that's out of date
    python sync_bootstrap.py --check    report only, exit 1 if anything's out of date
    python sync_bootstrap.py FILE ...   only consider the given notebook(s)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MARKER = "# RTB_BOOTSTRAP_CELL"
GENERATED_HEADER = (
    f"{MARKER} -- sets up the environment (Colab / JupyterLite / local install); "
    "click to expand. Generated from docs/notebooks/_rtb_nb_bootstrap.py by "
    "sync_bootstrap.py -- do not hand-edit.\n"
)
# Folds the cell's code by default (VS Code and JupyterLab both honour this) --
# a reader sees the one-line header above and the printed status line below,
# not the boilerplate, unless they choose to expand it.
FOLDED_METADATA = {"jupyter": {"source_hidden": True}}

HERE = Path(__file__).resolve().parent
TEMPLATE_PATH = HERE / "_rtb_nb_bootstrap.py"


def generated_source(template: str, extras: list[str] | tuple[str, ...] = ()) -> list[str]:
    """Build the full source (as nbformat-style lines) for the bootstrap cell.

    :param extras: names of the toolbox extras the notebook needs, for example
        ``["ik"]``; passed on to ``ensure_installed``.
    """
    call = f"ensure_installed(extras={list(extras)!r})" if extras else "ensure_installed()"
    body = f"{GENERATED_HEADER}\n{template.rstrip()}\n\nCOLAB = await {call}\n"
    lines = body.splitlines(keepends=True)
    return lines


def notebook_extras(nb: dict) -> list[str]:
    """The extras a notebook asks for, from ``metadata["rtb_bootstrap"]["extras"]``."""
    return list(nb.get("metadata", {}).get("rtb_bootstrap", {}).get("extras", []))


def cell_source_text(cell: dict) -> str:
    source = cell.get("source", [])
    return "".join(source) if isinstance(source, list) else source


def is_bootstrap_cell(cell: dict) -> bool:
    if cell.get("cell_type") != "code":
        return False
    text = cell_source_text(cell)
    first_line = text.splitlines()[0].strip() if text.strip() else ""
    return first_line.startswith(MARKER)


def process_notebook(path: Path, template: str, fix: bool) -> bool:
    """Check (and optionally fix) one notebook's bootstrap cell.

    :returns: True if the notebook's bootstrap cell was (or would be) changed.
    """
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb.get("cells", []):
        if not is_bootstrap_cell(cell):
            continue

        desired_source = generated_source(template, notebook_extras(nb))
        current_source = cell.get("source", [])
        if not isinstance(current_source, list):
            current_source = [current_source]

        current_jupyter_metadata = cell.get("metadata", {}).get("jupyter", {})
        desired_jupyter_metadata = FOLDED_METADATA["jupyter"]

        if current_source != desired_source or current_jupyter_metadata != desired_jupyter_metadata:
            changed = True
            if fix:
                cell["source"] = desired_source
                cell.setdefault("metadata", {})["jupyter"] = dict(desired_jupyter_metadata)
        break

    if changed and fix:
        with path.open("w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")

    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="report out-of-date notebooks without modifying them; exit 1 if any found",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="only consider these notebooks (default: every *.ipynb in this folder)",
    )
    args = parser.parse_args()

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    fix = not args.check
    notebooks = [p.resolve() for p in args.files] if args.files else sorted(HERE.glob("*.ipynb"))

    any_changed = False
    for nb_path in notebooks:
        changed = process_notebook(nb_path, template, fix=fix)
        if changed:
            any_changed = True
            rel = nb_path.relative_to(HERE.parent.parent)
            if fix:
                print(f"{rel}: bootstrap cell out of date, regenerating")
            else:
                print(f"{rel}: bootstrap cell out of date")

    if args.check and any_changed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
