#!/usr/bin/env python
"""Clear cell outputs and execution counts from every notebook in this folder.

Pure stdlib (no dependency on jupyter/nbconvert being installed), so it can run as
a commit hook on any machine with just Python. Equivalent to
`jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace`, which is what
clear_notebook_outputs.sh does for ad-hoc/manual use.

Usage:
    python clear_outputs.py            clear any notebook that has output
    python clear_outputs.py --check    report only, exit 1 if anything has output
    python clear_outputs.py FILE ...   only consider the given notebook(s)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def clear_cell(cell: dict) -> bool:
    """Clear one cell's outputs/execution count in place.

    :returns: True if the cell was (or would be) changed.
    """
    if cell.get("cell_type") != "code":
        return False

    changed = False
    if cell.get("outputs"):
        changed = True
        cell["outputs"] = []
    if cell.get("execution_count") is not None:
        changed = True
        cell["execution_count"] = None
    if "execution" in cell.get("metadata", {}):
        changed = True
        del cell["metadata"]["execution"]

    return changed


def process_notebook(path: Path, fix: bool) -> bool:
    """Check (and optionally clear) one notebook's outputs.

    :returns: True if the notebook was (or would be) changed.
    """
    with path.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # a list, not a generator: any() would stop at the first dirty cell and leave
    # the rest uncleared
    changed = any([clear_cell(cell) for cell in nb.get("cells", [])])

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
        help="report notebooks with output without modifying them; exit 1 if any found",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="only consider these notebooks (default: every *.ipynb in this folder)",
    )
    args = parser.parse_args()

    fix = not args.check
    notebooks = [p.resolve() for p in args.files] if args.files else sorted(HERE.glob("*.ipynb"))

    any_changed = False
    for nb_path in notebooks:
        changed = process_notebook(nb_path, fix=fix)
        if changed:
            any_changed = True
            rel = nb_path.relative_to(HERE.parent.parent)
            if fix:
                print(f"{rel}: found output, clearing it")
            else:
                print(f"{rel}: found output")

    if args.check and any_changed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
