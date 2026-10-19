#!/usr/bin/env python
"""Clear cell outputs, execution counts and editor metadata from every notebook here.

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

# What an editor writes into a notebook's metadata depends on who opened it last: the
# name of their Python environment, its version, interpreter paths, widget state and
# so on.  That is noise in every diff, so notebooks are committed with the same
# kernel and language, and with only the metadata that belongs to this repository.
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
LANGUAGE_INFO = {"name": "python"}
OWN_METADATA = ("rtb_bootstrap",)


def normalize_metadata(nb: dict) -> bool:
    """Set a notebook's metadata to its canonical form, in place.

    :returns: True if it was (or would be) changed.
    """
    old = nb.get("metadata", {})
    new = {"kernelspec": dict(KERNELSPEC), "language_info": dict(LANGUAGE_INFO)}
    new.update({k: old[k] for k in OWN_METADATA if k in old})
    if old == new:
        return False
    nb["metadata"] = new
    return True


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


def canonical_text(nb: dict) -> str:
    """The notebook as nbformat (and so VS Code and Jupyter) writes it."""
    return json.dumps(nb, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


def process_notebook(path: Path, fix: bool) -> bool:
    """Check (and optionally clear) one notebook's outputs and metadata.

    :returns: True if the notebook was (or would be) changed.
    """
    raw = path.read_text(encoding="utf-8")
    nb = json.loads(raw)

    # a list, not a generator: any() would stop at the first dirty cell and leave
    # the rest uncleared
    changed = any([clear_cell(cell) for cell in nb.get("cells", [])])
    changed = normalize_metadata(nb) or changed
    text = canonical_text(nb)
    changed = changed or text != raw

    if changed and fix:
        with path.open("w", encoding="utf-8", newline="\n") as f:
            f.write(text)

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
