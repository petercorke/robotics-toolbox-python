#!/usr/bin/env python
"""Git pre-commit hook: tidy the notebooks that are about to be committed.

Notebooks are committed without outputs, with the same kernel and metadata, and written
as Jupyter writes them (see README.md).  This does that for the notebooks in the commit,
so that editing a notebook in VS Code and committing it is all there is to do.  CI runs
the same checks, but never has to complain.

Install it once, in any clone or worktree of the repository:

    python docs/notebooks/pre_commit.py --install

It refuses the commit, with a message, if a notebook still has merge conflict markers.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True  # a hook should leave no __pycache__ behind

HERE = Path(__file__).resolve().parent
STUB = """#!/bin/sh
# installed by docs/notebooks/pre_commit.py --install
top=$(git rev-parse --show-toplevel)
script="$top/docs/notebooks/pre_commit.py"
[ -f "$script" ] || exit 0
exec python "$script" || exec python3 "$script"
"""


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_rtb_nb_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tidy(paths: list[Path]) -> list[Path]:
    """Tidy notebooks in place.

    :param paths: the notebooks
    :returns: the ones that were changed
    :raises SystemExit: if a notebook has merge conflict markers
    """
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if any(line.startswith(("<<<<<<<", ">>>>>>>")) for line in text.splitlines()):
            raise SystemExit(
                f"{path.name} still has merge conflict markers: resolve the conflict, "
                "then commit again"
            )

    sync_bootstrap = _load("sync_bootstrap")
    clear_outputs = _load("clear_outputs")
    template = (HERE / "_rtb_nb_bootstrap.py").read_text(encoding="utf-8")

    changed = []
    for path in paths:
        # the bootstrap cell and the cell ids first, then outputs, metadata and
        # formatting, which has the last word
        a = sync_bootstrap.process_notebook(path, template, fix=True)
        b = clear_outputs.process_notebook(path, fix=True)
        if a or b:
            changed.append(path)
    return changed


def install(hooks_dir: Path) -> str:
    """Write the hook that runs this script before every commit.

    :param hooks_dir: the ``hooks`` folder of the git directory
    :returns: a message saying what was done
    """
    hook = hooks_dir / "pre-commit"
    if hook.exists() and hook.read_text(encoding="utf-8") != STUB:
        return f"{hook} already exists and is not ours, left alone"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook.write_text(STUB, encoding="utf-8", newline="\n")
    hook.chmod(0o755)
    return f"installed {hook}"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def main() -> int:
    if "--install" in sys.argv:
        hooks = Path(_git("rev-parse", "--git-common-dir").strip()).resolve() / "hooks"
        print(install(hooks))
        return 0

    top = Path(_git("rev-parse", "--show-toplevel").strip())
    names = _git(
        "diff", "--cached", "--name-only", "--diff-filter=ACM", "--", "docs/notebooks"
    ).split()
    paths = [top / n for n in names if n.endswith(".ipynb")]
    if not paths:
        return 0

    changed = tidy(paths)
    if changed:
        subprocess.run(["git", "add", "--", *map(str, changed)], check=True)
        for path in changed:
            print(f"tidied {path.name} (outputs, metadata, formatting)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
