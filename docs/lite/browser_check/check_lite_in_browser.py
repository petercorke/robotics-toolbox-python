#!/usr/bin/env python
"""Check JupyterLite in a real browser: the toolbox installs, and plots and widgets work.

The Node/Pyodide check in ``../wasm_check/`` tests the install and the Python side
without a browser.  This one builds a small JupyterLite site that contains the wheel
built from this checkout and one notebook, ``browser_check.ipynb``, opens it in
Chromium, and runs the notebook, which does what the notebooks of the site do:

1. runs the generated bootstrap cell and prints the version of the toolbox,
2. imports the toolbox and prints a hash of the source of ``PyPlot.py``,
3. shows an ``ipywidgets`` slider,
4. plots a robot with ``%matplotlib ipympl`` (an interactive figure),
5. animates a trajectory with ``%matplotlib ipympl``.

It requires: the bootstrap line, the hash of the installed ``PyPlot.py`` to be that of
the checkout (the site installs the wheel built with it, not the last release from
PyPI, which is what a plain ``micropip.install`` does), the slider and the figure to be
drawn, and the animation to show at least ``--min-frames`` different images while the
cell is still running (an ``ipympl`` figure cannot update then, so the Toolbox pushes
frames of its own).

Usage::

    pip install playwright && playwright install chromium
    pip install jupyterlite-core "jupyterlite-pyodide-kernel==0.6.1" jupyter-server ipympl
    RTB_PURE_WHEEL=1 python -m build --wheel --outdir docs/lite/pypi
    python docs/lite/browser_check/check_lite_in_browser.py

``CHROMIUM_PATH`` selects a particular Chromium executable.  A screenshot of the final
page is saved on failure, to ``--screenshot``.  Exits 0 if the checks pass, 1 if not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NOTEBOOKS = REPO / "docs" / "notebooks"
PYPLOT = REPO / "src" / "roboticstoolbox" / "backends" / "PyPlot" / "PyPlot.py"

BOOTSTRAP_TEXT = "Running in browser using RTB v"
# the cells of the notebook, in order
BOOTSTRAP, SETUP, WIDGET, FIGURE, ANIMATION = range(5)

CELLS = [
    None,  # the generated bootstrap cell, see make_notebook
    """%matplotlib ipympl
import hashlib, importlib
import roboticstoolbox as rtb
module = importlib.import_module("roboticstoolbox.backends.PyPlot.PyPlot")
print("PYPLOT_MD5", hashlib.md5(open(module.__file__, "rb").read()).hexdigest())
robot = rtb.models.DH.Puma560()
trajectory = rtb.jtraj(robot.qz, robot.qn, 100).q
print("READY")""",
    """import ipywidgets
slider = ipywidgets.IntSlider(value=3, description="n")
display(slider)""",
    "robot.plot(robot.qn);",
    "robot.plot(trajectory);",
]


class CheckFailed(Exception):
    """A check failed; the message says which."""


def make_notebook(path: Path) -> None:
    """Write browser_check.ipynb, with the real generated bootstrap cell."""
    sys.path.insert(0, str(NOTEBOOKS))
    import sync_bootstrap

    template = (NOTEBOOKS / "_rtb_nb_bootstrap.py").read_text(encoding="utf-8")
    sources = list(CELLS)
    sources[BOOTSTRAP] = "".join(sync_bootstrap.generated_source(template))
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": f"cell{i}",
                "metadata": {},
                "outputs": [],
                "source": source,
            }
            for i, source in enumerate(sources)
        ],
        "metadata": {
            "kernelspec": {
                "name": "python",
                "display_name": "Python (Pyodide)",
                "language": "python",
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


def build_site(lite_dir: Path, work: Path) -> Path:
    """Build a JupyterLite site with the wheel in lite_dir/pypi and the notebook."""
    source = work / "lite"
    (source / "files").mkdir(parents=True)
    for name in ("jupyter-lite.json", "jupyter_lite_config.json"):
        shutil.copy(lite_dir / name, source / name)
    shutil.copytree(lite_dir / "pypi", source / "pypi")
    if not list((source / "pypi").glob("roboticstoolbox_python-*.whl")):
        raise CheckFailed(f"no wheel in {lite_dir / 'pypi'}: build it first")
    make_notebook(source / "files" / "browser_check.ipynb")

    out = work / "site"
    subprocess.run(
        [sys.executable, "-m", "jupyter", "lite", "build", "--output-dir", str(out)],
        cwd=source,
        check=True,
        capture_output=True,
    )
    # let the script drive JupyterLab through its commands
    config = out / "jupyter-lite.json"
    data = json.loads(config.read_text(encoding="utf-8"))
    data.setdefault("jupyter-config-data", {})["exposeAppInBrowser"] = True
    config.write_text(json.dumps(data), encoding="utf-8")
    return out


def run_checks(page, timeout_s: float, min_frames: int) -> list[str]:
    """Run the notebook in the page and check it; returns what passed."""
    passed = []
    cells = page.locator(".jp-NotebookPanel:not(.lm-mod-hidden) .jp-Cell")

    page.wait_for_selector(
        ".jp-NotebookPanel:not(.lm-mod-hidden) .jp-Notebook", timeout=90_000
    )
    page.wait_for_function("window.jupyterapp !== undefined", timeout=60_000)
    time.sleep(3)
    page.evaluate(
        "() => { window.jupyterapp.commands.execute('notebook:run-all-cells'); return 1 }"
    )

    prompts = lambda: page.evaluate(  # noqa: E731
        "[...document.querySelectorAll("
        "'.jp-NotebookPanel:not(.lm-mod-hidden) .jp-Cell .jp-InputPrompt')]"
        ".map(e => e.textContent.trim())"
    )
    frames: list[str] = []
    deadline = time.monotonic() + timeout_s
    finished_at = None
    while time.monotonic() < deadline:
        time.sleep(1)
        state = prompts()
        if len(state) == len(CELLS) and "*" in state[ANIMATION]:
            # the animation is running: what is on the screen must keep changing
            area = cells.nth(ANIMATION).locator(".jp-OutputArea")
            try:
                if area.count() and area.first.is_visible():
                    frames.append(
                        hashlib.md5(area.first.screenshot(timeout=3000)).hexdigest()
                    )
            except PlaywrightError:
                pass
        if len(state) == len(CELLS) and all(
            p and "*" not in p and p != "[ ]:" for p in state
        ):
            finished_at = finished_at or time.monotonic()
            if time.monotonic() - finished_at > 3:
                break
    else:
        raise CheckFailed(f"the notebook did not finish within {timeout_s:.0f} s")

    outputs = page.evaluate(
        "[...document.querySelectorAll('.jp-NotebookPanel:not(.lm-mod-hidden) .jp-Cell')]"
        ".map(c => { const o = c.querySelector('.jp-OutputArea'); return o ? o.innerText : '' })"
    )
    for i, text in enumerate(outputs):
        if "Traceback" in text or "Error" in text.split("\n")[0]:
            raise CheckFailed(f"cell {i + 1} raised:\n{text[-1200:]}")

    if BOOTSTRAP_TEXT not in outputs[BOOTSTRAP]:
        raise CheckFailed(
            f"no '{BOOTSTRAP_TEXT}' from the bootstrap cell:\n{outputs[BOOTSTRAP][-800:]}"
        )
    passed.append(
        next(line for line in outputs[BOOTSTRAP].splitlines() if BOOTSTRAP_TEXT in line)
    )

    expected = hashlib.md5(PYPLOT.read_bytes()).hexdigest()
    reported = [
        line.split()[1] for line in outputs[SETUP].splitlines() if "PYPLOT_MD5" in line
    ]
    if reported != [expected]:
        raise CheckFailed(
            "the toolbox in the browser is not the wheel built from this checkout "
            f"(PyPlot.py md5 {reported}, expected {expected}): the site installed "
            "the last release from PyPI instead of the wheel built with it"
        )
    passed.append("the installed toolbox is the wheel built with the site")

    if cells.nth(WIDGET).locator(".widget-slider, .jupyter-widgets").count() == 0:
        raise CheckFailed("the ipywidgets slider was not drawn")
    passed.append("ipywidgets slider drawn")

    if cells.nth(FIGURE).locator(".jupyter-matplotlib-canvas-container").count() == 0:
        raise CheckFailed("the ipympl figure of a single pose was not drawn")
    passed.append("ipympl figure drawn (interactive)")

    distinct = len(set(frames))
    if distinct < min_frames:
        raise CheckFailed(
            f"the animation showed {distinct} different images in {len(frames)} "
            f"samples while it was running, wanted at least {min_frames}"
        )
    passed.append(f"animation: {distinct} different images in {len(frames)} samples")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lite-dir", type=Path, default=REPO / "docs" / "lite")
    parser.add_argument("--timeout", type=float, default=420, help="seconds to run")
    parser.add_argument("--min-frames", type=int, default=5)
    parser.add_argument("--screenshot", type=Path, default=Path("lite_check_failed.png"))
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        site = build_site(args.lite_dir, Path(tmp))
        server = subprocess.Popen(
            [sys.executable, "-m", "http.server", "8890", "--directory", str(site)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            time.sleep(2)
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    executable_path=os.environ.get("CHROMIUM_PATH") or None
                )
                # a tall window, since JupyterLab draws only the cells that are visible
                page = browser.new_context(
                    viewport={"width": 1300, "height": 5000}
                ).new_page()
                errors: list[str] = []
                page.on("pageerror", lambda e: errors.append(str(e)[:200]))
                try:
                    page.goto(
                        "http://localhost:8890/lab/index.html"
                        "?path=browser_check.ipynb&reset",
                        wait_until="load",
                    )
                    passed = run_checks(page, args.timeout, args.min_frames)
                except (CheckFailed, PlaywrightError) as e:
                    print(f"FAIL: {e}")
                    for line in errors[:5]:
                        print(f"  page error: {line}")
                    try:
                        page.screenshot(path=str(args.screenshot))
                        print(f"screenshot: {args.screenshot}")
                    except PlaywrightError:
                        pass
                    return 1
                finally:
                    browser.close()
        finally:
            server.terminate()

    for line in passed:
        print(f"PASS: {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
