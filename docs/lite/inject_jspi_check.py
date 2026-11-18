"""Add the browser JSPI check to every page of a built JupyterLite site.

Run after ``jupyter lite build``::

    python inject_jspi_check.py ../build/html/lite

It copies ``jspi-check.js`` to the site root and adds a ``<script>`` tag for it to the
``<head>`` of each page that starts the kernel.  The script shows a banner if the
browser has no WebAssembly JSPI, which the Pyodide kernel needs.  Safe to run twice.
"""

import shutil
import sys
from pathlib import Path

SCRIPT = "jspi-check.js"

# the pages JupyterLite generates that can run a kernel
PAGES = ["lab", "notebooks", "tree", "repl", "consoles", "edit", "doc"]


def inject(site: Path) -> list[Path]:
    """Put the check into a built JupyterLite site.

    :param site: the site's root, the folder holding ``lab/`` and ``jupyter-lite.json``
    :returns: the pages that were changed
    :raises FileNotFoundError: if there is no page to change, so the check cannot
        silently be missing from a deployed site
    """
    site = Path(site)
    pages = [
        site / p / "index.html" for p in PAGES if (site / p / "index.html").exists()
    ]
    if not pages:
        raise FileNotFoundError(f"no JupyterLite pages found under {site}")
    shutil.copy(Path(__file__).with_name(SCRIPT), site / SCRIPT)

    changed = []
    for index in pages:
        html = index.read_text()
        if SCRIPT in html:
            continue
        # relative to the page, which is one folder below the site root
        tag = f'<script src="../{SCRIPT}"></script>'
        if "<head>" not in html:
            raise ValueError(f"{index} has no <head> to add the script to")
        index.write_text(html.replace("<head>", f"<head>\n    {tag}", 1))
        changed.append(index)
    return changed


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    for path in inject(Path(sys.argv[1])):
        print(f"added {SCRIPT} to {path}")
