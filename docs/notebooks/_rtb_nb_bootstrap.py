"""Environment bootstrap for roboticstoolbox-python's Jupyter notebooks.

Installs the toolbox (and reports how) across the three environments a notebook in
this folder might run in: a local Jupyter/VS Code install, Google Colab, and
JupyterLite (Pyodide/WASM, in-browser).

This file is the single source of truth for that logic. Every notebook's own
bootstrap cell is a generated copy of this file's content, produced by
sync_bootstrap.py -- see docs/notebooks/README.md for the full explanation.
"""

import subprocess
import sys
from pathlib import Path


async def ensure_installed() -> bool:
    """Install roboticstoolbox-python if needed, and report the environment.

    :returns: True if running on Google Colab, False otherwise.
    """
    if sys.platform == "emscripten":
        import micropip

        # spatialmath-python <= 1.1.18 lists pre-commit (a dev tool) as a runtime
        # dependency; its virtualenv requires packaging>=26.3, which conflicts with
        # the packaging 24.2 built into Pyodide, so a normal resolve fails.  Install
        # it without dependencies first: it is then already satisfied when the
        # toolbox's own dependencies are resolved below.  Remove this line once a
        # spatialmath release without that dependency is out
        # (rai-opensource/spatialmath-python#250).
        await micropip.install("spatialmath-python", deps=False)
        await micropip.install("ipympl")

        wheels = sorted(Path("/pypi").glob("roboticstoolbox_python-*.whl"))
        if wheels:
            # Prefer the wheel bundled with this JupyterLite site. Relative,
            # not absolute: a leading slash resolves against the origin, not
            # this site's own base URL (a GitHub Pages project subpath).
            await micropip.install(f"pypi/{wheels[-1].name}")
        else:
            # Fall back to PyPI when running outside the published site layout.
            await micropip.install("roboticstoolbox-python")
        where, colab = "in browser", False
    else:
        try:
            import google.colab  # noqa: F401
        except ImportError:
            where, colab = "locally", False
        else:
            print("Installing roboticstoolbox-python...")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "roboticstoolbox-python",
                ],
                check=True,
            )
            where, colab = "on Colab", True

    import roboticstoolbox

    version = getattr(roboticstoolbox, "__version__", "unknown")
    print(f"Running {where} using RTB v{version}")
    return colab
