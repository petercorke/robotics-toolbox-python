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


async def ensure_installed(extras: tuple[str, ...] | list[str] = ()) -> bool:
    """Install roboticstoolbox-python if needed, and report the environment.

    :param extras: toolbox extras the notebook needs, for example ``["ik"]``.  They
        are set per notebook in its metadata (``rtb_bootstrap.extras``) and
        passed here by the generated cell.
    :returns: True if running on Google Colab, False otherwise.
    """
    if sys.platform == "emscripten":
        # In JupyterLite, piplite looks in this site's own package index first, so the
        # toolbox it installs is the wheel that was built with the site.  micropip
        # alone cannot see that index and would install the last release from PyPI.
        # Outside JupyterLite (for example a test under Node) there is no piplite.
        try:
            import piplite

            install = piplite.install
        except ImportError:
            import micropip

            install = micropip.install

        # spatialmath-python <= 1.1.18 lists pre-commit (a dev tool) as a runtime
        # dependency; its virtualenv requires packaging>=26.3, which conflicts with
        # the packaging 24.2 built into Pyodide, so a normal resolve fails.  Install
        # it without dependencies first: it is then already satisfied when the
        # toolbox's own dependencies are resolved below.  Remove this line once a
        # spatialmath release without that dependency is out
        # (rai-opensource/spatialmath-python#250).
        await install("spatialmath-python", deps=False)
        await install("ipympl")
        if "ik" in extras:
            # SymPy is one of Pyodide's own packages.  python-flint is not installed:
            # the one bundled with Pyodide 0.27 is too old for the toolbox, which
            # then uses SymPy for the polynomial steps, about 2x slower for a
            # general 6R arm (10 s to derive a Kinova Jaco).
            await install("sympy")
        await install("roboticstoolbox-python")
        where, colab = "in browser", False
    else:
        try:
            import google.colab  # noqa: F401
        except ImportError:
            where, colab = "locally", False
            if "ik" in extras:
                try:
                    import sympy  # noqa: F401
                except ImportError:
                    print(
                        "This notebook needs SymPy: "
                        "pip install 'roboticstoolbox-python[ik]'"
                    )
        else:
            print("Installing roboticstoolbox-python...")
            requirement = "roboticstoolbox-python"
            if extras:
                requirement += "[" + ",".join(extras) + "]"
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", requirement],
                check=True,
            )
            where, colab = "on Colab", True

    import roboticstoolbox

    version = getattr(roboticstoolbox, "__version__", "unknown")
    print(f"Running {where} using RTB v{version}")
    return colab
