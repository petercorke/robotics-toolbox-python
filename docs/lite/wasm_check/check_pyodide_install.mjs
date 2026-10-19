#!/usr/bin/env node
/**
 * Dependency-completeness check for the Pyodide/JupyterLite install path.
 *
 * Scope, deliberately narrow: this confirms that every package
 * _rtb_nb_bootstrap.py's Pyodide branch installs actually resolves, and
 * that `import roboticstoolbox` succeeds afterward -- catching exactly
 * the class of bug where a dependency that cannot be installed in Pyodide
 * (for example spatialmath-python's pre-commit requirement) breaks the
 * JupyterLite demo silently.
 *
 * What this deliberately does NOT test: whether the real notebook's
 * relative-URL wheel-install path resolves correctly inside a real
 * browser's Pyodide Web Worker. Node has no equivalent of a Worker's base
 * URL, and faking one convincingly requires already knowing the right
 * answer -- that question needs a real browser, and was already answered
 * that way (see docs/notebooks/README.md). This check mounts the wheel
 * directly into Pyodide's own virtual filesystem instead (the `emfs:`
 * micropip scheme) and installs from there, sidestepping the HTTP/URL
 * layer entirely -- deliberately, not as a shortcut around it.
 *
 * Reuses the real, unmodified ensure_installed() from _rtb_nb_bootstrap.py
 * rather than re-listing the install steps here -- that would just be a
 * second, driftable copy of the exact list this check exists to keep
 * honest.
 *
 * Usage:
 *   node check_pyodide_install.mjs <path-to-wheel>
 */

import { loadPyodide } from "pyodide";
import fs from "node:fs";
import path from "node:path";

const wheelPath = process.argv[2];
if (!wheelPath) {
  console.error("Usage: node check_pyodide_install.mjs <path-to-wheel>");
  process.exit(2);
}

const wheelName = path.basename(wheelPath);
const bootstrapPath = new URL("../../notebooks/_rtb_nb_bootstrap.py", import.meta.url);

const pyodide = await loadPyodide();
await pyodide.loadPackage("micropip");

pyodide.FS.writeFile("/" + wheelName, fs.readFileSync(wheelPath));
// The bootstrap looks for the site's bundled wheel in /pypi, and falls back to
// installing from PyPI if there is none.  Without this the check would test the
// last release, not the wheel just built.
pyodide.FS.mkdirTree("/pypi");
pyodide.FS.writeFile("/pypi/" + wheelName, fs.readFileSync(wheelPath));
pyodide.FS.writeFile("/_rtb_nb_bootstrap.py", fs.readFileSync(bootstrapPath));

const driver = `
import sys
sys.path.insert(0, "/")

import micropip
import _rtb_nb_bootstrap as bootstrap

# Test-only scaffolding: redirect the one HTTP/relative-URL install call to
# the wheel already mounted in Pyodide's own virtual filesystem, so this
# check doesn't need to fake a browser Web Worker's base URL (see this
# file's module docstring for why that's out of scope here). Everything
# else -- the install steps, which is what this check actually
# exists to catch drift in -- runs completely unmodified.
_real_install = micropip.install
async def _redirect_wheel_install(requirements, **kwargs):
    if isinstance(requirements, str) and requirements.startswith("pypi/"):
        requirements = "emfs:/${wheelName}"
    return await _real_install(requirements, **kwargs)
micropip.install = _redirect_wheel_install

await bootstrap.ensure_installed(extras=['ik'])

import roboticstoolbox as rtb  # noqa: F401
rtb.models.DH.Panda().fkine([0, 0, 0, 0, 0, 0, 0])

# the notebook ik-symbolic.ipynb: SymPy only, since python-flint is not usable in
# Pyodide 0.27.  A closed-form arm and a general 6R arm (the slow one, about 10 s).
import time, warnings
warnings.simplefilter("ignore")
for name in ("Puma560", "Jaco"):
    robot = getattr(rtb.models.DH, name)()
    t = time.time()
    ik = robot.analytic_ik()
    sol = ik.solve(robot.fkine([0.1] * robot.n))
    assert len(sol.q) > 0, name
    print(f"analytic_ik {name}: {type(ik).__name__}, {time.time() - t:.1f}s")
print("import roboticstoolbox: OK")
`;

try {
  await pyodide.runPythonAsync(driver);
  console.log("\nPASS: Pyodide install check succeeded.");
} catch (err) {
  console.error(`
============================================================
FAIL: the Pyodide/JupyterLite install check failed.

This means _rtb_nb_bootstrap.py's Pyodide branch doesn't actually
install and import cleanly. Most likely cause: a new dependency in
pyproject.toml that cannot be installed in Pyodide, or one whose
requirements conflict with Pyodide's built-in packages.

To reproduce locally:
    RTB_PURE_WHEEL=1 python -m build --wheel --outdir docs/lite/pypi
    node docs/lite/wasm_check/check_pyodide_install.mjs docs/lite/pypi/*.whl

The traceback below says which package is the problem. If it is a
conflict, install that package with deps=False in the Pyodide branch of
docs/notebooks/_rtb_nb_bootstrap.py, as is done for spatialmath-python, then
run python docs/notebooks/sync_bootstrap.py and commit the regenerated
notebooks.

Original error below.
============================================================
`);
  console.error(err);
  process.exitCode = 1;
}
