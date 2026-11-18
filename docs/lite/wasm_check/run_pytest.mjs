#!/usr/bin/env node
/**
 * Run the test suite inside real Pyodide under Node, using the notebook install
 * path.  This checks WASM behaviour; browser widget presentation is covered by
 * ../browser_check/ instead.
 *
 * Usage: node docs/lite/wasm_check/run_pytest.mjs <wheel>
 */

import { loadPyodide } from "pyodide";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const wheelPath = process.argv[2];
if (!wheelPath) {
  console.error("Usage: node docs/lite/wasm_check/run_pytest.mjs <wheel>");
  process.exit(2);
}

const root = fileURLToPath(new URL("../../../", import.meta.url));
const wheelName = path.basename(wheelPath);
const pyodide = await loadPyodide();
await pyodide.loadPackage("micropip");

function copyTree(source, target) {
  pyodide.FS.mkdirTree(target);
  for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
    if (["__pycache__", ".ipynb_checkpoints", ".env"].includes(entry.name)
        || entry.name.startsWith(".env.")) continue;
    const src = path.join(source, entry.name);
    const dst = target + "/" + entry.name;
    if (entry.isDirectory()) copyTree(src, dst);
    else if (entry.isFile() && !entry.name.endsWith(".pyc")) {
      pyodide.FS.writeFile(dst, fs.readFileSync(src));
    }
  }
}

// Some tests inspect source, compile examples or read notebook resources.  Keep
// their repository layout, but import the toolbox from the installed wheel.
for (const directory of ["tests", "src/roboticstoolbox", "examples", "docs/notebooks"]) {
  copyTree(path.join(root, directory), "/repo/" + directory);
}
pyodide.FS.writeFile("/repo/pyproject.toml", fs.readFileSync(path.join(root, "pyproject.toml")));
// The JupyterLite capability-banner tests read these alongside the notebooks.
pyodide.FS.mkdirTree("/repo/docs/lite");
for (const name of ["inject_jspi_check.py", "jspi-check.js"]) {
  pyodide.FS.writeFile("/repo/docs/lite/" + name, fs.readFileSync(path.join(root, "docs/lite", name)));
}
pyodide.FS.writeFile("/" + wheelName, fs.readFileSync(wheelPath));
pyodide.globals.set("_wheel_url", "emfs:/" + wheelName);

const exitCode = await pyodide.runPythonAsync(`
import os
import sys
import types

os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, "/repo/docs/notebooks")
import micropip
import _rtb_nb_bootstrap as bootstrap

# As in check_pyodide_install.mjs, only replace piplite's wheel lookup.  The real
# bootstrap still owns the install steps and resolves all other dependencies.
piplite = types.ModuleType("piplite")
async def install(requirements, deps=True, **kwargs):
    if requirements == "roboticstoolbox-python":
        requirements = _wheel_url
    return await micropip.install(requirements, deps=deps, **kwargs)
piplite.install = install
sys.modules["piplite"] = piplite
await bootstrap.ensure_installed(extras=["ik"])
await micropip.install("pytest")

os.chdir("/repo")
sys.path.insert(0, "/repo")
import pytest
print(f"Pyodide: {sys.version}; {sys.platform}")
# fd capture and pytest-timeout rely on OS facilities unavailable in Pyodide.
int(pytest.main([
    "tests", "-m", "not slow", "--ignore=tests/test_blocks.py", "-q",
    "-p", "no:cacheprovider", "-p", "no:timeout", "--capture=sys", "-rfE",
]))
`);
process.exitCode = exitCode;
