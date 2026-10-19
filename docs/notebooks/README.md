# Notebooks

These notebooks run locally (Jupyter, JupyterLab, VS Code), on Google Colab, and in the
browser with JupyterLite.  Locally, install the toolbox first with
`pip install roboticstoolbox-python`, and Jupyter or the VS Code Jupyter extension.

## How the environment is set up

Google Colab fetches only the single `.ipynb` it is opened from, so each notebook
carries its own copy of the code that installs the toolbox.  That copy is the cell whose
first line starts with `# RTB_BOOTSTRAP_CELL`.  It can be anywhere in a notebook (it is
found by that line, not by its position) and is folded by default.

* `_rtb_nb_bootstrap.py` is the single source of truth for that cell.
* `python sync_bootstrap.py` regenerates the cell in every notebook; with `--check` it
  only reports and exits 1 if any notebook is out of date.  **Never edit the cell in a
  notebook**, edit `_rtb_nb_bootstrap.py` and run the script.
* In the browser the cell installs `spatialmath-python` without its dependencies first,
  because release 1.1.18 lists `pre-commit` as a runtime dependency, which conflicts
  with the `packaging` built into Pyodide (rai-opensource/spatialmath-python#250).
  Then it installs the toolbox with its dependencies, from the wheel bundled with the
  JupyterLite site if there is one, or else from PyPI.

## Outputs are not committed

Notebooks are committed without outputs or execution counts, so diffs show only what
was changed.  `python clear_outputs.py` clears them, and `--check` reports.  Both checks
run in the `docs.yml` workflow, and `tests/test_notebook_bootstrap.py` tests the scripts.

## Testing the install in the browser

`docs/lite/wasm_check/` runs the bootstrap in Pyodide under Node.js against the wheel
just built, in CI.  See its README.

`ik-symbolic.ipynb` has no bootstrap cell yet: it needs the `ik` extra (SymPy and
python-flint), and is not part of the JupyterLite site.
