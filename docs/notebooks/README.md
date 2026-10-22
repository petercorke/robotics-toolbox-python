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
  Then it installs the toolbox with its dependencies.  In JupyterLite this goes through
  `piplite`, which looks in the site's own package index first, so the toolbox is the
  wheel that was built with the site and not the last release from PyPI (a plain
  `micropip.install` cannot see that index).  `docs/lite/browser_check/` checks it.

## The first cell

The first cell of a notebook is ordinary markdown, written by hand: the logo and title,
the copyright line and a description.  The logo is an absolute GitHub URL, since Colab
fetches only the one `.ipynb` file and cannot see `figs/`.  A notebook that was updated
for API drift says so there ("Modernised <date>").

The second cell is the generated bootstrap cell, below.

## Editing a notebook

Edit it as you like, in VS Code or anywhere, and commit it.  That is all.

Install the commit hook once, in any clone or worktree:

    python docs/notebooks/pre_commit.py --install

and every commit that includes a notebook first tidies it: no outputs or execution
counts, a cell id on every cell, the same kernel and language in the metadata (an editor
writes the name and version of your Python environment, which is noise in every diff),
and the file written as Jupyter writes it, so that opening and saving a notebook never
shows up as a change.  It refuses the commit, with a message, if a notebook still has
merge conflict markers.

Without the hook, run `python docs/notebooks/clear_outputs.py` before committing, and
`python docs/notebooks/sync_bootstrap.py` after changing `_rtb_nb_bootstrap.py` or a
notebook's `rtb_bootstrap` metadata.  `docs.yml` runs both with `--check`, so CI tells you
if you forgot.

## Outputs are not committed

Notebooks are committed without outputs or execution counts, so diffs show only what
was changed.  `python clear_outputs.py` clears them, and `--check` reports.  Both checks
run in the `docs.yml` workflow, and `tests/test_notebook_bootstrap.py` tests the scripts.

## Testing the install in the browser

`docs/lite/wasm_check/` runs the bootstrap in Pyodide under Node.js against the wheel
just built, in CI.  See its README.

## Extras

A notebook that needs a toolbox extra says so in its own metadata, not in the cell:

```json
"metadata": {"rtb_bootstrap": {"extras": ["ik"]}}
```

`sync_bootstrap.py` then generates `ensure_installed(extras=['ik'])`.  For `ik`, Colab
installs `roboticstoolbox-python[ik]`, a local install is told what to install if SymPy
is missing, and in the browser SymPy comes from Pyodide's own packages.  python-flint is
not installed in the browser: the one bundled with Pyodide 0.27 is too old, and the
toolbox then uses SymPy alone (about 2x slower; 10 s for a general 6R arm).
`ik-symbolic.ipynb` is the one notebook that does this.
