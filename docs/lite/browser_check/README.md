# JupyterLite browser check

Runs JupyterLite in a real browser (Chromium, with Playwright) and checks what the
Node check in `../wasm_check/` cannot see: that the site installs the toolbox that was
built with it, and that plots and widgets work.

`check_lite_in_browser.py` builds a small site from `docs/lite` (its configuration and
the wheel in `docs/lite/pypi`) plus one notebook that it writes, `browser_check.ipynb`,
which starts with the real generated bootstrap cell.  It opens that in Chromium and runs
the notebook, which requires:

| Check | Why |
|---|---|
| the bootstrap cell prints `Running in browser using RTB v...` | the install path works at all |
| a hash of the installed `PyPlot.py` equals that of the checkout | the site installs the wheel built with it, not the last release from PyPI (a plain `micropip.install` cannot see the site's own index, `piplite` can; this is what went wrong once and nothing else noticed) |
| an `ipywidgets` slider is drawn | widgets work in JupyterLite |
| an `ipympl` figure of one pose is drawn | `%matplotlib ipympl`, which the notebooks use, works |
| a trajectory shows at least 5 different images while its cell is running | an `ipympl` figure cannot update while a cell runs, so the Toolbox pushes inline frames for an animation (`tests/test_notebook_animation.py` checks the same thing with a kernel and no browser) |

It runs in `docs.yml` after the site is built, as an informational step until it has
passed on `main`.  To run it yourself:

```bash
pip install playwright && playwright install chromium
pip install jupyterlite-core "jupyterlite-pyodide-kernel==0.6.1" jupyter-server ipympl
RTB_PURE_WHEEL=1 python -m build --wheel --outdir docs/lite/pypi
python docs/lite/browser_check/check_lite_in_browser.py
```

`CHROMIUM_PATH` selects a Chromium executable, and `--screenshot` says where to save the
page when a check fails.  The run takes about two minutes: most of it is building the
site and downloading Pyodide's packages in the browser.

The kernel version is pinned in `docs.yml` (`jupyterlite-pyodide-kernel==0.6.1`) for
browsers without WebAssembly JSPI; see the comment there before changing it.
