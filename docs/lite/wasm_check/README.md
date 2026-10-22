# Pyodide install check

A real-Pyodide, real-WASM check that the JupyterLite install path actually works —
not a mock, not a guess. Runs Pyodide under Node.js (the `pyodide` npm package),
which is the same WASM runtime a real browser uses, just without a browser or a
Web Worker around it.

## What this catches

A new dependency that cannot be installed in Pyodide (not pure Python, or with a
requirement that conflicts with a package built into Pyodide) breaks the JupyterLite
demo silently. An example is `spatialmath-python` 1.1.18 listing `pre-commit`, which
needs `packaging>=26.3` where Pyodide has 24.2; the bootstrap installs it with
`deps=False` for that reason. This check exists so that class of bug fails CI instead
of shipping: it runs the real `ensure_installed()` from
`docs/notebooks/_rtb_nb_bootstrap.py` against the pure wheel, installs the
dependencies for real, and confirms `import roboticstoolbox` works, a forward kinematics call
returns, the analytic inverse kinematics of a Puma 560 and a Kinova Jaco work (with SymPy
only), and that a trajectory plotted under the `ipympl` backend is pushed as a
different inline image for every step.

It reuses that function unmodified rather than re-listing the install steps here —
a second copy of that list would just be a second place for it to drift.

## What this deliberately does NOT catch

Whether the site's own package index really serves the wheel, and so whether the
notebook installs the wheel built with the site and not the last release from PyPI.
In JupyterLite the bootstrap installs through `piplite`, which looks in the site's
index first; there is no site or `piplite` under Node.  This check stands in for it with
a small fake `piplite` module that installs the wheel, mounted in Pyodide's own virtual
filesystem (`emfs:/<wheel>`, a micropip scheme that reads straight from Pyodide's
filesystem), and passes every other package on to micropip, see
`check_pyodide_install.mjs`.  Everything else in the bootstrap runs unmodified.

`../browser_check/` covers that, in a real browser, and also draws a widget, a figure and
an animation.  It would have caught the time the bootstrap installed from PyPI whatever
the site contained.

## Usage

```bash
RTB_PURE_WHEEL=1 python -m build --wheel --outdir docs/lite/pypi
npm --prefix docs/lite/wasm_check install   # once per checkout, or after a version bump
node docs/lite/wasm_check/check_pyodide_install.mjs docs/lite/pypi/*.whl
```

Exits 0 and prints `PASS: Pyodide install check succeeded.` on success. On failure,
prints a diagnostic block explaining the likely cause (almost always: a
dependency of RTB that cannot be installed in Pyodide, needing a workaround in
`_rtb_nb_bootstrap.py`) and how to fix it,
followed by the real traceback.

## Known flakiness

Each run installs ~35 packages fresh from a CDN (jsdelivr). An occasional network
hiccup can produce a failure (e.g. `AbortError`) unrelated to any real regression —
if a CI run fails here, a straight re-run is a reasonable first move before
assuming something's actually broken. `pyodide`'s own local package cache
(`node_modules/.cache` after the first run) makes repeated local runs faster and
more reliable than the first one.

## Version pinning

`package.json` pins the `pyodide` npm package to `0.27.6`, matching
`docs.yml`'s deliberate `jupyterlite-pyodide-kernel==0.6.1` pin (see that
workflow's own comment for the full reasoning: newer kernel releases bundle a
JSPI-dependent Pyodide that either crashes or silently races on browsers
without JSPI support, e.g. Safari as of 2026-08). These two pins have to move
together — if one changes without the other, this check silently stops
representing what the real deployed site actually runs, which could let it
pass while the real site fails, or vice versa. Once that kernel pin is
revisited, this one needs updating in the same change.
