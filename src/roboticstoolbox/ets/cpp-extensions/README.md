# Fast/optimized kinematics (fknm)

## Synopsis

nanobind C++/C extension that accelerates the ETS hot path: forward
kinematics, Jacobians, Hessians, and IK (Newton-Raphson, Gauss-Newton,
Levenberg-Marquardt), used by `ETS`/`Robot`.

This is one of two such extensions in the codebase — the other,
**frne** (`_frne_c`, recursive Newton-Euler inverse dynamics, used by
`DHRobot.rne()`), lives at `robot/cpp-extensions/` since it's a
dynamics concern, not an ETS one. See that directory's own README for
frne specifics; this file covers fknm plus the architecture patterns
shared by both.

It's optional: `ets/fknm.py` is a pure-Python facade that tries to
import the compiled extension and transparently falls back to an
equivalent pure-Python implementation when it's unavailable
(Pyodide/WASM builds without the extension compiled, CI paths that force
the Python implementation, or symbolic/SymPy inputs, which the C++ side
can't handle).

## How it works

* Built via nanobind, driven by the top-level `CMakeLists.txt` /
  scikit-build-core. The compiled modules install as
  `roboticstoolbox._fknm_c` / `roboticstoolbox._frne_c` at the top level of
  the installed package (`CMakeLists.txt`: `DESTINATION roboticstoolbox`) —
  that install location is independent of where this source directory lives.
* `fknm.py` / `frne.py` each do `from roboticstoolbox._fknm_c import ...`
  (or `_frne_c`) inside a `try/except ImportError`, and provide a pure-Python
  equivalent for every function on the `except` path. Symbolic inputs are
  detected per-call and routed to the Python path even when the extension is
  available.
* **Lazy serialization into the extension.** The C++ side needs its own copy
  of the robot's state (`ETS`/`ET` structs for fknm, a `Robot` struct for
  frne) — Python objects can't be handed across the boundary directly. Rather
  than re-serializing on every mutation, `ETS`/`BaseETS` and `DHRobot` each
  keep a C++-side mirror object (`_fknm`, `_frne`) plus a dirty flag
  (`_fknm_stale`, `_frne_stale`). Mutating methods are wrapped with the
  `@_dirties_fknm` / `@_dirties_frne` decorators, which just set the flag
  after the wrapped call — they don't touch the mirror themselves. The
  mirror is only actually rebuilt (`_copy_to_cpp()`) the next time a C++
  function is about to be called and the flag is set. This means a whole
  sequence of mutations (e.g. building up an `ETS` element by element, or
  setting several dynamic parameters on a `DHLink`) costs one rebuild total,
  not one per mutation, and a mirror that's never used for a C++ call is
  never built at all.

## Gotcha!

Internal transform matrices (ET/ETS results, Jacobians) are stored
**column-major** — Eigen's native layout, not NumPy's default row-major
layout. This is intentional (it's what `Eigen::Map` wants) and callers don't
need to do anything about it: the nanobind bindings (`EigenRef4d`,
`EigenRefJd`) accept C-contiguous, F-contiguous, or arbitrary-stride NumPy
arrays and let Eigen copy element-by-element using the real strides, so no
Python-side layout conversion is ever required before calling into C++.

# Files

| File | Purpose |
| ---- | ------- |
| Eigen | Vendored Eigen headers (header-only) |
| fknm_nb.cpp | nanobind glue for `_fknm_c` — binds FK, Jacobian, Hessian, IK, and ET init/update |
| ik.cpp | Fast inverse kinematics (Newton-Raphson, Gauss-Newton, Levenberg-Marquardt) |
| ik.h | " |
| linalg.cpp | SE(3) matrix operations |
| linalg.h | " |
| methods.cpp | Jacobians, Hessians |
| methods.h | " |
| structs.cpp | ET, ETS mirror (currently unused/dead — not part of the `_fknm_c` CMake sources) |
| structs.h | " |
