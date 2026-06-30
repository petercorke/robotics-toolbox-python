# Technical Debt

## Robot class hierarchy redesign

### Background

`ERobot.py` is dead code: it is a 5-line pass-through alias for `Robot` with a
commented-out deprecation warning. `Robot` (in `Robot.py`) IS what ERobot was —
the ETS-specification robot.

The three robot types differ in their **specification language**, not their
underlying representation — all three ultimately compile to ETS:

| Class | Specification | Underlying |
|-------|--------------|------------|
| `Robot` (= ERobot) | ETS directly | ETS |
| `DHRobot` | DH parameters (α, a, d, θ) stored on `DHLink` | converted to ETS via `DHLink.ets` |
| `PoERobot` | Screw axes (Twist3) | converted to ETS via `_update_ets()` |

`DHRobot(Robot)` and `PoERobot(Robot)` inheriting from `Robot` is therefore not
wrong at the representation level — they ARE ETS robots underneath. `RobotURDF`
(subclass of `Robot`) fits correctly: URDF robots are natively ETS.

### The type problem

`DHLink` plays two roles simultaneously: it is a *specification carrier* (holds
DH parameters `alpha`, `a`, `d`, `theta`) AND it is a `Link` subclass (the
compiled ETS form). Because `Robot(BaseRobot[Link])` pins `LinkType=Link`,
pyright sees all links as `Link` and cannot see the DH parameters, causing 67+
issues in DHRobot.py.

### Proposed fix

Make `Robot` generic (don't pin `LinkType`) and pin at the concrete subclasses:

```python
RobotLinkType = TypeVar("RobotLinkType", bound=Link)

class Robot(BaseRobot[RobotLinkType], RobotKinematicsMixin): ...  # still generic
class DHRobot(Robot[DHLink]): ...     # pins to DHLink
class PoERobot(Robot[PoELink]): ...   # pins to PoELink
class RobotURDF(Robot[Link]): ...     # pins to Link (as now)
```

This eliminates all `links` property overrides and `# type: ignore[union-attr]`
workarounds, and makes `for link in robot` automatically yield the correct
subtype.

### Naming

`Robot` is a poor name for what is specifically an *ETS-based robot*. A rename
to `ETSRobot` would be accurate, but that is an API break. Consider doing this
in a major version increment, retaining `Robot` as a deprecated alias.

`ERobot` can be deleted outright (it has no functionality).

**Best done alongside:** the ETS/fknm refactor — the type picture simplifies
dramatically if hierarchy and representation are both cleaned up together.

---

## Forward-looking design: one Robot class, polymorphic Link.A(q)

### Core idea

A robot's fundamental job is to concatenate transforms — one per link — to find
the end-effector pose. The right abstraction is:

```python
class Link:          # abstract base
    def A(self, q: float) -> SE3: ...   # local transform for this link

class DHLink(Link):  # DH parameters: α, a, d, θ
    def A(self, q: float) -> SE3:
        # T = Rot(z, θ+q) * Trans(z,d) * Trans(x,a) * Rot(x,α)  — classic DH

class PoELink(Link): # screw axis S
    def A(self, q: float) -> SE3:
        # T = exp(S * q)

class ETSLink(Link): # ETS sequence
    def A(self, q: float) -> SE3:
        # evaluates ETS; can dispatch to fknm for speed
```

`Robot` holds a `list[Link]` and FK is simply:

```python
def fkine(self, q):
    T = SE3()
    for link, qi in zip(self.links, q):
        T = T * link.A(qi)
    return T
```

### Factory constructors (spatialmath style)

```python
robot = Robot.DH([DHLink(...), ...])        # DH-parameter robot
robot = Robot.ETS([ETSLink(...), ...])      # ETS robot
robot = Robot.PoE([PoELink(...), ...], T0)  # PoE/screw robot
robot = Robot.URDF("puma560.urdf")          # URDF → ETSLink internally
```

Each returns a plain `Robot`. No subclasses needed. Typing is clean:
`links: list[Link]`, `A(q) -> SE3` on every link.

### Speed vs. pedagogy

- **`DHLink.A(q)`** — four-line DH formula, directly readable from any robotics
  textbook. Slower than fknm but pedagogically transparent.
- **`PoELink.A(q)`** — `exp(S * q)`, directly maps to screw theory. Same.
- **`ETSLink.A(q)`** — dispatches to fknm C extension for the fast path. This
  is Jesse's optimisation, preserved where it matters (ETS/URDF robots).

### Why the current design diverged

The original design had separate subclasses (`DHRobot`, `ERobot`) with genuinely
different internal representations. Jesse then converted *everything* to ETS
under the hood (for fknm) without unifying the class hierarchy. The result:
three robot classes that all secretly run the same ETS machinery, but with 67+
type errors because the link types don't unify cleanly.

### What this fixes

- No Generic iterator problem — `links: list[Link]`, always.
- No `links` property override needed — there is only one `Robot`.
- `DHLink.alpha`, `DHLink.a`, `DHLink.d`, `DHLink.theta` remain accessible
  naturally — the specification lives on the link object, not in a parallel table.
- `ERobot` dead code can be deleted.
- Type-checking is clean: `Link.A(q) -> SE3` is the sole interface contract.

### What to preserve from Jesse's work

The fknm C extension is a genuine performance win for ETS robots. The batch FK
path (`robot._fkine_fknm(q)`) that operates on the full ETS array can survive
as an optimised overload on `Robot.fkine()` when all links are `ETSLink`. This
is an internal implementation detail, not a design constraint.

---

## `accel_x` in Dynamics.py is incomplete (NameError at runtime)

`DynamicsMixin.accel_x` (Dynamics.py ~L1294) computes operational-space forward dynamics but references variables `T` and `J` that are never defined in scope. The function would raise `NameError: name 'T' is not defined` if called.

From the surrounding comments:
```
# Ja = T J       (T maps geometric → analytical Jacobian, J is self.jacob0(qk))
# Jad = Td J + T Jd
# assume Td = 0  → Jad = T Jd
```

The broken line is:
```python
xdd[k, :] = T @ (Jd @ qdk + J @ qdd)   # T and J undefined
```

Likely correct form (given Td=0 assumption):
```python
J0 = self.jacob0(qk)
T  = Ja @ np.linalg.pinv(J0)            # analytical-to-geometric transform
xdd[k, :] = T @ Jd @ qdk + Ja @ qdd    # = T Jd qd + Ja qdd
```

**Action:** fix the implementation and add a test that exercises `accel_x` against a known robot (e.g. Puma560).

---

## Link deepcopy drops collision geometry (coal pickling limitation)

`Link.__deepcopy__` silently drops coal `CollisionObject` instances because the
coal library does not support pickling. The workaround warns at runtime when
shapes are lost.

The root cause is that a fresh `DHLink` can reach the coal objects of an
unrelated URDF robot through shared class-level state in `Link` or `Robot`
(exact attribute TBD). This means:

- Copied DH links that happen to run after URDF robot tests lose nothing (DH
  links have no collision shapes), but the warning path is never exercised in
  isolation.
- If the shared reference is ever followed for other purposes (e.g. iteration,
  serialisation) it could cause similar failures or unexpected aliasing.

**Proper fix (Option 2):** obtain the full `deepcopy` traceback with
`--tb=long` when running `test_ERobot.py` followed by
`test_Link.py::TestDHLink::test_copy`, trace which attribute chain connects the
fresh `DHLink` to a coal object, and remove or weak-ref that shared state.

---

## ETS / fknm / frne refactor

### Context

`fknm.cpp` is a C++ extension (raw CPython API + Eigen) that accelerates FK,
Jacobian, Hessian, and IK. `frne.c` is CPython glue around `ne.c` (pure C
Newton-Euler maths) used by Dynamics. Both are built via scikit-build-core /
CMakeLists.txt and the Pyodide WASM wheel is already built in CI
(`CIBW_PLATFORM: pyodide`, uploaded as a GitHub release asset).

### Phase 0 — Tests first (prerequisite for all phases)

Write test coverage before touching any C code:

- `eval()` / `fkine()` with and without fknm — mock the C import to force the
  Python fallback path
- Symbolic (SymPy) inputs through `fkine()`, `jacob0()`, `jacobe()`
- `jacob0`, `jacobe`, `hessian0`, `hessiane` numerical results against a
  reference robot (Puma560 — DH params and reference values are well-known)
- Dynamics: `rne()` via frne/ne against known torque values
- Pyodide simulation: fknm import mocked as `ImportError`, all paths correct

### Phase 1 — Facade module

**Problem:** `from roboticstoolbox.fknm import ETS_fkine, ...` is a hard import
of the `.so`. If unavailable the module fails to load. The fallback is a
`try/except BaseException: pass` in `eval()` that swallows real bugs. The
symbolic path is detected after the fact via `dtype == 'O'`.

**Fix:**

- Rename the C extension to `_fknm_c` so that `roboticstoolbox/fknm.py` can be
  a pure Python module
- `fknm.py` tries `from roboticstoolbox._fknm_c import ...`; on `ImportError`
  provides pure Python implementations of `ETS_init`, `ETS_fkine`, `ETS_jacob0`,
  `ETS_jacobe`, `ETS_hessian0`, `ETS_hessiane`, and the IK wrappers
- Pure Python implementations are the existing fallback code consolidated from
  `eval()`, `jacob0()` etc. in ETS.py — they already exist, just scattered
- Symbolic detection (`_is_symbolic(q)`) lives inside each facade function;
  callers never check
- `eval()` and friends in ETS.py become a single unconditional call — no
  `try/except`, no `dtype == 'O'` guard

### Phase 2 — BaseETS redesign and unified C++ state management

#### Unified design: parallel structure across fknm and frne

Both C extensions share the same lifecycle pattern:

| Concept | ETS / fknm | DHRobot / frne |
|---|---|---|
| C++ handle | `self._fknm` | `self._frne` (was `_rne_ob`) |
| Dirty flag | `self._fknm_stale` | `self._frne_stale` (was `_dynchanged`) |
| Build/update C++ object | `_copy_to_cpp()` (was `_update_internals`) | `_copy_to_cpp()` (was `_init_rne`) |
| Dirty on mutation | `@_dirties_fknm` decorator | `@_dirties_frne` decorator (was `@_listen_dyn`) |
| Guard before C call | `if self._fknm_stale: self._copy_to_cpp()` | `if self._frne_stale: self._copy_to_cpp()` |
| Free C++ object | implicit (GC) | `delete_rne()` — kept as explicit early free |

Both use **lazy rebuild**: the dirty flag is set on mutation; `_copy_to_cpp()` is
called only when a C function is about to be invoked. This eliminates the need for
the `_building` context manager (multiple mutations during construction just set the
flag once; `_copy_to_cpp()` runs on the first C call).

**`@_dirties_fknm`** (on `BaseETS` mutation methods):
```python
def _dirties_fknm(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._fknm_stale = True
        return result
    return wrapper
```
Applied to `__setitem__`, `__delitem__`, `insert` — `MutableSequence` derives
`append`, `extend`, `pop`, `remove` from these three, so all mutations propagate.

**`@_dirties_frne`** (on `DHLink` property setters — rename of `@_listen_dyn`):
Same signal path as before (`robot.dynchanged()` → `robot._frne_stale = True`),
plus `link._hasdynamics = True`. The second effect is intentional and stays: setting
a dynamic parameter IS the act of declaring the link has dynamics — the decorator is
the single chokepoint where both effects belong. Name change only.

**`_copy_to_cpp()` on DHRobot** frees existing C allocation before reallocating:
```python
def _copy_to_cpp(self):
    if self._frne is not None:
        delete(self._frne)      # free old PyMem_RawMalloc'd Robot struct
    self._frne = init(self.n, self.mdh, L, -self.gravity)
    self._frne_stale = False
```
This makes `delete_rne()` redundant for the update-before-compute case.
`delete_rne()` is kept as a public "free memory now" escape hatch, but is no longer
required for correctness. With nanobind (Phase 3), it becomes fully redundant
because nanobind binds the Robot struct with a proper C++ destructor — `_frne`
going out of scope triggers the free automatically.

#### BaseETS structural fix

**Problem:** `BaseETS(UserList)` stores its list in `UserList.self.data` (plain
attribute) but shadows it with a `@property` redirecting to `self._data`. The
`data` setter does not call `_copy_to_cpp()`. Mutation via `self.data.append(x)`
bypasses all hooks.

**Fix:**

- Drop `UserList`; inherit from `collections.abc.MutableSequence` instead
- Backing store: `self._data: list[ET]` — no property hack needed
- Implement the five required abstract methods (`__getitem__`, `__len__`,
  `__setitem__`, `__delitem__`, `insert`) decorated with `@_dirties_fknm`
- `MutableSequence` provides `append`, `extend`, `pop`, `remove`, `__contains__`
  etc. for free, all routing through the decorated methods
- `data` property (if needed externally) becomes simple read-only `return self._data`
- No `_building` context manager needed — lazy rebuild handles bulk construction

### Phase 3 — nanobind port

**Problem:** `fknm.cpp` and `frne.c` use the raw CPython C API (`PyArg_ParseTuple`,
`PyObject*`, `PyMethodDef` boilerplate). nanobind gives the same performance with
far less boilerplate, safer reference counting, and better Emscripten/Pyodide
support.

**Fix:**

- Port `fknm.cpp` to nanobind; Eigen (already present) stays unchanged —
  nanobind and Eigen are natural companions
- Port `frne.c` (CPython glue only) to a nanobind `.cpp` binding file; `ne.c`
  maths is pure C with no Python API and stays as-is (rename to `ne.cpp` only
  if C++ features are needed, otherwise leave it)
- Both build via the existing CMakeLists.txt / scikit-build-core pipeline —
  scikit-build-core is already in pyproject.toml
- Verify `build_pyodide` CI job (`CIBW_PLATFORM: pyodide`) still passes;
  keep `continue-on-error: true` during transition
- The facade from Phase 1 means `ETS.py` is untouched by this phase
