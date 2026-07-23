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

### Related: `Robot.rne()`'s misuse guard checks `mdh`, not class identity

`Robot.rne()` (`Robot.py`) assumes Featherstone's joint-last-in-segment
structure. A cheap runtime guard rejects the incompatible case:

```python
assert getattr(self, "mdh", True), ...
```

Two design questions came up while adding this guard 2026-07-21, both worth
recording since they'll resurface if the class hierarchy redesign above
happens:

1. **Blocklist vs. allowlist.** First attempt was a class-name blocklist
   (`assert not any(c.__name__ == "DHRobot" for c in type(self).__mro__)`).
   An allowlist (`Robot`/`ERobot`/`URDFRobot` by name) was considered and
   rejected: it would have wrongly rejected `PoERobot`, a fully compliant
   type (its `_update_ets()` also appends the joint ET last) that was easy
   to overlook — demonstrating exactly the fragility an allowlist has here.
2. **Class identity was the wrong axis entirely.** `DHLink._to_ets()` shows
   joint-last compliance actually tracks the `mdh` flag, not the `DHRobot`
   class: the MDH branch reorders a revolute link's `d` translation to
   *precede* the joint rotation (valid — a z-rotation and z-translation
   about/along the same axis commute), so the joint ET is last regardless
   of `d`. A `DHRobot(mdh=True)` instance is therefore structurally fine
   through `Robot.rne()`, while `mdh=False` is not — the class-name check
   would have wrongly rejected the compliant MDH case too. The final guard
   checks `self.mdh` directly (defaulting to `True` via `getattr` for
   non-DHRobot types, which have no DH-convention concept and are already
   guaranteed joint-last some other way).

Verifying the MDH case numerically (nonzero `d`, `alpha`, mass, and full
inertia tensor, compared against `rne_python()`) surfaced a real, separate
bug it would otherwise have masked: `SpatialInertia(m=link.m, r=link.r)` in
`Robot.rne()`'s inertia accumulation never passed `I=link.I` — the
rotational inertia tensor was silently dropped for every link, for every
`Robot`/`ERobot`/`URDFRobot`/`PoERobot`/MDH-`DHRobot` call, not just the MDH
edge case. Fixed alongside this guard. Not caught earlier because the
`TwoLink`-based `Robot.rne()`-vs-`rne_python()` equivalence tests used
default (zero) inertia; the tests that *did* set `inertia=True` only
compared the C path against `rne_python()`, never `Robot.rne()`.

**Revisit when the class hierarchy redesign above happens.** If `DHRobot`
stops being a `Robot` subclass (e.g. becomes `Robot[DHLink]` under the
generic-`LinkType` proposal, or the "one Robot class, polymorphic `Link.A(q)`"
design below is adopted), the whole question may become moot — either there's
only one `Robot` class and the guard is unnecessary, or the DH-convention
distinction is structurally explicit rather than a name-based/attribute-based
runtime check.

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

### CI evidence this is currently broken (2026-07-04)

Once the `rtb-data`/`coal`/`robot_descriptions` dependency issues were fixed
(see elsewhere in this file), `main`'s CI surfaced a **Python-3.10-only**
failure, reproducing on both Windows and macOS: `AttributeError: <class
'roboticstoolbox.robot.ETS.ETS'> does not have the attribute 'ETS_jacob0'`
(also `'ETS_jacobe'`, `'ETS_hessian0'`, `'ETS_hessiane'`) — 22-74 test
failures per job depending on platform. Being Python-version-specific
rather than OS-specific points at exactly the facade/fallback gap Phase 1
below describes: the C extension likely isn't loading on 3.10 (build or
ABI mismatch) and the pure-Python fallback attributes it's supposed to
provide don't exist yet because Phase 1 hasn't been done. Not investigated
further — needs a dedicated debugging pass, not a quick fix.

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

### Phase 3.5 — per-ET result buffer and sparse op functions (post-nanobind)

**Problem:** `rx`, `ry`, `rz`, `tx`, `ty`, `tz` each write all 16 elements of
the output matrix on every call, even though only 4 (rotation) or 1 (translation)
elements actually change between evaluations at different joint angles. This is
efficient for the current shared scratch buffer (`ret` in the FK loop), but
leaves performance on the table for the common IK/Jacobian pattern where the
same joint ET is evaluated repeatedly with different `q` values.

**Fix (depends on Phase 2 per-ET buffer):**

Once each joint ET owns its own `double[16]` result buffer (analogous to how
static ETs already own `et->T`):

- Initialise the buffer to identity once at `ET_init` time
- Each op function only overwrites the elements that depend on eta:
  - `rx`/`ry`/`rz`: 4 elements (the 2×2 rotation sub-block)
  - `tx`/`ty`/`tz`: 1 element (the relevant translation component)
- The matrix multiply `ret * U` in `_ET_T` uses the per-ET buffer directly
  instead of copying into a shared scratch

This eliminates ~12 zero-stores per rotation ET and ~15 stores per translation
ET in every FK/Jacobian/Hessian call.

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

---

## `tools/p_servo.py` — cross-package import papered over with a lazy import

### Background

`roboticstoolbox/__init__.py` loads `roboticstoolbox.tools` before
`roboticstoolbox.robot`, because `robot/*.py` modules need things from `tools`
(`rtb_get_param`, `ArrayLike`, …) at their own module scope. `tools/p_servo.py`
sits in `tools/` but needs `Angle_Axis` from the `fknm` facade, which lives in
`robot/` (`robot/fknm.py`) since it's a robot-kinematics concept. Any
module-level `from roboticstoolbox.robot.fknm import Angle_Axis` in
`p_servo.py` forces Python to run `robot/__init__.py` *before* `tools/__init__.py`
has finished — which breaks, because `robot/*.py` in turn expects `tools` to
already be fully initialised. True circular dependency between the two
packages, not a matter of which subfolder `fknm.py` lives in.

**Current fix (workaround, not a real solution):** the import was moved inside
the `angle_axis()` function body so it's deferred until first call, well after
package init completes. This works but means `p_servo.py`'s dependency on
`robot` is now invisible at a glance — a future edit that hoists it back to
module scope re-breaks the import chain with no obvious cause (this exact bug
recurred once already, see the `robot/cpp-extensions` move in this session).

### Proper fix

`p_servo`/`angle_axis` doesn't conceptually belong in `tools/` — it's a
robot pose-error/servoing function, not a generic tool, and its only
non-trivial dependency (`Angle_Axis`) is a robot-kinematics primitive. Move
`p_servo.py` into `robot/` (or a `robot/control.py`-style module) so the
dependency direction is `robot → robot`, not `tools → robot`, eliminating the
need for the lazy-import workaround entirely. Requires updating the handful of
call sites and the `roboticstoolbox/tools/__init__.py` / top-level `__init__.py`
re-exports that expose `p_servo`/`angle_axis` at package scope.

---

## `rtb-data`

`rtb-data` is a separate PyPI package (data files: meshes, xacro/URDF sources,
etc.) built from the `rtb-data/` subdirectory of this repo, published
independently of `roboticstoolbox-python` itself. This section groups
everything currently known to be wrong or incomplete about it.

### Publishing is manual and has drifted from `main`

`roboticstoolbox-python`'s own release process (`release-please` +
`release.yml`) knows nothing about `rtb-data`. As of 2026-07-03, `rtb-data/`
on `main` had drifted well ahead of what's on PyPI (missing
`rtb-data/pyproject.toml` entirely at one point — see the "add missing
rtb-data package config and data files" fix), and several `test_models.py`
cases failed on CI as a direct result (xacro files the installed PyPI package
didn't have yet).

**Proposed fix:** automatically publish a new `rtb-data` release alongside
`roboticstoolbox-python`'s own release, **but only if `rtb-data/` actually
changed** since the last `rtb-data` publish. The whole point of `rtb-data`
being a separate package is that it's large (meshes, STL/OBJ files) and
should get pushed infrequently — don't turn this into "publish rtb-data on
every roboticstoolbox-python release" regardless of whether anything in it
moved. A GH Actions step (likely in `release.yml` or a new workflow,
triggered the same way) that:

1. Diffs `rtb-data/` against the tree at the last `rtb-data` publish (tag or
   recorded commit SHA)
2. If unchanged, no-op
3. If changed, bump `rtb-data/pyproject.toml`'s version and publish to PyPI

would keep the two packages in sync without manual "did I remember to publish
rtb-data" steps, while preserving the "infrequent, only-when-needed" intent.

### Move `rtb-data/` into a `packages/` folder

`rtb-data/` currently sits at repo root alongside the main `roboticstoolbox`
source tree, even though it's an independently-versioned, independently
-published PyPI package. Only two files reference the path today
(`pyproject.toml`'s `sdist.exclude`, and `rtb-data/pyproject.toml` itself),
so the move is cheap. Worth doing since `GRAPHICS-BACKEND.md` and
`SWIFT-MPL-SPLIT.md` both describe splitting graphics backends into further
sub-packages — better to establish the `packages/` convention while there's
one occupant than to reshuffle after two or three exist. Also sequences well
with the publishing-automation fix above: build that config against the new
path from the start rather than writing it against `rtb-data/` and moving it
later. Queued behind the 1.3.1 maintenance release (2026-07-03) —
deliberately not bundled with it, to avoid adding another moving part to an
already-unstable `main`.

**Proposed fix:** `git mv rtb-data packages/rtb-data`, update `sdist.exclude`,
land as its own small PR to `main`, independent of any release-process work
in flight.

### Bundled xacro tree has a naming mismatch: `LBR` (Kuka)

Found 2026-07-05, fixed 2026-07-20. `LBR.py` loads
`"kuka_description/kuka_lbr_iiwa/urdf/lbr_iiwa_14_r820.xacro"` from the
bundled `rtb-data` xacro tree (an explicit `.xacro` suffix path, so this
doesn't go through `robot_descriptions` at all — see below). The xacro file
itself does:

```xml
<xacro:include filename="$(find kuka_lbr_iiwa_support)/urdf/lbr_iiwa_14_r820_macro.xacro"/>
```

but the bundled directory is named `kuka_lbr_iiwa` (no `_support` suffix), so
`$(find kuka_lbr_iiwa_support)` fails to resolve —
`xacrodoc.packages.PackageNotFoundError: Package not found:
kuka_lbr_iiwa_support`.

The originally-proposed fix (rename the bundled directory to
`kuka_lbr_iiwa_support/`, requiring an `rtb-data` release) turned out not to
work even in isolation: `XacroDoc.from_file()`'s package auto-discovery
doesn't register a directory as a package purely by name-matching against
its own `$(find ...)` references — nothing in the bundled data carries a ROS
`package.xml` marker for it to key off. So renaming the directory alone,
tested empirically against a throwaway copy, still raised the same
`PackageNotFoundError`.

**Actual fix:** `URDFRobot.__init__`/`URDF_file()` gained an
`extra_packages: dict[str, str] | None` kwarg (mirroring the existing
`patch=` convention for known-broken upstream references) that registers
additional package-name aliases via `xacrodoc.packages.update_package_cache()`
before parsing. `LBR.py` now passes
`extra_packages={"kuka_lbr_iiwa_support": "kuka_description/kuka_lbr_iiwa"}`.
No `rtb-data` release needed — fixed entirely within `roboticstoolbox-python`.

### `Valkyrie` and `Fetch` load via `robot_descriptions`, whose supplied files are broken upstream — patched on the way in

Found 2026-07-05, fixed the same day. Unlike LBR, `Valkyrie.py`
(`URDF_read("valkyrie")`) and `Fetch.py`
(`super().__init__("fetch", ...)`) both pass a bare name with no
`.urdf`/`.xacro` suffix, which `URDFRobot.py`'s `URDF_file()` already routes
through `_load_urdf_from_RD()` — i.e. these do **not** use the bundled
`rtb-data` tree at all, they're already on `robot_descriptions`. Both failed,
but the root cause was upstream data, not the loading path:

- **Valkyrie**: RD's `valkyrie_description` (robot_descriptions v2.0.0)
  resolves to `nasa-urdf-robots/val_description/model/robots/valkyrie_sim.urdf`
  (repo `gkjohnson/nasa-urdf-robots`, commit `54cdeb1d`, 2020-11-14). Despite
  the `.urdf` extension and RD listing it as `formats={URDF}`, the file still
  declared `xmlns:xacro` and contained a live, unexpanded
  `<xacro:v1_pelvis_sensors_usb .../>` macro call (line 2432) — it was never
  actually compiled. There are **zero `.xacro` files anywhere in the whole
  cloned repo**, so the macro definition simply isn't present upstream — an
  incompleteness in `gkjohnson/nasa-urdf-robots` itself.
- **Fetch**: RD's `fetch_description` (robot_descriptions v2.0.0) resolves to
  `roboschool/roboschool/models_robot/fetch_description/robots/fetch.urdf`
  (repo `openai/roboschool`, commit `c8ee2812`). Line 655 had
  `<sensor:camera name="rgb">`, an XML element using the `sensor:` namespace
  prefix, but the root `<robot name="fetch">` element never declared
  `xmlns:sensor="..."` anywhere in the document — old pre-SDF Gazebo
  camera-sensor syntax (ROS Fuerte/Groovy era, ~2012) that was never
  well-formed XML. `xml.parsers.expat` correctly rejected it
  (`unbound prefix`).

Neither was fixable upstream: `openai/roboschool` is **archived**
(confirmed via GitHub API, `archived: true`, deprecated ~2023-04) so it
can't take PRs at all; `gkjohnson/nasa-urdf-robots` isn't archived but
"fixing" it would mean *inventing* a macro definition from scratch (no
source of truth exists for what `v1_pelvis_sensors_usb` should actually
contain), which isn't a responsible thing to submit upstream.

**Fix applied:** Valkyrie is referenced in the RVC3 textbook, so a working
model is required — not just optional cleanup. `Fetch` was equally easy to
fix, so both were patched rather than one being deleted. Added an optional
`patch: Callable[[str], str]` hook to `URDF_file`/`URDF_read`/
`URDFRobot.__init__` (`URDFRobot.py`) that runs on the raw file text before
xacro/XML processing. Each model defines a small local patch function
(`_patch_valkyrie_urdf` in `Valkyrie.py`, `_patch_fetch_urdf` in `Fetch.py`)
that surgically regex-strips the one broken element — both are
Gazebo-simulation-only blocks (a sensor plugin macro call, a camera plugin
config block respectively) with zero bearing on kinematics, dynamics, or
geometry, so dropping them is safe. Each patch function's docstring records
the exact robot_descriptions version and upstream commit it targets, so if
either upstream repo is ever fixed, the regex simply stops matching (a
no-op) and the patch can be confirmed-safe-to-delete.

### Docstrings for RD-loaded models don't credit/link `robot_descriptions`

At least 8 model classes load via `robot_descriptions` under the hood
(`Fetch`, `Frankie` (as `panda`), `Jaco`, `PR2`, `UR10`, `UR3`, `UR5`,
`Valkyrie`, `YuMi` — found by scanning for bare-name, no-suffix arguments to
`URDF_read`/`URDFRobot.__init__`), but none of their docstrings mention that
the underlying model comes from `robot_descriptions` at all — a user reading
e.g. `UR5`'s docstring has no way to know it's fetched from a third-party
package (or to go find that package's own model listing/license/source
repo).

**Proposed fix:** add a line to each affected class docstring naming
robot_descriptions as the source, with an anchor link to its GitHub repo,
e.g.:

```rst
This model is provided by `robot_descriptions
<https://github.com/robot-descriptions/robot_descriptions.py>`_.
```

Same URL already used for the clickable terminal hyperlink in
`URDFRobot.py`'s error messages (`_RD_URL`) — worth a shared constant/snippet
so the two don't drift. Mechanical, low-risk change across ~8 files; good
candidate for a single small PR once the runblock-cleanup pass is done.

---

## `intro.rst` still describes PyBullet as the collision backend

### Background

The Toolbox switched its collision backend from PyBullet to `coal`
(pyproject.toml's `collision` extra is `["coal; sys_platform != 'win32'",
"trimesh"]`), and README.md/docs/source/install.rst have been updated to
match (2026-07-03). `docs/source/intro.rst`'s "Collision checking" section
(around line 650) was missed — it still says checking is "dramatically
improved... using [PyBullet]_" and carries a `[PyBullet]_` citation in the
references list. This is prose adapted from the ICRA2021 paper, not a plain
install matrix, so it wasn't rewritten opportunistically; it needs a
deliberate pass to reword the narrative and swap/remove the citation.

### Proposed fix

Rewrite the "Collision checking" section to describe `coal` (GJK/EPA,
`CollisionObject`/`BVHModelOBBRSS`, primitive shapes) instead of PyBullet,
update or drop the `[PyBullet]_` reference, and add a one-line note that
collision checking is unavailable on Windows via pip (see the `coal` Windows
wheel gap noted below).

---

## `coal` has no Windows wheels on PyPI — collision checking is Linux/macOS-only via pip

### Background

`coal` (the actively-maintained FCL/hpp-fcl successor, used by Pinocchio)
publishes wheels for Linux (manylinux) and macOS (incl. arm64) but not
Windows, and its sdist can't build there either (needs `cmeel-assimp>=6.0.5`,
unavailable for Windows on PyPI). This broke `pip install .[dev]` on Windows
CI outright (coal was an unconditional `dev`/`all`/`collision` dependency).
Fixed 2026-07-03 by marking `coal` `sys_platform != 'win32'` in
pyproject.toml — Windows installs now succeed but simply don't get collision
checking; `tests/__init__.py`'s `skip_no_collision_checking` marker and
`CollisionShape.py`'s lazy `_require_coal()` already handle the missing-coal
case gracefully at runtime.

`coal` does have real Windows builds — just via conda-forge (confirmed:
~56 win-64 builds of `coal-python`), not PyPI. So this isn't "coal doesn't
work on Windows," it's "coal isn't pip-installable on Windows."

### Options considered

1. **(current)** No collision checking on Windows via pip; document
   `conda install -c conda-forge coal-python` as a manual escape hatch.
2. Add `python-fcl` (BerkeleyAutomation fork) as a Windows-specific backend —
   it does publish win_amd64/macos-arm64/manylinux-aarch64 wheels on PyPI.
   Not a drop-in: `CollisionShape.py` calls coal's API directly and
   non-trivially (`BVHModelOBBRSS`, `CollisionObject`,
   `DistanceRequest`/`DistanceResult`, `Cylinder`/`Sphere`/`Box`), so this
   means writing and maintaining a second backend with a different API
   shape and (likely) weaker performance — coal superseded python-fcl's
   underlying library for a reason. Worth doing only if Windows collision
   support becomes an actual blocker for someone, not preemptively.

### Proposed fix

None needed unless Windows collision-checking demand shows up — option 1 is
the deliberate resting state, not a stopgap.

---

## Release process: single-branch release-please + stacked PRs on a red `main`

### Background

Two compounding problems made 2026-07 CI work much harder than it should
have been:

1. **release-please here only understands one linear history.** Its config
   (`.github/release-please-config.json`) tracks a single package (`.` =
   roboticstoolbox-python) off `main`'s tip — it has no concept of releasing
   from an older point in history, and no concept of `rtb-data/` as a second
   component (see the `rtb-data` publishing note above). The standing
   `chore: release X.Y.Z` PR it maintains (#520 as of this writing) just
   accumulates every conventional-commit merged to `main`, including
   in-progress/breaking work — it's inert until merged, so leaving it alone
   is always safe, but it also means release-please can't be the vehicle for
   a small out-of-band fix once `main` has diverged from what's actually
   releasable.
2. **Multiple PRs were stacked on top of a `main` with red CI**, each trying
   to fix a different symptom (Windows `coal` install failure, stale
   `rtb-data` causing test failures, etc.) without `main` itself ever going
   green first. Every PR branched from a broken base inherits that breakage,
   so fixing the root cause in one PR doesn't unblock siblings that branched
   before the fix landed — CI never went green across any of them, even
   though each individual fix was correct in isolation.

### Proposed fix / operating pattern

- Land CI-health fixes directly against `main` as small, independent PRs —
  don't stack feature/fix work on top of other unmerged fix PRs. Get `main`
  green first, then resume feature branches from a clean base.
- When a fix needs to ship for the *currently released* version but `main`
  has moved on to in-progress/breaking work (see the rtb-data version-pin
  case, 2026-07-03), release from the last good tag on a dedicated
  maintenance branch (e.g. `maintenance/1.3.x` off `v1.3.0`), out-of-band
  from release-please. This only works cleanly because the release pipeline
  triggers off `release: created` (any tag), not off `main` — see
  `.github/workflows/release.yml`.
- Note there is **no approval gate** on the `pypi` deployment environment
  (`protection_rules: []` — confirmed via the GitHub API 2026-07-03): once a
  GitHub Release is created (or `release.yml` is run via
  `workflow_dispatch`), a successful build publishes to PyPI immediately.
  There is no dry-run; rehearse locally (`python -m build`, install into a
  throwaway venv) before creating the real tag.
- Longer term, fixing the rtb-data multi-package gap (see above) so
  release-please manages both packages would remove the main reason for
  doing out-of-band releases at all.

---

## `test_collision.py` doesn't consistently use `skip_no_collision_checking`

### Background

`tests/__init__.py` provides a `skip_no_collision_checking` marker
specifically so collision tests degrade gracefully when `coal` isn't
installed (e.g. Windows, per the `coal` Windows-wheel gap noted above).
`test_ELink.py`, `test_ERobot.py`, and `test_Robot.py` use it correctly.
`test_collision.py` itself does not (or not consistently) — on Windows CI
(2026-07-04, once the `coal` install fix let Windows jobs reach the Test
step at all) it produced 52 hard failures, all `ImportError: The 'coal'
package is required for collision functionality`, instead of skips.

### Proposed fix

Audit `test_collision.py`'s test classes and apply `@skip_no_collision_checking`
(or an equivalent module-level `pytestmark`) wherever a test exercises real
collision geometry rather than the `collision=False` guard paths.

---

## Fixed: stale pre-step residual accepted as solution in `IKSolver._solve` (was `test_IK_GN3` flake)

### Background

Seen failing on `macos-latest, Python 3.12` CI (2026-07-04):
`AssertionError: 1e-05 not greater than 0.05291452734038758` — a
Gauss-Newton IK convergence tolerance check, originally logged here as an
unreproducible numerical flake. Root-caused 2026-07-06: `IKSolver._solve`
(`src/roboticstoolbox/robot/IK.py`) calls `step()`, which computes `E` for
`q` *before* applying that iteration's update, then mutates `q` in place and
hands the mutated array back. `_solve` then checks `E < self.tol` and, if
true, returns the **post-step** `q` — a point whose actual residual was
never checked. For undamped solvers (`IK_GN`, `IK_NR`) this update can
overshoot arbitrarily, so a q with real error >0.05 (or worse — measured up
to ~1.5 in a 500-seed sweep) was routinely reported as a converged, tol-1e-6
success. This affected `IK_NR`, `IK_GN`, and `IK_LM` (all share `_solve`);
roughly half of "successful" `IK_GN` solves on the UR5 test case were
actually invalid.

The compiled solver path (`_IK_loop` in
`src/roboticstoolbox/robot/cpp-extensions/ik.cpp`) does not have this bug —
it evaluates error *before* deciding whether to step, and only steps when
not yet converged, so it never returns an unverified post-step point. That
Python/C++ behavioral divergence is what led to re-investigating this.

### Fix

`_solve` now snapshots `q` before calling `step()` and, on convergence,
returns that pre-step snapshot rather than the mutated array `step()`
handed back — matching the C++ semantics. See the fix on
`bug/ik-gn-stale-residual`.

---

## Fixed: `tol` is a quadratic residual, so default IK tests under-guaranteed pose accuracy

### Background

`test_DHRobot.py::test_ikine_LM` and `test_blocks.py::RobotBlockTest::test_ikine`
intermittently (the former) or deterministically at `seed=0` (the latter)
failed a `places=4` (~5e-5) check on `np.linalg.norm(T - fkine(sol.q))`, even
though `sol.success` was `True`. Root cause: `IKSolver`'s `tol` bounds the
*quadratic* weighted angle-axis error
:math:`E = \tfrac{1}{2}\vec{e}^\top\mat{W}_e\vec{e}`, not the linear-scale
pose error. With the default `tol=1e-6`, the actual pose error is only
guaranteed to be on the order of :math:`\sqrt{2\cdot\text{tol}} \approx 1.4
\times 10^{-3}` — looser than the `places=4` checks assumed. Confirmed
concretely on the Panda `test_ikine` case: `sol.residual = 2.87e-8` (well
under `tol=1e-6`) still coincided with a raw pose difference of `3.27e-4`,
consistent with the quadratic/linear relationship, not a solver defect.

`test_DHRobot.py::test_ikine_LM` additionally had no pinned `seed`, so it
was genuinely non-deterministic between runs (unlike `test_blocks.py`'s
`seed=0`, which just deterministically landed on the same marginal
solution every time — a reproducible near-miss, not a random flake).

### Fix

- Documented the quadratic-vs-linear relationship directly on `tol`'s
  docstring (`IKSolver` and all four solver subclasses in
  `src/roboticstoolbox/robot/IK.py`).
- Both tests now pass `tol=1e-10` explicitly (comfortably bounding pose
  error under `places=4`'s ~5e-5 threshold), and `test_ikine_LM` now pins
  `seed=0` for reproducibility.

---

## Removed: `robot_descriptions` CI caching (was solving the wrong problem)

### Background

PR #530 (2026-07-03) added `actions/cache` steps for `~/.cache/robot_descriptions`
to `ci.yml`'s `test-core`, `test`, and `coverage` jobs, reasoning that the
package's lazy `git clone` of upstream asset repos was "slow and
occasionally flaky" against the per-test `--timeout=50`. That diagnosis was
wrong: `robot_descriptions` had never been added to `pyproject.toml` at all
(see the "missing robot_descriptions dependency" fix, 2026-07-04) — every
run hit `ModuleNotFoundError` immediately, before any clone was ever
attempted. The caching apparatus was solving a plausible-sounding symptom
of a problem that didn't exist yet. Removed 2026-07-04 rather than left in
place now that the real fix has landed.

Two independent reasons this wasn't worth keeping even setting the wrong
-diagnosis issue aside: the cache key (`robot-descriptions-v1-${{
runner.os }}`) is keyed only by OS, not Python version, while the `test`
job's matrix is `os × python-version` — same-OS jobs run in parallel and
all start cold, so within a single CI run it provided no benefit across
the 4 Python versions per OS anyway; and the naming-fallback logic in
`_load_rd_module` (`URDFRobot.py`) that keeps `robot_descriptions`'
implementation-detail name out of user-facing errors is legitimate,
separate, correct UX and was *not* removed — only the CI caching steps
were part of the wrong-theory rabbit hole.

### If CI timing/flakiness resurfaces

Now that `robot_descriptions` is genuinely installed and genuinely
git-clones on first use of models like YuMi/PR2/UR3/5/10/Jaco, real
network-fetch time is incurred every run. If that turns out to matter in
practice, re-add caching with a corrected key that includes
`matrix.python-version` (or better, restructure so only one job per OS
does the network fetch and others restore from it) — don't just restore
this exact removed code.

---

## Python-3.10-specific workaround: `sys.modules` lookup in `test_fknm_fallback.py`

### Background

`tests/test_fknm_fallback.py` (2026-07-05) has a `_ETS_module =
sys.modules["roboticstoolbox.robot.ETS"]` workaround, with a long comment
explaining why: `roboticstoolbox/robot/ETS.py` defines a class also called
`ETS`, `robot/__init__.py`'s `from ...ETS import ETS` shadows the
submodule of the same name on the `roboticstoolbox.robot` package, and
Python 3.10's `unittest.mock` resolves dotted-string `patch()` targets via
plain `getattr` (falling back to import only on `AttributeError`) — so it
gets fooled by the shadowing and raises `AttributeError`. Python 3.11+
rewrote this resolution to use `pkgutil.resolve_name`, which isn't fooled.
Not a real code bug (the actual fknm/facade fallback machinery was always
fine) — purely a Python-3.10 `unittest.mock` limitation the test had to
work around.

Python 3.10 reaches end-of-life in **October 2026** (per the official
CPython release schedule). `pyproject.toml`'s `requires-python = ">=3.10"`
and the module/class name collision in `ETS.py` aren't going anywhere on
their own, but this specific workaround exists *only* because of 3.10's
mock behavior.

### Proposed fix

When `requires-python` drops support for 3.10 (naturally, around/after its
EOL), search for this specific workaround and simplify
`test_fknm_fallback.py` back to plain `patch("roboticstoolbox.robot.ETS.ETS_fkine", ...)`
-style dotted strings, since 3.11+'s `pkgutil.resolve_name`-based resolution
handles the shadowing correctly on its own. Also worth a quick sweep for
any other `sys.version_info`/Python-3.10-specific conditionals elsewhere in
the codebase at that point, so 3.10 cleanup happens in one pass rather than
piecemeal.

---

## Repeated `:returns:`/`:param:` field markers in `robot/*.py` docstrings

### Background

While fixing the "Field list ends without a blank line" Sphinx build warnings
(2026-07-05 — a broken `Examples`/`.. runblock::` template pattern across
`BaseRobot.py`, `ETS.py`, `Link.py`, `ET.py`, `Dynamics.py`, likely from the
same refactor work already covered by the "ETS / fknm / frne refactor" entry
above), several docstrings turned up with **repeated field markers** that
don't actually trigger a Sphinx/docutils warning (repeated field names are
silently accepted, not flagged) but are still wrong:

- `BaseRobot.hasdynamics`/`hascollision`: two consecutive `:returns:` lines
  (only one is really shown depending on renderer; the two sentences should
  be merged into one field).
- `BaseRobot.ets`: `:param :param start: ...` / `:param :param end: ...` —
  a duplicated `:param` marker, plus a stray trailing colon after the
  description.
- `BaseRobot.todegrees`: **six** consecutive `:returns:` lines that are
  clearly meant to be one multi-line description.
- A `path`-returning method and a gripper/end-effector method with three
  consecutive `:returns:` lines each.
- `BaseRobot` (one of the `q`-random methods, ~line 1804):
  `:returns: Random joint configuration :rtype: ndarray(n)` — `:rtype:`
  crammed onto the same line as `:returns:` instead of being its own field.

Left alone deliberately (2026-07-05) since fixing the actual build warnings
was the scoped task — this is docstring *content* quality, not a build
break, and merging multi-sentence explanations correctly needs a human
read of intent rather than a mechanical script.

### Proposed fix

Sweep `robot/*.py` for `^\s*:returns:.*\n\s*:returns:` (and `:param:`)
regex hits and manually merge each into a single well-formed field.

---

## `napoleon` + `sphinx_autodoc_typehints`: bare NumPy-style section headers conflict

### Background

While fixing Sphinx build warnings (2026-07-05), `BaseRobot.dotfile`'s
docstring kept reporting "Explicit markup ends without a blank line" at a
line that had nothing wrong with it by inspection. Root-caused via an
isolated minimal Sphinx build (bisecting the extension list): the
combination of `sphinx.ext.napoleon` + `sphinx_autodoc_typehints` conflicts
when a docstring mixes bare NumPy-style section headers (`Note` underlined
with `----`, recognized and reformatted by Napoleon's heuristics) with
explicit reST fields (`:param:`, `:returns:`) and typehint injection from
`sphinx_autodoc_typehints`. Neither extension alone reproduces it.

Confirmed napoleon is still genuinely needed, not just historical baggage:
`tools/urdf/utils.py` and `tools/urdf/urdf.py` have real NumPy-style
`Parameters`/`Returns` docstrings (looks vendored/adapted from an external
URDF library) that Napoleon actually converts. Everywhere else in the
codebase (confirmed via grep) writes explicit reST fields directly and
doesn't need Napoleon's conversion — but two `robot/BaseRobot.py` docstrings
had a stray bare `Note` header mixed into otherwise-reST content, which is
what triggered the conflict. Also found `napoleon_custom_sections =
["Synopsis"]` in `conf.py` was dead config — nothing in the codebase uses a
"Synopsis" section — removed.

### Proposed fix

Fixed the two known `Note` occurrences (now `.. rubric:: Notes`, matching
the pattern already used elsewhere in the same file). If this warning
resurfaces elsewhere, the fix is the same: replace the bare NumPy-style
header with the equivalent explicit directive (`.. rubric:: Notes`,
`.. warning::`, etc.) rather than relying on Napoleon to recognize it —
don't touch `tools/urdf/*.py`, which needs Napoleon's real NumPy-style
parsing left alone.

---

## Codebase is mid-migration from NumPy-style to reST-style docstrings

### Background

The `Note` header bug above is one instance of a broader pattern: this
codebase has clearly moved from NumPy-style docstrings (bare underlined
section headers — `Parameters`/`Returns`/`Notes`/`See Also`, each followed
by a `----` underline) to explicit reST fields (`:param:`, `:returns:`,
`:seealso:`), but the migration is incomplete. Checked 2026-07-05: the
`:seealso:` field is used **202** times across the codebase — clearly the
intended, current convention — but the vestigial bare `See Also`
NumPy-style heading still appears **48** times across 6 files. Each one is
a latent instance of the same Napoleon-heuristic-recognition risk that hit
`BaseRobot.dotfile`/`random_q` (see above) — most haven't triggered a
visible warning yet only because they haven't hit the right combination of
circumstances, not because they're actually safe.

The same is likely true for other NumPy-style section names (`Notes`,
`Warning`, `Raises`, `Attributes`, `Examples`) wherever they appear as a
bare underlined heading instead of the reST equivalent — this file only
audited `Note`/`See Also` specifically because those are what broke the
build this session. `tools/urdf/utils.py` and `tools/urdf/urdf.py` are the
one confirmed exception: genuine vendored/adapted NumPy-style docstrings
that Napoleon is correctly converting — don't touch those.

### Proposed fix

A future pass should grep for each NumPy-recognized bare section header
(`^\s*(Notes?|Warnings?|Parameters|Returns|Raises|Yields|Attributes|Methods|References|See Also|Examples)\s*$`
followed by a matching-length `-+` underline) outside `tools/urdf/`, and
convert each to its reST equivalent (`:seealso:`, `.. rubric:: Notes`,
`.. warning::`, etc.) — same treatment as this session's `Note` fixes,
just systematically rather than one-off as each breaks the build.

---

## `roboticstoolbox.models.list()` shadows the builtin `list`

Found 2026-07-05, fixed 2026-07-20. `src/roboticstoolbox/models/list.py`
renamed to `catalog.py`, function renamed `list` → `catalog`. `models.list()`
is kept as a deprecated `FutureWarning`-emitting shim (public API, so not
removed outright) that forwards to `catalog()`; to be dropped in a future
major version. Docs (`arm_dh.rst`/`arm_erobot.rst`) and `tests/test_models.py`
updated to call `catalog()` directly.

---

## `_fknm_c` (nanobind) leaks `_ETObj`/`_ETSObj` when created from a background thread

### Background

Found 2026-07-06 while investigating a `KeyboardInterrupt` traceback from
`examples/plot_swift.py` (unrelated Swift fix), which ended in:

```
nanobind: leaked 32 instances!
 - leaked instance 0x... of type "roboticstoolbox._fknm_c._ETObj"
 - leaked instance 0x... of type "roboticstoolbox._fknm_c._ETSObj"
nanobind: leaked 2 types!
nanobind: leaked 17 functions!
nanobind: this is likely caused by a reference counting issue in the binding code.
See https://nanobind.readthedocs.io/en/latest/refleaks.html
```

This is nanobind's built-in reference-leak detector, printed at
interpreter shutdown when C++-bound objects from `_fknm_c` (`_ETObj`,
`_ETSObj` — the compiled ETS/element representations behind `fkine`,
`jacob0`, etc.) are still alive. Confirmed via direct testing:

- **Not new, and not specific to Ctrl+C or Swift.** The identical message
  appears on a completely normal, successful `docs && make html` build
  (Sphinx's autodoc/runblock machinery constructs many robot models while
  importing modules) — nothing to do with signal handling.
- **Not simply "many calls."** 2000 `fkine()`/`jacob0()` calls in a loop
  on the main thread: clean, no leak. The same calls in a loop on a
  background `threading.Thread` for as little as 0.5s: leaks every time.

Repro:

```python
import threading, time
import roboticstoolbox as rtb

p = rtb.models.Panda()
stop = False

def loop():
    while not stop:
        p.fkine(p.qr)
        p.jacob0(p.qr)

t = threading.Thread(target=loop, daemon=True)
t.start()
time.sleep(0.5)
stop = True
```

So the trigger is specifically **fknm object creation from a non-main
thread**, not call volume. This is directly relevant to `swift-sim`,
whose rendering/stepping machinery runs on background threads — any
script using the Swift backend is a candidate to hit this, independent
of whether it's interrupted.

### Proposed fix

Not root-caused yet — needs the actual `_fknm_c` nanobind binding code
(wherever `_ETObj`/`_ETSObj` are defined and bound) checked against
nanobind's refleak guide above. Leading hypothesis: a reference cycle
between the Python-side ETS wrapper and the C++ object that only cyclic
GC can break, where a non-main-thread-created object's cycle isn't
collected before process/thread teardown the way a main-thread one is.
Worth first confirming whether this is purely cosmetic (a shutdown-time
diagnostic with no real-world consequence for a long-running process) or
an actual growing-memory leak in a long-lived Swift session — the repro
above only demonstrates the message, not measured memory growth.

---

## `IK.py`'s solvers are typed against `ETS` but only need a small FK/Jacobian surface

### Background

`IKSolver._solve`/`step`/`_random_q`/`_check_jl`/`_null_Σ` (`src/roboticstoolbox/robot/IK.py`)
all take an `ets: "rtb.ETS"` parameter, but only ever call a small, fixed
set of things on it:

- `n` (int) — joint count
- `qlim` (2×n array) — joint limits
- `jindices` — joint index array
- `joints()` — iterate joint ETs (used once, to find max jindex)
- `eval(q)` — forward kinematics, raw matrix
- `jacob0(q)` — base-frame Jacobian
- `jacobm(q)` — manipulability Jacobian (only for the null-space `kq`/`km` terms)

Typing this as literally `ETS` overstates the coupling: nothing in `IK.py`
needs the rest of `ETS`'s surface (`merge`/`swap`/`split`/etc.), and the
solvers would work unchanged against anything else exposing this same
shape. (Note: this isn't a circular-import problem today — `IK.py`
already does `import roboticstoolbox as rtb` and only references
`rtb.ETS` inside string-quoted type hints, deferred and never evaluated
at runtime. This is about honestly documenting the dependency, not
working around an import cycle.)

`RobotProto.py` already establishes exactly this pattern for two other
mixins: `KinematicsProtocol` (needs just `_T` + `.ets()`, used by
`RobotKinematicsMixin`) and the larger `RobotProto` (used by the
`Dynamics` mixin). Both are `typing.Protocol` classes specifically so
mixins can be type-checked against a structural contract without
importing the concrete class.

### Proposed fix

Add a third protocol alongside those two — proposed name `IKProtocol`
(matches the existing `KinematicsProtocol`/`RobotProto` naming, and says
what it's for without the discarded, less-precise "IK-capable" framing)
— declaring exactly the seven-item surface above. Change every
`ets: "rtb.ETS"` in `IK.py` to `ets: "IKProtocol"`. `ETS` already
structurally satisfies it, so no change needed there; the payoff is that
the real dependency becomes explicit rather than implicit.

**Deferred** while the `/ets` package split (moving `ET`/`ETS` out of
`/robot`, see the ETS/fknm refactor entry above) is underway — worth
revisiting together since the import surface `IK.py` needs from the new
package location is the same shape this protocol would formalize.

## `URDF.UR5`'s `gripper_link_index=7` is a hardcoded position, not a stable identifier

### Background

Found 2026-07-17: CI failed on `URDF.UR5` construction (`ValueError:
incorrect vector length: expected 4, got (6,)` from `addconfiguration_attr`)
across every OS/Python combination, while local runs passed. Root cause:
`pyproject.toml`'s `robot_descriptions` dependency had no upper bound, and
`robot_descriptions` 3.0.0 (uploaded 2026-07-11) changed which upstream repo
"ur5" resolves to — `Universal_Robots_ROS2_Description` instead of whatever
2.0.0 pointed at. `UR5.__init__` calls `super().__init__("ur5", ...,
gripper_link_index=7)`, a raw positional index into the parsed link list
meant to mark the gripper/tool attachment link. The two versions parse to
different link counts (11 vs. 13) and orderings, so index `7` now lands on
`wrist_2_link` — a real arm joint — instead of the intended attachment
point, silently dropping 2 of 6 joints from `self.n`.

`gripper_link_index` was calibrated against `UR5.py`'s older local, vendored
xacro file (before it was switched to the bare name `"ur5"`, which triggers
live `robot_descriptions` resolution) — it was never designed to survive the
upstream source changing under it, which is now possible on every fresh
install.

**Immediate fix**: pinned `robot_descriptions>=2.0,<3.0` in `pyproject.toml`
(same floor+ceiling pattern already used for `rtb-data`), restoring the
known-working data. This does not fix the underlying fragility, just stops
it from firing today.

### Confirmed: a name/structure-based fix is viable, not a dead end

Compared the full raw link lists (`URDF_file("ur5")`, independent of the
gripper-link logic) between the two `robot_descriptions` versions:

- 2.0.0 (11 links): `world, base_link, shoulder_link, upper_arm_link,
  forearm_link, wrist_1_link, wrist_2_link, wrist_3_link, ee_link, base,
  tool0`
- 3.0.0 (13 links): `world, base_link, base_link_inertia, shoulder_link,
  upper_arm_link, forearm_link, wrist_1_link, wrist_2_link, wrist_3_link,
  ft_frame, base, flange, tool0`

All 6 real arm joints have identical names and `isjoint` status in both.
Only the gripper-attachment link's name differs (`ee_link` in 2.0.0), but
`tool0` exists in **both** — just at a different index (9 vs. 12). So a
name-based lookup (`"tool0"`) or, more robustly, a structural one ("the
first fixed link after the last actuated joint") would survive this exact
upstream change, unlike the index.

### Proposed fix

Replace `gripper_link_index: int` with a lookup that doesn't depend on
absolute position — either match by a small set of known tool-frame names
(`tool0`, `ee_link`, `flange`, ...) with a fallback, or derive it
structurally (first non-actuated link following the last actuated one in
the kinematic chain). Worth checking how many other `URDF.*` models pass a
raw `gripper_link_index` the same fragile way before deciding on the
general fix — this UR5 case is likely not unique.

**Deferred**: not fixed now, since the pin already resolves the immediate
CI break and this needs its own design pass rather than a rushed fix
bundled into a dependency-pin PR.

---

## `spatialgeometry` is vendored (pure-Python) inside RTB, but CI also installs the real external package from git

### Background

`src/spatialgeometry/` is a full copy of the `spatialgeometry` package
living inside this repo, bundled into RTB's own wheel
(`pyproject.toml`'s `[tool.scikit-build] wheel.packages` and
`sdist.include` both list `src/spatialgeometry` alongside
`src/roboticstoolbox`). Per git history this was deliberate:

- `1b522e65` — "bundle spatialgeometry as pure-Python (removes external
  dep)"
- `f99c154f` — "change collision backend from PyBullet to Coal"

i.e. the external `spatialgeometry` PyPI package was compiled (native
extensions) and became a NumPy-1-vs-2 compatibility hazard, so RTB
vendored its own pure-Python copy to stop depending on it directly.

However, `.github/workflows/ci.yml` still separately does, in every job
that needs `swift` (`test`, `coverage`, `docs-build`):

```
pip install git+https://github.com/petercorke/spatialgeometry.git@future
pip install git+https://github.com/petercorke/swift.git@future
```

This installs the *external* `spatialgeometry` under the same top-level
import name (`spatialgeometry`) that RTB's own wheel provides. Since
`pip install .[dev]` runs afterwards and rebuilds/installs RTB's own
wheel (which includes its bundled `src/spatialgeometry`), the files from
the external git install are almost certainly clobbered by RTB's vendored
copy — meaning the `@future`-branch install is likely a no-op in
practice, or at best relies on file-level overlap being total. This
hasn't been directly diagnosed/proven this session, just flagged as
suspicious: two packages exporting the same import name, installed in
sequence, is fragile regardless of which one currently "wins".

### Proposed fix

Decide on one ownership model and stop straddling both:

- **Option A**: keep vendoring `spatialgeometry` inside RTB (current
  state) and drop the redundant `pip install
  git+.../spatialgeometry.git@future` step from `ci.yml` — only
  `swift.git@future` is actually needed (assuming `swift` itself doesn't
  require anything from the external `spatialgeometry` that the vendored
  copy lacks; needs verifying).
  `swift.git@future`'s own dependency on `spatialgeometry` also needs
  checking — if `swift` unconditionally pulls in the real
  `spatialgeometry` from PyPI, that's the actual source of the NumPy-2
  crash for `spatialgeometry` again, RTB's vendoring or not.
- **Option B** (what the user favours): un-vendor `src/spatialgeometry`
  from RTB entirely, get `spatialgeometry` a proper NumPy-2-compatible
  PyPI release (companion to whatever eventually resolves the
  `swift`/`spatialgeometry` `@future`-branch hotfix noted elsewhere in
  this file under CI), and go back to a normal external dependency in
  `pyproject.toml`. Cleaner long-term (one source of truth for
  `spatialgeometry`, matches how `spatialmath-python` is already handled
  as a normal external dependency), but is real work: whoever maintains
  `petercorke/spatialgeometry` needs to actually cut and publish a
  release from its `future` branch first, or RTB is back to the same
  NumPy-1 compiled-wheel crash it vendored its way out of.

Either way, the current arrangement (vendored copy + redundant external
git install in CI) should not persist indefinitely without a decision.

## `src/roboticstoolbox/blocks/` has essentially no type hints

Noticed 2026-07-21 while fixing `blocks/arm.py`'s `gravity` parameter
(docstring claimed `float`, actual runtime value is always a 3-vector
passed straight through to `Robot.rne()`/`gravload()` -- the annotation
was simply wrong, and nothing caught it because there was no annotation
to check against).

`blocks/` has 5 files (`arm.py`, `mobile.py`, `quad_model.py`,
`spatial.py`, `uav.py`) defining `bdsim` block classes -- `arm.py` alone
has 16 `__init__` methods, none with parameter or return type hints.
Unlike the rest of the codebase (which follows the modern-syntax type
hint convention in this repo's global instructions), these blocks are
essentially undocumented at the type level, so mismatches like the
`gravity: float` one can and did sit unnoticed.

## `examples/ik_speed.py` doesn't run — stale imports, predates this session

Found 2026-07-23 while adding a portable CPU-info one-liner to
`examples/rne_speed.py` (a new RNE timing-comparison script) and looking
for a sibling to riff off. `ik_speed.py` fails immediately: `import fknm`
at module scope (line 4) — `fknm` hasn't existed as a top-level importable
module since the fknm/frne refactor moved it to
`roboticstoolbox.robot.fknm` (merged to `main` 2026-07-03, well before this
session). Not caused by, or related to, the current dynamics-overhaul
branch — pre-existing breakage that had gone unnoticed.

Also stale, found by inspection while there (none blocking, all part of the
same cleanup):

- Unused imports: `swift`, `spatialgeometry as sg`, `sys`, `from numpy
  import ndarray`, `from typing import Union, overload, List, Set`.
- No type hints (consistent with the `blocks/` entry above — this predates
  the modern-syntax convention too).

**Proposed fix:** update the `fknm` import to `from roboticstoolbox.robot
import fknm` (or import the specific facade functions actually used),
verify the script runs end-to-end again, then sweep the unused imports.
While there, port over the `cpu_info()` helper added to `rne_speed.py`
(portable one-line CPU description — name, core count, clock speed where
the OS exposes one) so `ik_speed.py`'s output reports what machine it ran
on too. Since that would make it the *second* real call site, worth
factoring `cpu_info()` into a small shared `examples/` helper at that point
rather than copy-pasting it again.

**Proposed fix:** add type hints throughout `blocks/`, following the same
convention as the rest of the codebase (`NDArray`/`ArrayLike` in
signatures, `X | None` not `Optional[X]`, etc. -- see this user's global
CLAUDE.md type-hint rules). Not urgent on its own, but worth doing
opportunistically whenever a block class is touched for another reason,
and worth a dedicated pass if `blocks/` sees more maintenance.
## JupyterLite "Try it Now" notebook: `ci.yml` stopgap for an upstream piplite indexing bug

### Background

Found and fixed (stopgap) 2026-07-21. The live JupyterLite deployment's
`docs/lite/pypi/all.json` package index only listed a `cp313` wasm wheel
for `roboticstoolbox-python`, even though the pinned
`jupyterlite-pyodide-kernel==0.6.1` (see the "Build JupyterLite site" step
in `ci.yml`) embeds Pyodide 0.27.6 / CPython 3.12 and needs a `cp312`
wheel. Confirmed on the deployed site: both the `cp312` and `cp313` wheel
*files* were present at `docs/lite/pypi/`, but only `cp313` made it into
`all.json`. Result: `piplite.install(["roboticstoolbox-python"])` in the
notebook's first cell fails with `ValueError: Can't find a pure Python 3
wheel for 'roboticstoolbox-python'` — micropip correctly rejects the
mismatched `cp313` entry, finds nothing else locally, falls through to
real PyPI, and PyPI has no wasm-tagged wheel at all (rejects the
`pyodide_*` platform tag).

Root cause is upstream, in `jupyterlite_pyodide_kernel/addons/piplite.py`'s
`get_wheel_index()`:

```python
all_json[normalized_name]["releases"][version] = [release]
```

This assigns a fresh single-item list per `(name, version)` key instead of
appending — so when a release carries wasm wheels for more than one
CPython version (as this repo's releases do: `cibuildwheel`'s pyodide
platform builds one per supported interpreter, e.g. `cp312` and `cp313`
both tagged version `1.3.1`), whichever wheel sorts alphabetically last
silently overwrites the others in the generated index. No build error —
the site builds and deploys fine, it just fails at runtime in the
browser. Confirmed this is not fixed in `jupyterlite_pyodide_kernel`
0.7.1 either (latest as of 2026-07-21), so it isn't something bumping the
pin resolves.

### Fix applied (stopgap)

Narrowed the `gh release download --pattern` in `ci.yml`'s "Fetch pyodide
wheel for JupyterLite" step from `*pyodide*` to `*cp312*pyodide*`, so only
the one wheel the pinned kernel actually needs ever reaches
`docs/lite/pypi/`, sidestepping the upstream overwrite bug entirely
rather than depending on it getting fixed.

### Proposed real fix

This whole fetch-a-prebuilt-wasm-wheel-from-a-GitHub-release mechanism,
and the CPython-tag pinning it depends on, is a maintenance burden in its
own right (see the "aspire to a pure-Python wheel" discussion elsewhere).
The real fix is a genuine `py3-none-any` pure-Python wheel published to
real PyPI (no compiled `_fknm_c`/`_frne_c`) — Pyodide's micropip resolves
that automatically via its own standard fallback logic, no custom
JupyterLite piplite index or GitHub-release-asset fetching required at
all. Once that lands, this stopgap (and the "Fetch pyodide wheel"/"Build
JupyterLite site" steps generally) should be revisited — possibly
removable outright. Tracked as a follow-on to the RNE correctness work
(`rne.md`), which is queued first.
