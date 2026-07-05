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

## `rtb-data` publishing is manual and has drifted from `main`

### Background

`rtb-data` is a separate PyPI package (data files: meshes, xacro/URDF sources,
etc.) built from the `rtb-data/` subdirectory of this repo, but published
independently — `roboticstoolbox-python`'s own release process
(`release-please` + `release.yml`) knows nothing about it. As of 2026-07-03,
`rtb-data/` on `main` had drifted well ahead of what's on PyPI (missing
`rtb-data/pyproject.toml` entirely at one point — see the "add missing
rtb-data package config and data files" fix), and several `test_models.py`
cases fail on CI as a direct result (xacro files the installed PyPI package
doesn't have yet).

### Proposed fix

Automatically publish a new `rtb-data` release alongside `roboticstoolbox-python`'s
own release, **but only if `rtb-data/` actually changed** since the last
`rtb-data` publish. The whole point of `rtb-data` being a separate package is
that it's large (meshes, STL/OBJ files) and should get pushed infrequently —
don't turn this into "publish rtb-data on every roboticstoolbox-python
release" regardless of whether anything in it moved. A GH Actions step
(likely in `release.yml` or a new workflow, triggered the same way) that:

1. Diffs `rtb-data/` against the tree at the last `rtb-data` publish (tag or
   recorded commit SHA)
2. If unchanged, no-op
3. If changed, bump `rtb-data/pyproject.toml`'s version and publish to PyPI

would keep the two packages in sync without manual "did I remember to publish
rtb-data" steps, while preserving the "infrequent, only-when-needed" intent.

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

## Move `rtb-data/` into a `packages/` folder

### Background

`rtb-data/` currently sits at repo root alongside the main `roboticstoolbox`
source tree, even though it's an independently-versioned, independently
-published PyPI package. Only two files reference the path today
(`pyproject.toml`'s `sdist.exclude`, and `rtb-data/pyproject.toml` itself),
so the move is cheap. Worth doing since `GRAPHICS-BACKEND.md` and
`SWIFT-MPL-SPLIT.md` both describe splitting graphics backends into further
sub-packages — better to establish the `packages/` convention while there's
one occupant than to reshuffle after two or three exist. Also sequences
well with the rtb-data multi-package release-please config above: build
that config against the new path from the start rather than writing it
against `rtb-data/` and moving it later. Queued behind the 1.3.1
maintenance release (2026-07-03) — deliberately not bundled with it, to
avoid adding another moving part to an already-unstable `main`.

### Proposed fix

`git mv rtb-data packages/rtb-data`, update `sdist.exclude`, land as its own
small PR to `main`, independent of any release-process work in flight.

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

## Flaky numerical IK test: `test_IK_GN3`

### Background

Seen failing on `macos-latest, Python 3.12` CI (2026-07-04) only:
`AssertionError: 1e-05 not greater than 0.05291452734038758` — a
Gauss-Newton IK convergence tolerance check. Didn't reproduce on the same
run's other platform/version combinations, so likely a genuine numerical
flake (seed-dependent convergence, or platform BLAS/LAPACK differences)
rather than a real regression. Not investigated further.

### Proposed fix

Watch for recurrence; if it keeps showing up, look at whether `test_IK_GN3`
seeds its initial joint configuration deterministically and whether the
tolerance is unreasonably tight for Gauss-Newton specifically (GN is known
to converge less reliably than LM from some seeds).

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
