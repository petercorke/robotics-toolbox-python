

# RNE (inverse dynamics) — issues and plan

Working notes from an investigation of the recursive Newton-Euler code paths,
prompted by `docs/notebooks/two-link-dynamics.ipynb` having to call
`rne_python()` explicitly instead of `rne()`. Captures what we found and the
plan for fixing it, before we start changing code.

See `examples/rne_compare.py` for the comparison/benchmark script referenced
throughout (Puma560, pose `qn`, `qd=qdd=[0.1]*6`, and a 1000-row trajectory
for timing).

## Fixed 2026-07-21: base-rotation handling (separate from issue 6 below)

Found while investigating `Robot.rne()` against `TwoLink` (base
`SE3.Rx(pi/2)`, standard DH): `Robot.rne()`'s Featherstone recursion never
referenced `self.base` at all -- gravity was applied along the chain's own
frame-0 z-axis regardless of base orientation, giving exactly zero gravity
torque for a robot (like `TwoLink`) whose own DH parameters are planar and
relies entirely on `self.base` to tilt into the plane gravity acts in.

Investigating further surfaced three more, related bugs, none of them the
DH/MDH convention issue (item 6) -- confirmed independent, since fixing
these did not change the standard-DH-vs-modified-DH divergence at all:

- `rne_python()`'s rotated-base branch was missing a negation:
  `vd = Rb @ gravity` where every other call site (including the same
  function's own identity-base branch) computes `-gravity`. Exact sign
  flip, confirmed against the Lagrangian-identity ground truth on TwoLink.
- `ne.c`/`frne_nb.cpp` never handled base rotation either -- worked around
  for 30 years by `DHRobot.rne()` pre-rotating (and negating) the gravity
  vector in Python before calling C. Moved into the C glue itself
  (`frne_nb.cpp`'s `frne()` now takes `self.base.R` directly, using the
  existing `rot_trans_vect_mult` C helper) -- `DHRobot.rne()` no longer
  needs to know about this at all.
- `rne_python()`'s `base_wrench=True` path allocated `wbase` with shape
  `(nk, n)` (joint count) instead of `(nk, 6)` (a wrench is always
  6-dimensional) -- worked by coincidence for 6-DOF Puma560, crashed for
  TwoLink (2 DOF).

Also added `base_wrench` support to the C path (`ne.c`'s backward
recursion already computes `f(0)`/`n(0)`, just never returned it), and
renamed `f_tip`/`n_tip` → `f_ee`/`n_ee` with docstrings/comments
standardized on "wrench applied to end-effector" throughout (`fext`
parameter name unchanged).

New regression tests in `tests/test_fknm_fallback.py`:
`TestRNERotatedBaseFallback`/`Reference` (TwoLink vs C, vs ground truth),
`TestBaseWrenchFallback`/`Reference` (C vs `rne_python()` base wrench,
with and without an end-effector wrench).

**Issue 6 below is now also resolved** — see its updated writeup.

## The three implementations

| | class | backing | symbolic-aware? |
|---|---|---|---|
| `rne()` | `DHRobot` | tries C extension (`frne.py` facade → `_frne_c`), falls back to `rne_python()` | no dispatch-time check (see Issue 1) |
| `rne_python()` | `DHRobot` | pure Python, direct array algebra, explicit `mdh`/non-`mdh` branches | yes, via `self.symbolic` (dtype="O", drops Coulomb friction) |
| `rne()` | `Robot` (base class of `DHRobot`) | pure Python, builds `SpatialVelocity`/`SpatialAcceleration`/`SpatialForce`/`SpatialInertia` objects from spatialmath | yes, via its own `symbolic` kwarg, no relation to `DHRobot`'s |

`DHRobot.rne()` overrides `Robot.rne()`, so in normal use nothing ever calls
`Robot.rne()` on a `DHRobot` — it's only reachable by an explicit unbound call
(`Robot.rne(puma, ...)`), which is what `rne_compare.py` does.

## Issues found

### 1. `DHRobot.rne()` has no symbolic pre-check, so it crashes instead of falling back

Current dispatch logic (`DHRobot.py`, `rne()`):
```python
if self._frne is None or self._frne_stale:
    self._copy_to_cpp()
if self._frne is None or base_wrench:
    return self.rne_python(...)
```
This only asks "is the C extension importable" or "was `base_wrench`
requested." It never checks whether `q`/`qd`/`qdd` or the model's own link
parameters are symbolic.

Confirmed failure: building the two-link notebook's model
(`DHRobot(..., symbolic=True)`) and calling `.rne(q, qd, qdd)` with symbolic
`q`/`qd`/`qdd` throws, uncaught:
```
TypeError: Cannot convert expression to float
```
This happens inside `_copy_to_cpp()`, which preallocates a **float64** array
(`L = np.zeros(24 * self.n)`) and assigns each link parameter into it —
`L[j+1] = self.links[i].a` throws when `a` is a sympy `Symbol`. Even past
that, `np.ascontiguousarray(q, dtype=float)` a few lines later would fail the
same way on symbolic `q`.

This is why the notebook has to call `.rne_python()` directly — there is
currently no way to just call `.rne()` and get correct automatic fallback.

### 2. Two disconnected symbolic-detection mechanisms already exist, neither used by `rne()`

- `BaseRobot.symbolic` / `self._symbolic`: a **construction-time flag**, set
  manually (e.g. `Puma560(symbolic=True)`, which builds the DH table itself
  out of `sympy` constants instead of floats). Read only inside
  `rne_python()`, to pick `dtype="O"` and to disable Coulomb friction
  (`sign()` doesn't work symbolically).
- `fknm.py`'s `_is_symbolic(q)`: a **runtime dtype check** on `q`
  (`np.asarray(q).dtype == object`), used throughout the ETS
  forward-kinematics/Jacobian facade to decide C vs. Python *before* ever
  calling into C, e.g. `ETS_fkine`: `if fknm is not None and not
  _is_symbolic(q): return _c_ETS_fkine(...)`.

`frne.py`/`DHRobot.rne()` has neither. The right pattern already exists next
door in `fknm.py` — `rne()` just doesn't use it.

### 3. Forgetting the `symbolic=True` flag breaks even the "safe" Python path

Confirmed: building a `DHRobot` with symbolic link parameters but *without*
`symbolic=True` leaves `robot.symbolic == False`, so `rne_python()` itself
allocates float64 arrays and crashes with the same
`Cannot convert expression to float` — even though `rne_python()` is supposed
to be the always-works fallback.

**Conclusion (per discussion): detect symbolic-ness of the model at build
time**, by inspecting the link parameters (`a`, `alpha`, `theta`, `d`, `m`,
`r`, `I`, ...) for non-numeric content, rather than trusting a user-supplied
constructor flag. `BaseRobot.__init__` already loops over every link once
(for geometry flags) — the same pass could set this. Keep `symbolic=` as an
optional override, not the source of truth.

### 4. Checking symbolic-ness of `q`, `qd`, `qdd` (three vectors, not one)

Not a mechanical problem — three `dtype == object` checks is negligible
cost. The real decision is policy when they disagree (e.g. `q` numeric but
`qd` symbolic). Recommendation: fall back to the Python path if **any** of
`q`/`qd`/`qdd` is non-numeric, matching the existing `fknm.py` idiom of
`not _is_symbolic(a) and not _is_symbolic(b)` (De Morgan: either symbolic ⇒
fallback). This check must also fold in the model's own symbolic-ness (issue
3) — a numeric `q`/`qd`/`qdd` against symbolic link masses still needs the
object-dtype path.

### 5. Trajectories are not batched into the C extension — confirmed

Instrumented `frne()` directly: a 1000-row trajectory triggers **1000
separate Python→C calls**, one per row
(`for i in range(trajn): tau[i, :] = frne(...)` in `DHRobot.rne()`). So today
there is no benefit to pre-stacking a trajectory into one array call vs.
calling `.rne()` 1000 times yourself — same Python-side loop either way.

Measured cost of this: 1000 bare `frne()` calls take 1.28 ms; `puma.rne()` on
the same stacked trajectory takes 1.95 ms — **~35% of wall time is Python-side
looping/marshalling overhead**, not the C computation. Passing the whole
trajectory into a single C call (looping in C) would reclaim most of that.

Priority: real but modest — already 223x faster than pure Python at n=6 DOF,
so this is a constant-factor win, not a complexity-class change. Matters more
for tight real-time control loops or larger N/DOF.

### 6. Fixed 2026-07-21: `Robot.rne()` and `rne_python()` each got ONE DH sub-convention right, and the other wrong — root-caused and resolved

Original finding (kept for the record): `Robot.rne()` doesn't model armature
inertia (`G`, `Jm`) or friction (`B`, `Tc`) at all — that explains *some*
divergence from `rne_python()` on Puma560. But zeroing those out on a copy of
Puma560 to isolate rigid-body-only dynamics, the two **still disagreed by
~70** (vs. a torque magnitude of ~30-45), including in a **pure gravity-load
case** (`qd = qdd = 0`), which has no velocity/Coriolis terms to get wrong at
all. So it wasn't just missing terms — confirming the hypothesis that **the
link frames differ, and the inertial parameters (`r`, `I`) are expressed
relative to those differing frames.**

Root cause, isolated with `examples/rne_dh_convention_check.py` (a single
revolute link, `alpha=1.2, a=d=0, r=[0.5,0,0]`, static so there are no
velocity terms at all, `q0=0.3`) and checked against an independent ground
truth — the Lagrangian identity `tau = d/dq[m g z_com(q)]`, computed directly
from `link.A(q)` and numerically differentiated, which doesn't rely on either
RNE implementation's spatial-vector bookkeeping:

```
                              truth      rne_python   Robot.rne
Standard DH (mdh=False)      0.0000     0.0000       4.5717   <- Robot.rne WRONG
Modified DH (mdh=True)       4.3675     0.0000       4.3675   <- rne_python WRONG
```

**Neither implementation was "the trusted reference" — each was only correct
for one DH sub-convention, for two structurally different reasons:**

- **`rne_python()`'s modified-DH branch had three real bugs**, found by
  term-by-term comparison against `ne.c`'s `MODIFIED` branch while exercising
  an `mdh=True` + rotated-base case that had apparently never been exercised
  before (`TestTwoLinkDHMDHEquivalence` in `tests/test_fknm_fallback.py`):
  1. Base rotation applied to gravity twice (once correctly before the
     recursion loop, once more via a redundant `Tj = base @ Tj` for the first
     link) — double-counted.
  2. Missing parentheses in the MDH revolute case's linear acceleration
     formula — `Rt @ cross(wd, pstar) + cross(w, cross(w, pstar)) + vd`
     should distribute `Rt` over the whole sum, matching `ne.c`'s
     `rot_trans_vect_mult` applied to the full bracket.
  3. The backward recursion's moment equation used `pstar` (the *next*
     link's offset) where it should use `r` (this link's own CoM offset) —
     `ne.c`'s equivalent is `R_COG(j) x F`, not `PSTAR(j+1) x F`.

  With all three fixed, `rne_python()` is correct for both DH conventions
  (`test_mdh_rne_python_now_agrees`).

- **`Robot.rne()`'s generic Featherstone recursion genuinely requires the
  joint to be the last element of its own ETS segment** (fixed geometry gets
  you *to* the joint; the joint is the last thing applied before the next
  link's frame) — this is inherent to the algorithm, not a bug to fix. This
  holds for `Robot`/`ERobot`/`URDFRobot` (`ETS.split()`), `PoERobot`
  (`_update_ets()` appends the joint last too), and — checked carefully,
  since it wasn't obvious — for `DHRobot(mdh=True)` too:
  `DHLink._to_ets()`'s MDH revolute branch reorders a nonzero `d` translation
  to *precede* the joint rotation specifically so the joint ET stays last;
  valid because a z-rotation and a z-translation about/along the same axis
  commute, so the reorder doesn't change the net transform. It does **not**
  hold for `DHRobot(mdh=False)` (standard DH), where the joint comes *before*
  the link's fixed `tz(d) * tx(a) * Rx(alpha)` transforms — structurally
  incompatible with the algorithm, not fixable without changing what
  `Robot.rne()` fundamentally assumes.

  **Action taken:** rather than teach `Robot.rne()` a second, DH-aware
  recursion, it now asserts on the incompatible case
  (`assert getattr(self, "mdh", True)` in `Robot.rne()`) instead of silently
  returning a wrong answer — see tech-debt.md for the blocklist-vs-allowlist
  design discussion behind this specific check, and
  `test_robot_rne_rejects_standard_dh` for the regression test.

- **A separate, previously-masked bug**, found while numerically verifying
  the `mdh=True` case above with a nonzero `d`/`alpha`/inertia tensor (rather
  than just trusting the structural argument): `Robot.rne()`'s inertia
  accumulation, `SpatialInertia(m=link.m, r=link.r)`, never passed
  `I=link.I` — the rotational inertia tensor was silently dropped for
  *every* link, on every call, regardless of DH convention. This affected
  `Robot`/`ERobot`/`URDFRobot`/`PoERobot`/`DHRobot(mdh=True)` alike, not just
  the MDH edge case that surfaced it — never caught earlier because the
  `TwoLink`-based equivalence tests used default (zero) inertia, and the
  tests that *did* set `inertia=True` only compared the C path against
  `rne_python()`, never `Robot.rne()`. Fixed; see
  `TestRobotRneInertiaTensor` in `tests/test_fknm_fallback.py`.

`Robot.rne()` also already passes `test_ERobot.py::test_invdyn`, which
validates it against a hand-built 2-link ETS robot (elementary `ET.Ry()`
transforms) checked against a known analytical result (Spong et al., 2nd
ed., p. 260) — consistent with it being correct for genuine
elementary-transform ETS chains.

Since essentially every built-in DH model in this codebase (Puma560, etc.)
uses **standard** DH, this is exactly why the original Puma560 comparison
showed `Robot.rne()` diverging and `rne_python()`/C agreeing — consistent
with, not contradicting, this finding.

**Net result:** `rne_python()`/`rne()` (C) are correct for both DH
conventions. `Robot.rne()` is correct for MDH `DHRobot`s and all
ETS-native robots (`Robot`/`ERobot`/`URDFRobot`/`PoERobot`), now including
the previously-dropped inertia tensor, and cleanly rejects the one
structurally-incompatible case (standard DH) instead of silently
miscomputing it. No unification was attempted — the two remain independent
implementations by design (see the architecture note at the top of
`Dynamics.py`), which is the right call given `Robot.rne()`'s algorithm
cannot represent standard DH's joint-first structure at all.

### 7. Fixed 2026-07-21: Housekeeping

- `rne_python()`'s docstring called itself `rne_dh` throughout — stale name
  from before a rename. Fixed.
- A dead commented-out symbolic-simplification block sat at the end of
  `rne_python()`. Removed.

## Plan (systematic, building on `examples/rne_compare.py`)

1. ~~**Root-cause the `Robot.rne()` / DH-link mismatch first**~~ **Done** —
   see issue 6. Two independent findings: `Robot.rne()`'s Featherstone
   recursion structurally requires joint-last ETS segments (true for MDH,
   false for standard DH — not fixable, now guarded); and a separate,
   previously-masked bug where it dropped the rotational inertia tensor
   entirely (`SpatialInertia` never got `I=link.I`) — fixed.
2. ~~**Formalize `rne_compare.py` into pytest regression tests**~~ **Done** —
   `tests/test_fknm_fallback.py`: `TestTwoLinkDHMDHEquivalence`,
   `TestRobotRneInertiaTensor`, `TestRNERotatedBase*`, `TestBaseWrench*`, and
   `TestTwoLinkAbsoluteGroundTruth` (an independent closed-form analytical
   check — Spong et al. Eq 7.87 — not derived from or shared with any RNE
   implementation here, added 2026-07-21 since everything else only checks
   these implementations against each other).
3. ~~**Decide unification vs. documented separation**~~ **Done** — kept
   separate (`Robot.rne()` and `DHRobot.rne_python()`/`rne()` remain
   independent implementations); documented in the `Dynamics.py` module
   docstring and here. The right call given `Robot.rne()`'s algorithm cannot
   represent standard DH's joint-first structure at all — there is nothing
   to unify it with for that case.
4. ~~**Auto-detect symbolic models at build time**~~ **Done** — `BaseRobot.py`:
   `_is_symbolic()` helper + `_SYMBOLIC_LINK_ATTRS` scan in `__init__`'s
   existing per-link loop, ORs into `self._symbolic` alongside the explicit
   constructor flag. Verified against issue 3's exact repro (a `DHRobot`
   built with a symbolic `a` but no `symbolic=True`) — now auto-detected.
5. ~~**Add a symbolic-aware dispatch check to `DHRobot.rne()`**~~ **Done** —
   routes to `rne_python()` if `self.symbolic` or any of `q`/`qd`/`qdd` is
   symbolic, mirroring `fknm._is_symbolic`'s idiom. **Also found and fixed a
   bug this step exposed**: `rne_python()` itself only checked
   `self.symbolic` (model-level) to pick its internal array `dtype`, not
   whether *this call's* `q`/`qd`/`qdd` were symbolic — so a numeric model
   called with symbolic `q` crashed inside `rne_python()`, defeating the
   point of routing there as the always-works fallback. Fixed via a local
   `symbolic_call` variable, used for both the `dtype` choice and the
   Coulomb-friction guard.
6. ~~**Add a `try/except` safety net**~~ **Done** — wraps the C call, falls
   back to `rne_python()` on `(TypeError, ValueError)`.
7. ~~**Batch trajectories into a single C call**~~ **Done** —
   `frne_nb.cpp`'s `frne()` binding now takes `(trajn, n)` q/qd/qdd arrays
   and loops over the whole trajectory inside C++ (gravity/base-rotation
   setup done once, not per row); `DHRobot.rne()` makes a single call
   instead of looping in Python. Confirmed via `examples/rne_compare.py`'s
   per-row C-call counter: 1 call for a 1000-row trajectory (was 1000).
   Measured speedup (`examples/rne_speed.py`, `rtb.models.DH.Panda()`,
   1000 distinct random poses): C ≈ 0.6 ms total (≈0.6 us/row) vs
   `rne_python()` ≈ 289 ms (≈474x) and `Robot.rne()` ≈ 446 ms (≈731x).
   Regression coverage: `tests/test_fknm_fallback.py`'s
   `TestRneTrajectoryVaryingRows` — deliberately distinct (not tiled) rows
   per trajectory, since a uniform trajectory can't distinguish "row i is
   computed correctly" from "row i is silently some other row's result",
   which matters specifically because this step introduced new row-indexing
   arithmetic in the C++ loop.
8. ~~**Housekeeping**~~ **Done** — fixed the `rne_dh` docstring naming,
   deleted the dead commented-out block.

**All plan steps (1-8) are now done.**
