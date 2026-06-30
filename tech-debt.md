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
