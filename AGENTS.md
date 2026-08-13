# robotics-toolbox-python — Agent Instructions

Part of the RVC ecosystem. **Read [rvc-ecosystem/AGENTS.md](https://github.com/petercorke/rvc-ecosystem/blob/main/AGENTS.md) first** — it defines shared conventions: repo ownership, math invariants, dependency boundaries, git/PR workflow, code standards, tech-debt tracking. This file only adds what's specific to this repo.

| | |
|---|---|
| PyPI package | `roboticstoolbox-python` |
| Nickname | RTB |
| Owner | Peter Corke (`petercorke`) |
| Default branch | `main` |
| Contribution model | Branch → PR; direct push to `main` at Peter's discretion |

## Notes specific to this repo

- Compiles `nanobind` C++ extensions via `scikit-build-core`, not Hatch — this is the
  permanent, justified exception noted in the ecosystem `AGENTS.md` build-tooling section.
- Depends on `spatialgeometry` directly (required); `swift` is an optional extra
  (`pip install roboticstoolbox-python[swift]`), not a stepping-stone to reach `spatialgeometry`.
- Uses `spatialmath` classes (`SE3`, `SO3`, `UnitQuaternion`, etc.) for all pose math — do not
  re-implement transform mathematics here.
- Tech-debt tracked as GitHub Issues labelled `tech-debt` (migrated 2026-08-09). The backlog
  was large enough that closely-related entries were clustered into parent issues with
  checklists rather than a flat 1:1 mapping — follow that pattern if filing a cluster of
  related findings rather than one issue each.
