# Changelog

## [1.3.1](https://github.com/petercorke/robotics-toolbox-python/compare/v1.3.0...v1.3.1) (2026-07-03)

Manual out-of-band release from the `v1.3.0` tag (not via release-please —
`main` has since diverged with in-progress, breaking work). See
`tech-debt.md` ("Release process: single-branch release-please + stacked
PRs on a red `main`") for why.

### Bug Fixes

* **deps:** pin `rtb-data<2` to protect this release line from an upcoming
  breaking `rtb-data` reorganisation (renamed/deleted model folders) that
  will ship as `rtb-data` 2.0 alongside the next `roboticstoolbox-python`
  major/minor release. Without this pin, any fresh `pip install
  roboticstoolbox-python` would silently pick up the incompatible new
  `rtb-data` and fail to load bundled robot models.

## [1.3.0](https://github.com/petercorke/robotics-toolbox-python/compare/v1.2.0...v1.3.0) (2026-06-14)


### Features

* bundle spatialgeometry as pure-Python (removes external dep) ([1b522e6](https://github.com/petercorke/robotics-toolbox-python/commit/1b522e65f1e50ca37d95d036cad7f405a7c71b5d))


### Build System

* setup proper release process ([d0e0559](https://github.com/petercorke/robotics-toolbox-python/commit/d0e0559a4c59e15d7eb87d9d34cedb11a9eb3fb6))
