# Examples

Standalone scripts that demonstrate the toolbox. They are not run by the test
suite; run one directly from the repository root, for example:

```shell
python examples/puma_jtraj.py
```

## Requirements

All scripts need the base install, `pip install roboticstoolbox-python`.
Where a script needs an optional extra it is listed in the table:

| Extra | Install | Provides |
|---|---|---|
| `swift` | `pip install roboticstoolbox-python[swift]` | the browser-based Swift visualiser |
| `qp` | `pip install roboticstoolbox-python[qp]` | `qpsolvers` and `quadprog` for the QP-based controllers |
| `collision` | `pip install roboticstoolbox-python[collision]` | `coal` and `trimesh` for collision checking. `coal` has no Windows build, so use Linux, macOS or WSL |

`pip install roboticstoolbox-python[all]` installs every extra at once.

## Getting started and plotting

| Script | What it shows | Extras |
|---|---|---|
| `readme.py` | The code from the top-level README: DH Panda, IK, a joint trajectory, Swift animation | swift |
| `plot.py` | DH Panda animated through random joint angles with velocity and force ellipsoids, matplotlib backend | |
| `plot_swift.py` | Minimal: load the URDF Panda and plot it | swift |
| `puma_swift.py` | URDF Puma 560 interpolating between its named configurations in Swift | swift |
| `robots.py` | Loads a dozen URDF models side by side in Swift | swift |
| `teach.py` | Interactive teach pendant on the DH Panda, matplotlib backend | |
| `teach_swift.py` | Hand-rolled Swift teach panel; a template for custom slider UIs | swift |
| `mexican-wave.py` | Fifteen cloned Pumas doing a wave in Swift | swift |

## Trajectories and dynamics

| Script | What it shows | Extras |
|---|---|---|
| `puma_jtraj.py` | Joint-space trajectory (`jtraj`) between Puma configurations; `--backend` and `--model` options | |
| `puma_fdyn.py` | Forward dynamics (`fdyn`) of the Puma under zero torque, plotted and animated | |

## Resolved-rate and QP control

| Script | What it shows | Extras |
|---|---|---|
| `RRMC.py` | Resolved-rate motion control on the DH Panda, matplotlib backend | |
| `RRMC_swift.py` | The same controller on the URDF Panda in Swift | swift |
| `park.py` | Null-space manipulability maximisation (Park 1999) | swift |
| `baur.py` | As `park.py` with an added joint-limit avoidance term (Baur 2012) | swift |
| `mmc.py` | Manipulability-maximising QP controller (Haviland and Corke 2020) | swift, qp |
| `swift_recording.py` | `mmc.py` with Swift video recording enabled | swift, qp |
| `neo.py` | NEO reactive obstacle avoidance using `link_collision_damper` | swift, qp, collision |

## Mobile manipulation

| Script | What it shows | Extras |
|---|---|---|
| `holistic_mm_omni.py` | Holistic mobile manipulation with an omnidirectional base (FrankieOmni) | swift, qp |
| `holistic_mm_non_holonomic.py` | The same with a non-holonomic base. Currently broken: `rtb.models.Frankie()` no longer includes a mobile base | swift, qp |

## Branched robots

| Script | What it shows | Extras |
|---|---|---|
| `branched_robot.py` | Two-arm YuMi controlled with one ETS per gripper | swift |

## Paper walkthroughs

| Script | What it shows | Extras |
|---|---|---|
| `icra2021.py` | The code listings from the ICRA 2021 paper "Not your grandmother's toolbox" | swift, collision |

## Benchmarks and checks

| Script | What it shows | Extras |
|---|---|---|
| `benchmark_ik.py` | Wall time per problem for the IK solvers, C++ versus pure Python. See the wiki page Benchmark-IK | |
| `benchmark_rne.py` | Correctness cross-check and timing of the `rne` implementations. See the wiki page Benchmark-RNE | |
| `_cpu_info.py` | Helper used by the two benchmark scripts; not runnable on its own | |
| `ik_exp.py` | Success rate and iteration counts of the IK solvers over random targets | |
| `ikine_evaluate.py` | Timing of analytic and numerical IK on the DH Puma and Panda | |
| `rne_compare.py` | Puma 560: C versus Python `rne`, and the DH-convention guard in `Robot.rne` | |
| `rne_dh_convention_check.py` | One-link DH and MDH gravity torque against a Lagrangian ground truth | |

## Mobile robots

| Script | What it shows | Extras |
|---|---|---|
| `mobile.py` | Snippets from RVC chapter 5; mostly commented out | |
| `vehicle1.py` | A `VehicleIcon` animation; mostly commented out | |
