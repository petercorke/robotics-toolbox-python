# Changelog

## Highlights

**Correctness fix in inverse dynamics (RNE).** `Robot.rne()` — used by every
URDF/ETS-based robot's inverse dynamics — was silently returning wrong
torques for any robot with translational offsets between links (i.e.
virtually all real robots), traced to an upstream `spatialmath-python`
change to how spatial forces transform. Fixed, along with two related bugs
in how static (non-actuated) link mass gets attributed to the correct
joint. Verified against the compiled C++ reference path and textbook
closed-form solutions.

**Tool transforms now work everywhere.** `robot.tool`, once set, is
automatically honored by `fkine`/`jacob0`/`jacobe`/`hessian0`/`hessiane`
(previously silently ignored unless passed explicitly each call), and all
seven IK solvers (`ik_LM`/`ik_NR`/`ik_GN`/`ikine_LM`/`ikine_NR`/`ikine_GN`/
`ikine_QP`) now accept a `tool=` argument directly.

**New:** `Panda(use_robot_descriptions=True)` — an opt-in alternate Panda
model with real inertial data, for users who need correct dynamics rather
than the default's faster collision geometry.

**Bug fixes:** `joint_velocity_damper` keyword-argument misuse across
several example scripts; `DHRobot.ikine_6s` undoing the tool transform on
the wrong side; UR5's gripper link now resolved by name instead of a
fragile raw index.

**Wiki refresh (not reflected in this changelog — wiki edits are direct
pushes to a separate repo, not commits here):** a full currency pass across
all 26 pages, stale/incorrect content fixed, ROS references removed, and
new pages added — Panda model comparison, reactive control, inverse
kinematics, and IK/RNE benchmark results with real numbers.

**Changed behavior: `EKF.plot_error()`.** The confidence bounds are now the
interval containing the requested fraction of a one-dimensional Gaussian
(±1.96σ for `confidence=0.95`); earlier versions drew ±2.45σ, which is the
95% radius of a two-dimensional region and was too wide for a scalar error.
The error is now plotted against time (it was plotted against sample number on
axes limited to time), `bgcolor` is honored, and SLAM no longer raises.
Figures regenerated with this version, for example Figure 6.10 in *Robotics,
Vision & Control* (3rd ed.), have narrower shaded bounds than the printed ones.


## [1.5.0](https://github.com/petercorke/robotics-toolbox-python/compare/v1.4.4...v1.5.0) (2026-10-23)


### Features

* **dynamics:** payload_capacity() replaces pay() and paycap() ([#881](https://github.com/petercorke/robotics-toolbox-python/issues/881)) ([f7d5548](https://github.com/petercorke/robotics-toolbox-python/commit/f7d55488dc39cd9656cdb17600632a6dfc25e1f4))
* **ets:** a joint has an offset, and merge() can merge a constant into a joint ([#869](https://github.com/petercorke/robotics-toolbox-python/issues/869)) ([4c91c01](https://github.com/petercorke/robotics-toolbox-python/commit/4c91c0189a8f1e8f3099fd71c3bcd0a024550dcf))
* **ets:** a trailing underscore numbers a joint name from one; document joint descriptors ([#811](https://github.com/petercorke/robotics-toolbox-python/issues/811)) ([68f525a](https://github.com/petercorke/robotics-toolbox-python/commit/68f525ab0eb597f7e962413192a174536f2a1a16))
* **ets:** add numerical inverse kinematics for planar ETS2 ([#826](https://github.com/petercorke/robotics-toolbox-python/issues/826)) ([aaf0902](https://github.com/petercorke/robotics-toolbox-python/commit/aaf0902f7f8b3de67ecb3069b272e751482ef3ba)), closes [#824](https://github.com/petercorke/robotics-toolbox-python/issues/824)
* **ets:** fkine and eval take no argument for a sequence without joints ([#830](https://github.com/petercorke/robotics-toolbox-python/issues/830)) ([155a4be](https://github.com/petercorke/robotics-toolbox-python/commit/155a4be2df4fc7ec6a9f66b9d9a60058758bc464))
* **ets:** merged() and swapped() return a modified copy, and merge() removes a transform that adds up to nothing ([#868](https://github.com/petercorke/robotics-toolbox-python/issues/868)) ([ee91ff9](https://github.com/petercorke/robotics-toolbox-python/commit/ee91ff99a007ff37b9eaaf09745e6b0474cb1272))
* **ets:** replace() and replaced() put a transform on another axis ([#874](https://github.com/petercorke/robotics-toolbox-python/issues/874)) ([67ea1a1](https://github.com/petercorke/robotics-toolbox-python/commit/67ea1a133a0b0557f56ba5eaabbbada96586e014))
* **ets:** support locking a joint in an ETS ([#740](https://github.com/petercorke/robotics-toolbox-python/issues/740)) ([38fc591](https://github.com/petercorke/robotics-toolbox-python/commit/38fc5910d706af8d9233bfe5db05f8c8d1b7ef75))
* **ets:** swap() carries a transform across a constant quarter or half turn ([#870](https://github.com/petercorke/robotics-toolbox-python/issues/870)) ([84bd1e5](https://github.com/petercorke/robotics-toolbox-python/commit/84bd1e57627d6ddce0741a4f55f6aca516ff6a6b))
* **ik:** analytic inverse kinematics for six-axis arms, with generated code in Python, C, C++ and Rust ([#842](https://github.com/petercorke/robotics-toolbox-python/issues/842)) ([580886e](https://github.com/petercorke/robotics-toolbox-python/commit/580886ec351412d1c6c0cb9be511a863b551c6df))
* **ik:** verbose=True prints the progress of the Python IK solvers, and why a solution is rejected ([#834](https://github.com/petercorke/robotics-toolbox-python/issues/834)) ([c225baf](https://github.com/petercorke/robotics-toolbox-python/commit/c225baf881612d764e981f02a3cccf1c24db0d2d))
* **mobile:** animate range-bearing sensor beam and sensing region ([#858](https://github.com/petercorke/robotics-toolbox-python/issues/858)) ([9402635](https://github.com/petercorke/robotics-toolbox-python/commit/94026359b02a27e997a5f02e943655b8fd31c7ec))
* **mobile:** planners ignore unknown keywords with a warning; plot(labels=True); distance= for metric ([#786](https://github.com/petercorke/robotics-toolbox-python/issues/786)) ([1a3a19f](https://github.com/petercorke/robotics-toolbox-python/commit/1a3a19f2a5206ef6bb16b735a709331da7d8dc76))
* **plot:** plot(env=) adds a robot to an earlier scene in a pose of its own, plot(ax=) draws on given axes, and color= sets the color ([#873](https://github.com/petercorke/robotics-toolbox-python/issues/873)) ([1daa4b3](https://github.com/petercorke/robotics-toolbox-python/commit/1daa4b3317ba18714fe4eaff69ab3c55c67b2b3a))
* **plot:** plot(notebook=) chooses the Swift frame in a cell or a browser window, and browser=notebook is deprecated ([#866](https://github.com/petercorke/robotics-toolbox-python/issues/866)) ([f68741b](https://github.com/petercorke/robotics-toolbox-python/commit/f68741b6d73e2283e89631165c6c01521ec10fa7))
* **plot:** view="xy", "xz" or "yz" starts a PyPlot plot in an axis-aligned view of a plane ([#879](https://github.com/petercorke/robotics-toolbox-python/issues/879)) ([649b8fb](https://github.com/petercorke/robotics-toolbox-python/commit/649b8fb1d6b513c416237408ae1425d382cb47e7))
* **robot:** add self-collision checks for non-adjacent links ([#816](https://github.com/petercorke/robotics-toolbox-python/issues/816)) ([8255224](https://github.com/petercorke/robotics-toolbox-python/commit/82552245c32d2f4e22988406999c0642e2c4567d))
* **typing:** ship a py.typed marker so type checkers use RTB's annotations ([#823](https://github.com/petercorke/robotics-toolbox-python/issues/823)) ([ea8b978](https://github.com/petercorke/robotics-toolbox-python/commit/ea8b978a38337216e65dfdf202947349ac949089))


### Bug Fixes

* add the payload to the end-effector link instead of overwriting it ([#699](https://github.com/petercorke/robotics-toolbox-python/issues/699)) ([79b169c](https://github.com/petercorke/robotics-toolbox-python/commit/79b169cd66f224a232165bd96755ed6e99626ecd))
* **backends:** PyPlot backend fixes, Connector as a Protocol, one window-title helper ([#788](https://github.com/petercorke/robotics-toolbox-python/issues/788)) ([dcb7982](https://github.com/petercorke/robotics-toolbox-python/commit/dcb79824e825510364301c8ef23dc74f26638f14))
* **blocks:** DiffSteer NameError and VehiclePlot shared defaults, with types and docstrings ([#775](https://github.com/petercorke/robotics-toolbox-python/issues/775)) ([f59eee8](https://github.com/petercorke/robotics-toolbox-python/commit/f59eee81bfbd18eabacf165c2d228060178c6dc4))
* **blocks:** JTraj ends at rest, and accepts qd0 and qdf ([#890](https://github.com/petercorke/robotics-toolbox-python/issues/890)) ([44ebffe](https://github.com/petercorke/robotics-toolbox-python/commit/44ebffe3292d9f6b4f504b5ebbe38a20f189e1eb))
* **dh:** DHRobot.jacobe applies the flip of a joint ([#806](https://github.com/petercorke/robotics-toolbox-python/issues/806)) ([d312bc8](https://github.com/petercorke/robotics-toolbox-python/commit/d312bc8320e1f55c90e6fb77de7abebbc0d55af6))
* **dhrobot:** format prismatic joint limits in string output ([#718](https://github.com/petercorke/robotics-toolbox-python/issues/718)) ([71d07a7](https://github.com/petercorke/robotics-toolbox-python/commit/71d07a7aff0e22a13e3ff315953c97ee39482358))
* **docs:** drop the obsolete cross-repo spatialgeometry wheel fetch ([#703](https://github.com/petercorke/robotics-toolbox-python/issues/703)) ([477b6e3](https://github.com/petercorke/robotics-toolbox-python/commit/477b6e360c8263be49b771190564335fc2607ac9))
* **dynamics:** apply prismatic offsets in Python RNE ([#800](https://github.com/petercorke/robotics-toolbox-python/issues/800)) ([544e574](https://github.com/petercorke/robotics-toolbox-python/commit/544e57471223315e7969555ef7a36d5da2ebaa5c))
* **dynamics:** dynamics methods require dynamic parameters, and hasdynamics is kept by the setters ([#846](https://github.com/petercorke/robotics-toolbox-python/issues/846)) ([b7f9096](https://github.com/petercorke/robotics-toolbox-python/commit/b7f909652e256d0c9fb0dd77e137bd57f7d40192))
* **dynamics:** preserve operational gravity overrides and output shape ([#797](https://github.com/petercorke/robotics-toolbox-python/issues/797)) ([b9e71a5](https://github.com/petercorke/robotics-toolbox-python/commit/b9e71a57d4ea6b0295852641b779a6c23fa150a0))
* **dynamics:** support arbitrary joint counts in paycap ([#731](https://github.com/petercorke/robotics-toolbox-python/issues/731)) ([6a45f73](https://github.com/petercorke/robotics-toolbox-python/commit/6a45f732d66c38edbf4e8638b635d52f2602edb3))
* **dynamics:** transform operational Coriolis terms consistently ([#796](https://github.com/petercorke/robotics-toolbox-python/issues/796)) ([c93afaf](https://github.com/petercorke/robotics-toolbox-python/commit/c93afaf6bb36586968511b9e2ffcb5ffe513cd65))
* **ekf:** plot_error in SLAM, against time, honouring bgcolor, with correct bounds ([#749](https://github.com/petercorke/robotics-toolbox-python/issues/749)) ([0119d66](https://github.com/petercorke/robotics-toolbox-python/commit/0119d66719d7b74200e7c76354523afb676aa636))
* **ets2:** preserve joint indices and flipped Jacobian signs ([#792](https://github.com/petercorke/robotics-toolbox-python/issues/792)) ([8df90ac](https://github.com/petercorke/robotics-toolbox-python/commit/8df90ac3eb27a0b20a0d44d1226adbe3f45e2dfc))
* **ets:** reject boolean joint indices in fixjoint ([#836](https://github.com/petercorke/robotics-toolbox-python/issues/836)) ([4512e47](https://github.com/petercorke/robotics-toolbox-python/commit/4512e47f8f43c2e08781bf8c47ec13f747913f00))
* **ets:** swap accepts transforms that commute, merge rejects a joint, ET2.s is a 3-vector ([#832](https://github.com/petercorke/robotics-toolbox-python/issues/832)) ([f7edf54](https://github.com/petercorke/robotics-toolbox-python/commit/f7edf54923e0fb284450627c41b15bda5b9bdd6a))
* **ik:** a q0 or mask of the wrong length is an error, in the C++ and the Python solvers ([#831](https://github.com/petercorke/robotics-toolbox-python/issues/831)) ([f333582](https://github.com/petercorke/robotics-toolbox-python/commit/f3335827b2725640d354156e168424ba7257f8de))
* **ik:** apply QP updates at active joint indices ([#794](https://github.com/petercorke/robotics-toolbox-python/issues/794)) ([38909d2](https://github.com/petercorke/robotics-toolbox-python/commit/38909d2a16af83eb0e4ee60b055ed962b82188c6))
* **ik:** IKSolution is a sequence of the rows of q, true if successful ([#717](https://github.com/petercorke/robotics-toolbox-python/issues/717)) ([be4ad92](https://github.com/petercorke/robotics-toolbox-python/commit/be4ad92ecebe3488147c6dfaebcaefb041907f2c))
* **ik:** preserve arbitrary half-turn pose error magnitude ([#793](https://github.com/petercorke/robotics-toolbox-python/issues/793)) ([fe51972](https://github.com/petercorke/robotics-toolbox-python/commit/fe519723d16e9a799e1406745faa53247e25b1d9))
* **ik:** Robot.ikine_* accept an (N, 4, 4) array of poses ([#714](https://github.com/petercorke/robotics-toolbox-python/issues/714)) ([385747b](https://github.com/petercorke/robotics-toolbox-python/commit/385747b89d9cb97bad2b6d5a5b8b9854860be22b))
* **ik:** trajectory residual is the maximum, and document trajectory behaviour ([#712](https://github.com/petercorke/robotics-toolbox-python/issues/712)) ([aee6fb2](https://github.com/petercorke/robotics-toolbox-python/commit/aee6fb25ebdb15a2ff138ee6753d5c0266913804))
* **lite:** the notebooks install the wheel built with the site, and a browser check proves it ([#863](https://github.com/petercorke/robotics-toolbox-python/issues/863)) ([126c410](https://github.com/petercorke/robotics-toolbox-python/commit/126c410eee0fc4071d4501e59938f57e97fc68c3))
* **mobile:** align grid planners and backgrounds in world coordinates ([#791](https://github.com/petercorke/robotics-toolbox-python/issues/791)) ([b4e05b4](https://github.com/petercorke/robotics-toolbox-python/commit/b4e05b4a9babb40d6d5318441bb65a2b37295053))
* **mobile:** Bug2 animation, trapped, border handling, and docstrings ([#780](https://github.com/petercorke/robotics-toolbox-python/issues/780)) ([44ce707](https://github.com/petercorke/robotics-toolbox-python/commit/44ce7075b70492eb61678eb8ca73f1cc1ff4e185))
* **mobile:** DistanceTransform, PRM, QuinticPoly and CurvaturePoly planners ([#781](https://github.com/petercorke/robotics-toolbox-python/issues/781)) ([53becb4](https://github.com/petercorke/robotics-toolbox-python/commit/53becb4cd76e42482fa6aff2720d284f63e8e7dd))
* **mobile:** EKF map coordinates and range-bearing observation rows ([#784](https://github.com/petercorke/robotics-toolbox-python/issues/784)) ([4a93c88](https://github.com/petercorke/robotics-toolbox-python/commit/4a93c8893c190ceb6fa5ef86026cbf7d5853fee8))
* **mobile:** EKF map ellipses use each landmark's covariance; description, state copy, sensor type ([#787](https://github.com/petercorke/robotics-toolbox-python/issues/787)) ([d909d7c](https://github.com/petercorke/robotics-toolbox-python/commit/d909d7c4d116938f94728c5cdca8286ed0505cca))
* **mobile:** make PurePursuit usable; rst docstrings and shaped types in drivers.py ([#772](https://github.com/petercorke/robotics-toolbox-python/issues/772)) ([3a3a1d8](https://github.com/petercorke/robotics-toolbox-python/commit/3a3a1d8ece30513d227472e7484fdec4a879cce7))
* **mobile:** maps: stricter arguments, clearer names, docstrings and types ([#779](https://github.com/petercorke/robotics-toolbox-python/issues/779)) ([ec1e3fe](https://github.com/petercorke/robotics-toolbox-python/commit/ec1e3fe5ee9c6df39246c39a8e417583505b44dc))
* **mobile:** PlannerBase argument checks, plot without side effects, and &lt;thing&gt;_style arguments ([#778](https://github.com/petercorke/robotics-toolbox-python/issues/778)) ([84a07d2](https://github.com/petercorke/robotics-toolbox-python/commit/84a07d27201c2abefff0ef1988354f2113b1d28d))
* **mobile:** PoseGraph access and errors, verbose output, style arguments, pgraph 1.0 names ([#783](https://github.com/petercorke/robotics-toolbox-python/issues/783)) ([7def8fa](https://github.com/petercorke/robotics-toolbox-python/commit/7def8faf372bf8ee0dcd3ff5f3c7eeba5cb37bd3))
* **mobile:** preserve angular turn-rate units for Unicycle ([#719](https://github.com/petercorke/robotics-toolbox-python/issues/719)) ([606223f](https://github.com/petercorke/robotics-toolbox-python/commit/606223f542f0420ec66b40c7d6e8763fccdc1157))
* **mobile:** reject negative occupancy grid indices ([#790](https://github.com/petercorke/robotics-toolbox-python/issues/790)) ([be029bd](https://github.com/petercorke/robotics-toolbox-python/commit/be029bd1b8c4f0bc0de4f5aacbd9cda7b0b934b0))
* **mobile:** remove debug output from RRT query ([#751](https://github.com/petercorke/robotics-toolbox-python/issues/751)) ([c7c9b58](https://github.com/petercorke/robotics-toolbox-python/commit/c7c9b58f3ff003d75eba486a6c741c8ebcd9d809))
* **mobile:** resolve the workspace in one place, with a documented precedence ([#776](https://github.com/petercorke/robotics-toolbox-python/issues/776)) ([07090ed](https://github.com/petercorke/robotics-toolbox-python/commit/07090ed6ee6d85b3542b30d36fcc790532b5b9f2))
* **mobile:** type ParticleFilter, and fix x0, gif and mp4 output ([#883](https://github.com/petercorke/robotics-toolbox-python/issues/883)) ([6b85d77](https://github.com/petercorke/robotics-toolbox-python/commit/6b85d77c01d0ececacc57cf46a1613b3e8caf09e))
* **mobile:** VehicleBase control handling and plot_xyt, tests and docstrings ([#774](https://github.com/petercorke/robotics-toolbox-python/issues/774)) ([2ee9fa9](https://github.com/petercorke/robotics-toolbox-python/commit/2ee9fa9527e399078a67fd4bc4d92f776c4a2616))
* **particlefilter:** do not pause between steps unless animating ([#748](https://github.com/petercorke/robotics-toolbox-python/issues/748)) ([ea7b4d6](https://github.com/petercorke/robotics-toolbox-python/commit/ea7b4d68d0318b9b2267459ba8427704047a9cb8))
* **particlefilter:** get_xyt returns (x, y, theta) as documented ([#752](https://github.com/petercorke/robotics-toolbox-python/issues/752)) ([14775a7](https://github.com/petercorke/robotics-toolbox-python/commit/14775a7a1973bf623f061f9b14c44b5a66c5cfd2))
* **planning:** retain zero-coordinate path endpoints ([#798](https://github.com/petercorke/robotics-toolbox-python/issues/798)) ([72169ae](https://github.com/petercorke/robotics-toolbox-python/commit/72169ae042ccf21dd3654fe47bc25e774a164ef0))
* **plot:** a trajectory is animated whatever %matplotlib says ([#860](https://github.com/petercorke/robotics-toolbox-python/issues/860)) ([b853cb8](https://github.com/petercorke/robotics-toolbox-python/commit/b853cb808e5bfe08ecb26572fb4b455ad1c8ce63))
* **plot:** drive Swift through its handle, and test Swift in a notebook with a real kernel ([#865](https://github.com/petercorke/robotics-toolbox-python/issues/865)) ([7894328](https://github.com/petercorke/robotics-toolbox-python/commit/789432842c90fc0f57815e4346851058803fe691))
* **plot:** one implementation of the block= contract for every plot method ([#820](https://github.com/petercorke/robotics-toolbox-python/issues/820)) ([2881ee1](https://github.com/petercorke/robotics-toolbox-python/commit/2881ee1ed1daa581609e575e380ba17596d749f0))
* **plot:** PyPlot and PyPlot2 step() raise ValueError for a dt of zero or less ([#875](https://github.com/petercorke/robotics-toolbox-python/issues/875)) ([934ca7d](https://github.com/petercorke/robotics-toolbox-python/commit/934ca7dedde32975ea1cd91f18ab12bc6d619603))
* **plot:** Swift is shown in the cell of a notebook, and a trajectory is animated in real time ([#864](https://github.com/petercorke/robotics-toolbox-python/issues/864)) ([fa353d4](https://github.com/petercorke/robotics-toolbox-python/commit/fa353d49260d1b7941c7342844b06e2a8fa18bc4))
* **poe:** construct frames for x-aligned screw axes ([#799](https://github.com/petercorke/robotics-toolbox-python/issues/799)) ([7334f2e](https://github.com/petercorke/robotics-toolbox-python/commit/7334f2e9b866615dbb699b14e0f2917db26fc3aa))
* populate mstraj velocity and acceleration ([#706](https://github.com/petercorke/robotics-toolbox-python/issues/706)) ([8007e33](https://github.com/petercorke/robotics-toolbox-python/commit/8007e33ea836b01c2c4dcf379a91f253ed092cec))
* **posegraph:** ignore blank lines in g2o and toro files ([#745](https://github.com/petercorke/robotics-toolbox-python/issues/745)) ([c66d855](https://github.com/petercorke/robotics-toolbox-python/commit/c66d8553efa77d997db0ed46d90b6ee397ab4e05))
* **prm:** honour explicit constructor distance threshold ([#754](https://github.com/petercorke/robotics-toolbox-python/issues/754)) ([788eaf3](https://github.com/petercorke/robotics-toolbox-python/commit/788eaf36089c72fa91de37034b1e60310fff0521))
* **prm:** query before plan raises RuntimeError rather than AttributeError ([#744](https://github.com/petercorke/robotics-toolbox-python/issues/744)) ([e094f89](https://github.com/petercorke/robotics-toolbox-python/commit/e094f89ea5f5214ed4e113a6ce885092042f2213))
* **puma560:** joint 2, 3 and 4 limits are those of RTB-MATLAB, as the comments say ([#835](https://github.com/petercorke/robotics-toolbox-python/issues/835)) ([5afb34a](https://github.com/petercorke/robotics-toolbox-python/commit/5afb34a8a0ba5c0ba601af41358867bf26b41546))
* **puma:** self-contained ikine_a with trajectory and failure handling; deprecate ikine_6s ([#715](https://github.com/petercorke/robotics-toolbox-python/issues/715)) ([1a609c3](https://github.com/petercorke/robotics-toolbox-python/commit/1a609c36c3fd5bde87763b1ebd5de79e429e1dfb))
* **robot:** hasgeometry is true for planar robots whose links have geometry ([#814](https://github.com/petercorke/robotics-toolbox-python/issues/814)) ([4543500](https://github.com/petercorke/robotics-toolbox-python/commit/45435001e5fef22c6157a58fbaf7aff726090d5d))
* **robot:** improve jacob0_dot docstring, typing overloads, and representation error handling ([#737](https://github.com/petercorke/robotics-toolbox-python/issues/737)) ([b83b469](https://github.com/petercorke/robotics-toolbox-python/commit/b83b469a93359d0fbd1b2722fcf1fcce9f2abf82))
* **robot:** jtraj takes joint coordinates only, SE3 arguments raise ([#716](https://github.com/petercorke/robotics-toolbox-python/issues/716)) ([68185fc](https://github.com/petercorke/robotics-toolbox-python/commit/68185fc3c53f172189e2b7f02d5249213335b950))
* **robot:** qlim and printing work for a robot with symbolic parameters ([#856](https://github.com/petercorke/robotics-toolbox-python/issues/856)) ([7666c63](https://github.com/petercorke/robotics-toolbox-python/commit/7666c631c7c99f2288eaaddff41bcd2b614eab49))
* **robot:** Robot.set_default_backend replaces rtbtool's monkey-patch; URDFRobot drops a dead guard ([#810](https://github.com/petercorke/robotics-toolbox-python/issues/810)) ([681c725](https://github.com/petercorke/robotics-toolbox-python/commit/681c7250da406adc0f8ebb627d8b3bbcaf09067c))
* **robot:** the first line of print(robot) names the class and the model ([#862](https://github.com/petercorke/robotics-toolbox-python/issues/862)) ([4766ca6](https://github.com/petercorke/robotics-toolbox-python/commit/4766ca6b75384dcaeefa74e0b6a1025e9a546438))
* **robot:** the first line of print(robot) names the class and the model, as DHRobot.Puma560 ([4766ca6](https://github.com/petercorke/robotics-toolbox-python/commit/4766ca6b75384dcaeefa74e0b6a1025e9a546438))
* **robot:** the line of sight of vision_collision_damper has the right orientation for parallel and antiparallel vectors ([82974b1](https://github.com/petercorke/robotics-toolbox-python/commit/82974b10ecafb3b248359a0bbb7ac44a77058455))
* **robot:** the line of sight of vision_collision_damper is oriented correctly for parallel and antiparallel vectors ([#845](https://github.com/petercorke/robotics-toolbox-python/issues/845)) ([82974b1](https://github.com/petercorke/robotics-toolbox-python/commit/82974b10ecafb3b248359a0bbb7ac44a77058455))
* **slam:** report missing PoseGraph vertices with file context ([#760](https://github.com/petercorke/robotics-toolbox-python/issues/760)) ([dba2bfc](https://github.com/petercorke/robotics-toolbox-python/commit/dba2bfcd34a4c3ac9cf6693f6ee317cb8d23c127))
* **tests:** expect forward-slash filename from spatialgeometry &gt;= 1.4.1 ([#702](https://github.com/petercorke/robotics-toolbox-python/issues/702)) ([c070066](https://github.com/petercorke/robotics-toolbox-python/commit/c07006666f810c74f89865a15849afeea3b57189))
* **tools:** report original Jacobian column indices ([#795](https://github.com/petercorke/robotics-toolbox-python/issues/795)) ([cb4b591](https://github.com/petercorke/robotics-toolbox-python/commit/cb4b591775ded0fd5d899c27e3b9834ad39ff107))
* **tools:** the data.py demo block called a function that does not exist ([#809](https://github.com/petercorke/robotics-toolbox-python/issues/809)) ([27fd83a](https://github.com/petercorke/robotics-toolbox-python/commit/27fd83ada6112520675a6f00810b6df35fde9129))
* **trajectory:** honour non-zero time origins in scalar profiles ([#698](https://github.com/petercorke/robotics-toolbox-python/issues/698)) ([06b8716](https://github.com/petercorke/robotics-toolbox-python/commit/06b8716877b97bd28fb3ab09d5e24178315a2cba))
* **trajectory:** solve the quintic in normalized time, Hermite form behind method= ([#805](https://github.com/petercorke/robotics-toolbox-python/issues/805)) ([97c6c0f](https://github.com/petercorke/robotics-toolbox-python/commit/97c6c0fe15038069c366e5474b3a7fb1cd215fa2))
* **urdf:** initialise name maps before transmission validation ([#801](https://github.com/petercorke/robotics-toolbox-python/issues/801)) ([057751f](https://github.com/petercorke/robotics-toolbox-python/commit/057751f22688ed898b273410cfabf1d5fa639472))
* **urdf:** report stage, element and line when a URDF/xacro fails to load ([848114b](https://github.com/petercorke/robotics-toolbox-python/commit/848114b3a50518cf8d020239fc8d86a8dece82b0))
* **urdf:** report stage, element and line when URDF/xacro loading fails ([#701](https://github.com/petercorke/robotics-toolbox-python/issues/701)) ([848114b](https://github.com/petercorke/robotics-toolbox-python/commit/848114b3a50518cf8d020239fc8d86a8dece82b0))
* **urdf:** rotate inertia tensors from the inertial frame into the link frame ([#707](https://github.com/petercorke/robotics-toolbox-python/issues/707)) ([566cf70](https://github.com/petercorke/robotics-toolbox-python/commit/566cf70a04ee68d9f2d21ee98b5c0ad33d43b268))


### Performance Improvements

* **mobile:** index completed PRM roadmaps and benchmark queries ([#889](https://github.com/petercorke/robotics-toolbox-python/issues/889)) ([0e8d604](https://github.com/petercorke/robotics-toolbox-python/commit/0e8d604048519999c2a080bd9d33231fc60cc4b2))
* **mobile:** reduce RRT collision work and assemble sparse pose graphs ([#817](https://github.com/petercorke/robotics-toolbox-python/issues/817)) ([ff0e902](https://github.com/petercorke/robotics-toolbox-python/commit/ff0e902f11fe8e06d1d493e0f646a83cfbe940c1))


### Documentation

* add Codacy grade badge to README ([0372615](https://github.com/petercorke/robotics-toolbox-python/commit/03726154a0e99c389d45442645b51d4f2012ff3a))
* AGENTS.md says what the Toolbox is for ([#872](https://github.com/petercorke/robotics-toolbox-python/issues/872)) ([a379193](https://github.com/petercorke/robotics-toolbox-python/commit/a379193b3a5cd9f05255edfa256fae8f6ef18e61))
* **agents:** say where the array type aliases are ([#757](https://github.com/petercorke/robotics-toolbox-python/issues/757)) ([5c7d3d4](https://github.com/petercorke/robotics-toolbox-python/commit/5c7d3d4e1dc39def66a768691364386c16ab26c7))
* ask contributors to check for existing PRs and claim Issues first ([#829](https://github.com/petercorke/robotics-toolbox-python/issues/829)) ([02af061](https://github.com/petercorke/robotics-toolbox-python/commit/02af061711cb07c84bbb78420a4c4ce0c3d85c7f))
* **desiderata:** bring up to date with the code, and add status markers ([#885](https://github.com/petercorke/robotics-toolbox-python/issues/885)) ([31d99a9](https://github.com/petercorke/robotics-toolbox-python/commit/31d99a94ab19b6024250796e11773adaba70f040))
* **ets:** a section for ET and ETS, and a documentation pass: typos, cross-references, zero warnings ([#833](https://github.com/petercorke/robotics-toolbox-python/issues/833)) ([374720a](https://github.com/petercorke/robotics-toolbox-python/commit/374720a13ea65f0cf81522ab8296ab8c777feac1))
* **ets:** add fixjoint examples and version markers ([#877](https://github.com/petercorke/robotics-toolbox-python/issues/877)) ([093a760](https://github.com/petercorke/robotics-toolbox-python/commit/093a760811a01237e12e47cde08d0491d426bdf3))
* **ik:** add the autosummary template that the IK pages of [#833](https://github.com/petercorke/robotics-toolbox-python/issues/833) refer to ([#838](https://github.com/petercorke/robotics-toolbox-python/issues/838)) ([8839ac6](https://github.com/petercorke/robotics-toolbox-python/commit/8839ac6661d796f989fbca34cfdeb1bff6c52cc2))
* **ik:** say the same thing everywhere about when the C++ ik_XX solvers cannot be used ([#871](https://github.com/petercorke/robotics-toolbox-python/issues/871)) ([bf42b9f](https://github.com/petercorke/robotics-toolbox-python/commit/bf42b9f31ed41d5111d8327a524564037231bb4c))
* **ik:** the QP solvers say they need qpsolvers and quadprog, and the qp extra that installs them ([#841](https://github.com/petercorke/robotics-toolbox-python/issues/841)) ([9b72b1a](https://github.com/petercorke/robotics-toolbox-python/commit/9b72b1ab26ee6fc128ab6bcf142640a6afe72b98))
* landing pages for arms, mobile robots and blocks, with tables, introductions and a clearer sidebar ([#844](https://github.com/petercorke/robotics-toolbox-python/issues/844)) ([f0e22c0](https://github.com/petercorke/robotics-toolbox-python/commit/f0e22c0a1eed07b01bf1fad4474703cb3d01035d))
* **lite:** keep JupyterLite contents in memory so republished notebooks are never stale ([#819](https://github.com/petercorke/robotics-toolbox-python/issues/819)) ([4e8073d](https://github.com/petercorke/robotics-toolbox-python/commit/4e8073d9930d51abf3223e0c69d71fd02d99dbae))
* **mobile:** correct docstrings for the estimators, sensor, drivers and animations ([#750](https://github.com/petercorke/robotics-toolbox-python/issues/750)) ([b6e86e8](https://github.com/petercorke/robotics-toolbox-python/commit/b6e86e8212127a712c31ca20073526c54bf1813a))
* **notebooks:** a standard generated first cell, and a shorter bootstrap cell ([#850](https://github.com/petercorke/robotics-toolbox-python/issues/850)) ([2b2fa83](https://github.com/petercorke/robotics-toolbox-python/commit/2b2fa832768aa0a195e1550ce468e23713c1dc20))
* **notebooks:** drop the deprecated path_marker argument ([#815](https://github.com/petercorke/robotics-toolbox-python/issues/815)) ([689ae2a](https://github.com/petercorke/robotics-toolbox-python/commit/689ae2a4048780260e7a950b057d59ae9313cb35))
* **notebooks:** drop the path_marker argument, which is deprecated ([689ae2a](https://github.com/petercorke/robotics-toolbox-python/commit/689ae2a4048780260e7a950b057d59ae9313cb35))
* **notebooks:** follow the notebook plotting recipe, and fix a broken qplot call ([#821](https://github.com/petercorke/robotics-toolbox-python/issues/821)) ([7690e80](https://github.com/petercorke/robotics-toolbox-python/commit/7690e80f04676f6eca9e0b225dfa10fadd1a184a))
* **notebooks:** modernise branched-robot.ipynb for the current Robot API ([#852](https://github.com/petercorke/robotics-toolbox-python/issues/852)) ([2d8abf1](https://github.com/petercorke/robotics-toolbox-python/commit/2d8abf156d8f408c343f752fb60a5687d8c7829b))
* **notebooks:** modernise ets.ipynb for the current ET and ETS API ([#851](https://github.com/petercorke/robotics-toolbox-python/issues/851)) ([1460de7](https://github.com/petercorke/robotics-toolbox-python/commit/1460de73d019ff371e79f3664c13c1705a6d5fc5))
* **notebooks:** modernise icra2021.ipynb for the current API ([#854](https://github.com/petercorke/robotics-toolbox-python/issues/854)) ([1ddf6dd](https://github.com/petercorke/robotics-toolbox-python/commit/1ddf6dd7f62e299ed1fc90542f3ba76357748c5d))
* **notebooks:** modernise iros2020.ipynb (renamed from iros2020-workshop) for the current API ([#855](https://github.com/petercorke/robotics-toolbox-python/issues/855)) ([af4de2c](https://github.com/petercorke/robotics-toolbox-python/commit/af4de2cebe16def065e9fd8459610d5f761d3029))
* **notebooks:** modernise symbolic.ipynb for the current API and make it fast to run ([#857](https://github.com/petercorke/robotics-toolbox-python/issues/857)) ([bd9e933](https://github.com/petercorke/robotics-toolbox-python/commit/bd9e933c63ab7490d7e64252a23720c1215e5ba6))
* **notebooks:** modernise two-link-dynamics.ipynb, including the sign convention of gravity ([#853](https://github.com/petercorke/robotics-toolbox-python/issues/853)) ([e778eb6](https://github.com/petercorke/robotics-toolbox-python/commit/e778eb6c1bd949e8edaf65871a01dac7c12a8f96))
* **notebooks:** review and refresh the notebooks, add an index ([#867](https://github.com/petercorke/robotics-toolbox-python/issues/867)) ([c72f1bc](https://github.com/petercorke/robotics-toolbox-python/commit/c72f1bc1ad6a669bc98987f6a8e89ff6b1029abd))
* **notebooks:** the first cell is hand-written markdown again, and cell ids are committed ([#859](https://github.com/petercorke/robotics-toolbox-python/issues/859)) ([742b6ed](https://github.com/petercorke/robotics-toolbox-python/commit/742b6edc488936b5baf5837332f710be7c356932))
* one style for docstring section headings, and 1.5.0 for the next release ([#882](https://github.com/petercorke/robotics-toolbox-python/issues/882)) ([6497cdc](https://github.com/petercorke/robotics-toolbox-python/commit/6497cdc6f6b4566ff3949b3c0bd6d12b1677d634))
* say which checks are required, and that Codacy is advisory ([#802](https://github.com/petercorke/robotics-toolbox-python/issues/802)) ([1ec699a](https://github.com/petercorke/robotics-toolbox-python/commit/1ec699a76f154a4b4253b83984bdb64bd4529802))
* update copyright information in LICENSE file ([#710](https://github.com/petercorke/robotics-toolbox-python/issues/710)) ([fc969a3](https://github.com/petercorke/robotics-toolbox-python/commit/fc969a3f914474d28c8bdaeca57285253d6f1a4b))


### Build System

* **deps:** bump pypa/cibuildwheel from 4.2.0 to 4.2.1 ([968ed04](https://github.com/petercorke/robotics-toolbox-python/commit/968ed048639c01c9032e256565731e2703de77fe))
* **deps:** bump pypa/cibuildwheel from 4.2.0 to 4.2.1 ([acc6842](https://github.com/petercorke/robotics-toolbox-python/commit/acc6842d47b36cc6c9eef97373ed7e3d45f27491))

## [1.4.4](https://github.com/petercorke/robotics-toolbox-python/compare/v1.4.3...v1.4.4) (2026-09-20)


### Features

* **urdf:** Panda(use_robot_descriptions=True) option; doc the tradeoff ([#687](https://github.com/petercorke/robotics-toolbox-python/issues/687)) ([124882b](https://github.com/petercorke/robotics-toolbox-python/commit/124882b06880dff49c015e1156041435eaa0023c))


### Bug Fixes

* **dhrobot:** ikine_6s undoes the tool transform on the wrong side ([#680](https://github.com/petercorke/robotics-toolbox-python/issues/680)) ([9fbffc6](https://github.com/petercorke/robotics-toolbox-python/commit/9fbffc6f09265f084e1b53c04b09a75a4aced643)), closes [#640](https://github.com/petercorke/robotics-toolbox-python/issues/640)
* **docs:** mock bdsim imports so blocks pages render in docs CI ([436e1f3](https://github.com/petercorke/robotics-toolbox-python/commit/436e1f3bbd26faa26eed97ddd47cbf47609bf216))
* **examples:** call joint_velocity_damper with keyword args ([#690](https://github.com/petercorke/robotics-toolbox-python/issues/690)) ([95ac98e](https://github.com/petercorke/robotics-toolbox-python/commit/95ac98e31876b2226b9f51778ba149e105de78c7))
* **examples:** call joint_velocity_damper with keyword args in fetch_vision.py ([#695](https://github.com/petercorke/robotics-toolbox-python/issues/695)) ([7aa2d31](https://github.com/petercorke/robotics-toolbox-python/commit/7aa2d317445c84e8eeb8ea64e78cafce2fb938d5))
* **joint_velocity_damper:** fix bug in q usage ([#666](https://github.com/petercorke/robotics-toolbox-python/issues/666)) ([8246559](https://github.com/petercorke/robotics-toolbox-python/commit/8246559202dd1722f8b6538daad66aed735ca928))
* **kinematics:** self.tool now applies by default; ik_LM/ik_NR/ik_GN/ikine_XX gain a tool= kwarg ([#697](https://github.com/petercorke/robotics-toolbox-python/issues/697)) ([61d078f](https://github.com/petercorke/robotics-toolbox-python/commit/61d078fcdf419199209ef5cf931c2c2ef51e0859))
* link_collision_damper ([#685](https://github.com/petercorke/robotics-toolbox-python/issues/685)) ([c040522](https://github.com/petercorke/robotics-toolbox-python/commit/c04052208094fe458efd707c84fe970580ac2fe2))
* preserve joint types and limits in numerical IK ([e122ccb](https://github.com/petercorke/robotics-toolbox-python/commit/e122ccb9a1ad003879a63ff0fa6d3accdcddde26))
* **rne:** attribute a sandwiched static link's mass to its preceding joint, not the following one ([#693](https://github.com/petercorke/robotics-toolbox-python/issues/693)) ([bebede0](https://github.com/petercorke/robotics-toolbox-python/commit/bebede019601f4b641cdc5ed8e390f480fad766d))
* **rne:** don't crash when a joint's kinematic parent is a base-mounted static link ([#696](https://github.com/petercorke/robotics-toolbox-python/issues/696)) ([dba179e](https://github.com/petercorke/robotics-toolbox-python/commit/dba179ed84eae8162257d434197d77609ceb0b15))
* **rne:** propagate backward-pass force through the correct transform direction ([#691](https://github.com/petercorke/robotics-toolbox-python/issues/691)) ([a80c9c1](https://github.com/petercorke/robotics-toolbox-python/commit/a80c9c1c4c609c35494d7f127054c408f993635f))
* **rne:** trailing static links after the last joint had their mass dropped ([#684](https://github.com/petercorke/robotics-toolbox-python/issues/684)) ([0d76187](https://github.com/petercorke/robotics-toolbox-python/commit/0d76187b21d86713fc13a36ed6bc00a275eab858))
* **rtbtool:** drop stray quote in banner's import line ([#678](https://github.com/petercorke/robotics-toolbox-python/issues/678)) ([b36c398](https://github.com/petercorke/robotics-toolbox-python/commit/b36c398aa81de96bc0784c25601ade1bb95e7e73))
* **urdf:** resolve UR5's gripper link by name, not a raw index ([#683](https://github.com/petercorke/robotics-toolbox-python/issues/683)) ([12ba598](https://github.com/petercorke/robotics-toolbox-python/commit/12ba598a58c1ae21db9145ce4dbca798e8065652)), closes [#578](https://github.com/petercorke/robotics-toolbox-python/issues/578)


### Documentation

* **models:** credit robot_descriptions in the 8 model classes that use it ([#681](https://github.com/petercorke/robotics-toolbox-python/issues/681)) ([5dc8618](https://github.com/petercorke/robotics-toolbox-python/commit/5dc86184df70ec4cc639db76d203fb534ce8a3f7))
* **models:** note UR10's near-singular inertia matrix is manufacturer data ([#682](https://github.com/petercorke/robotics-toolbox-python/issues/682)) ([97b0a3f](https://github.com/petercorke/robotics-toolbox-python/commit/97b0a3fa554733ca1247793f199cc6a6d9ba40c1))
* **readme:** move research applications section to the wiki ([#689](https://github.com/petercorke/robotics-toolbox-python/issues/689)) ([8facc5e](https://github.com/petercorke/robotics-toolbox-python/commit/8facc5eeef8fc5c9ec93b5f1c1f340b0a4fa03ea))


### Miscellaneous

* release as 1.4.4 ([758a81b](https://github.com/petercorke/robotics-toolbox-python/commit/758a81b26fe1302a660aeba3a47b79abb76c2855))

## [1.4.3](https://github.com/petercorke/robotics-toolbox-python/compare/v1.4.2...v1.4.3) (2026-09-12)


### Bug Fixes

* check IK convergence before solver step ([aabe415](https://github.com/petercorke/robotics-toolbox-python/commit/aabe415af02c0fe9336fd9eb3693e962b5b577e2))


### Documentation

* expand trchain runblock examples ([#655](https://github.com/petercorke/robotics-toolbox-python/issues/655)) ([9e6c00c](https://github.com/petercorke/robotics-toolbox-python/commit/9e6c00c63b7cee1c94476bac5bc7e113e48cf9ab))
* fix broken URL ([#657](https://github.com/petercorke/robotics-toolbox-python/issues/657)) ([2d24e0d](https://github.com/petercorke/robotics-toolbox-python/commit/2d24e0d8f75c15bd7937f78347b876b0d51d7cf4))
* **readme:** remove wasm info ([#656](https://github.com/petercorke/robotics-toolbox-python/issues/656)) ([4ba11d1](https://github.com/petercorke/robotics-toolbox-python/commit/4ba11d19523ef79dcba1beede3fc79ca0cbe8f8f))
* rewrite CONTRIBUTING.md ([cd9c619](https://github.com/petercorke/robotics-toolbox-python/commit/cd9c619040f3f98acaea9a67520c67b3d74b30e9))
* showcase trchain runblock examples ([9e6c00c](https://github.com/petercorke/robotics-toolbox-python/commit/9e6c00c63b7cee1c94476bac5bc7e113e48cf9ab))


### Build System

* **deps:** bump actions/cache from 4 to 6 ([1ebf302](https://github.com/petercorke/robotics-toolbox-python/commit/1ebf302c125f0cec4fb368eac8ce4d0c7364cad2))
* **deps:** bump pypa/cibuildwheel from 4.1.1 to 4.2.0 ([efbca2a](https://github.com/petercorke/robotics-toolbox-python/commit/efbca2a27d287fd4aa6e69e3c2f1c7ede875fc3b))

## [1.4.2](https://github.com/petercorke/robotics-toolbox-python/compare/v1.4.1...v1.4.2) (2026-08-26)

**Headline**: Fixes to inverse kinematics and forward kinematics for branched robots. Arising from a big trawl through long standing issues (thanks to all those users), many were resolved by recent work, but some were not.
Issues down from 104 to 43 of which 20 are `tech-debt` to be worked through.

### Features

* **ets:** fkine/jacob0/jacobe/hessian0/hessiane accept compact q ([#653](https://github.com/petercorke/robotics-toolbox-python/issues/653)) ([d5b6990](https://github.com/petercorke/robotics-toolbox-python/commit/d5b699025e48afeca147fb2e8b246557d2059c47))
* **examples:** add ikine_XX (C++ and pure-Python ETS) to ik_speed.py ([2f41a71](https://github.com/petercorke/robotics-toolbox-python/commit/2f41a711eba92f97ad1ed38dcd4fa69a87bccb26))
* **examples:** add ikine_XX timing columns to ik_speed.py ([#654](https://github.com/petercorke/robotics-toolbox-python/issues/654)) ([2f41a71](https://github.com/petercorke/robotics-toolbox-python/commit/2f41a711eba92f97ad1ed38dcd4fa69a87bccb26))


### Bug Fixes

* **blocks:** FDyn_X.output() crashed/asserted on unset xdd before deriv() ran ([#647](https://github.com/petercorke/robotics-toolbox-python/issues/647)) ([8caa363](https://github.com/petercorke/robotics-toolbox-python/commit/8caa363a98c158621d6418ccad8f0bbdc93fee11))
* **examples,dhrobot:** repair broken ikine_LMS/ikine_min example references ([#650](https://github.com/petercorke/robotics-toolbox-python/issues/650)) ([6691113](https://github.com/petercorke/robotics-toolbox-python/commit/66911133fdbdb5cf73cb24fcca058f457270099c))
* **ik:** _random_q() silently produces garbage for non-finite joint limits ([#648](https://github.com/petercorke/robotics-toolbox-python/issues/648)) ([2f43a3c](https://github.com/petercorke/robotics-toolbox-python/commit/2f43a3c2812dc8f30d8601a54d453ef158c8b8e0))
* **ik:** unify ik_XX/ikine_XX return types and fix IK docs ([#652](https://github.com/petercorke/robotics-toolbox-python/issues/652)) ([07a60f4](https://github.com/petercorke/robotics-toolbox-python/commit/07a60f4a9fe9bf3de94aaf2428f93f75a7d77d3f))
* **kinematics:** fkine_all() computes wrong poses for links past a nominal end-effector ([#643](https://github.com/petercorke/robotics-toolbox-python/issues/643)) ([25eedcd](https://github.com/petercorke/robotics-toolbox-python/commit/25eedcd4294856cab0a8de346c72ce0a48a50b2e))
* **mobile:** restore missing distance-transform diagonal ([835457e](https://github.com/petercorke/robotics-toolbox-python/commit/835457e0428ade74b68220395a09224737f40d79))
* **models,ik:** correct YuMi gripper parents and IK failure-path q compaction ([#649](https://github.com/petercorke/robotics-toolbox-python/issues/649)) ([7c40242](https://github.com/petercorke/robotics-toolbox-python/commit/7c40242dec4dafcc12940c3eeb9da67f9d2aa7fa))
* **pyplot:** robot.plot(..., movie=...) crashes on save ([#645](https://github.com/petercorke/robotics-toolbox-python/issues/645)) ([2e2bfe2](https://github.com/petercorke/robotics-toolbox-python/commit/2e2bfe2ada4c43f3212db48f052986b25da6254d))
* **pyplot:** scalar plot options (jointaxislength, eelength) crash on override ([a4aea19](https://github.com/petercorke/robotics-toolbox-python/commit/a4aea193354631dc5e208f00f556ff181ba713bb))
* **pyplot:** scalar plot options crash on override ([#642](https://github.com/petercorke/robotics-toolbox-python/issues/642)) ([a4aea19](https://github.com/petercorke/robotics-toolbox-python/commit/a4aea193354631dc5e208f00f556ff181ba713bb))


### Documentation

* fix broken PyPlot backend example in intro.rst ([#641](https://github.com/petercorke/robotics-toolbox-python/issues/641)) ([5e8d2f9](https://github.com/petercorke/robotics-toolbox-python/commit/5e8d2f96f557d2a0014217e8820cbadcd9136889))


### Miscellaneous

* release 1.4.2 ([440a9ec](https://github.com/petercorke/robotics-toolbox-python/commit/440a9eca955430e1fd6f97ed63781e86648d1026))

## [1.4.1](https://github.com/petercorke/robotics-toolbox-python/compare/v1.4.0...v1.4.1) (2026-08-25)


### Bug Fixes

* **ci:** add CIBW_ENABLE=pyodide-eol, required to build the cp312 pyodide target ([2243fc2](https://github.com/petercorke/robotics-toolbox-python/commit/2243fc2990197baa5e60d53e49d8d32af8daa185))
* **ci:** drop the explicit pyodide-version override, let cibuildwheel 3.4.1 self-resolve ([394d258](https://github.com/petercorke/robotics-toolbox-python/commit/394d258f9e65acebc94374eb1bf93fde980c2fd6))
* **ci:** keep JupyterLite wasm build on JSPI-safe cp312, not cibuildwheel's new cp313 default ([a315095](https://github.com/petercorke/robotics-toolbox-python/commit/a3150958dee84285f147702b40e5fe3b13f2283c))
* **ci:** pin cibuildwheel 3.4.1 for the pyodide build job, not 4.1.1 ([e428935](https://github.com/petercorke/robotics-toolbox-python/commit/e428935b9fdb8c33647bbcfb1e011999a147eb3a))
* **ci:** stop skip-cascade from release-please poisoning downstream jobs ([f252325](https://github.com/petercorke/robotics-toolbox-python/commit/f2523251a95f19d73d4fb393a15878cc3fbc7803))
* compute accel_x with analytical Jacobian derivative ([1c4ffe6](https://github.com/petercorke/robotics-toolbox-python/commit/1c4ffe6145ddc2a42584006a44147094d0f537b4))
* **plot:** use matplotlib colormaps registry instead of removed cm.get_cmap ([ad14468](https://github.com/petercorke/robotics-toolbox-python/commit/ad1446895c15bb679acf0ff35ffd901e1d591362))
* two real bugs found by running the suite against the pure-Python path ([d4d2af3](https://github.com/petercorke/robotics-toolbox-python/commit/d4d2af3a81dc18731019dbd04c562c588d628e81))


### Build System

* add RTB_PURE_WHEEL to build a genuine py3-none-any wheel ([d61729c](https://github.com/petercorke/robotics-toolbox-python/commit/d61729c38db2174fbe6933f59a9d85447a03339e))

## [1.4.0](https://github.com/petercorke/robotics-toolbox-python/compare/v1.3.1...v1.4.0) (2026-08-22)

**Highlights**: using significantly reworked versions of [Swift](https://github.com/jhavl/swift) (v2.0.0) and [Spatial Geometry](https://github.com/jhavl/spatialgeometry) (v1.4.0) -- lots of bug fixes, new features and documentation.  Collision detection changed from PyBullet (no longer well maintained) to [Coal](https://github.com/coal-library/coal).  Integrates with [Robot Descriptions](https://github.com/robot-descriptions) for access to over 150 URDF models, loaded dynamically and cached locally -- this means that RTB can now ship with a very minimum set of mesh models.  Significant work on RNE models (DH Python, DH C, ETS Featherstone) all consistent and checked against analytic model.  `robot.teach()` now supports the Swift backend directly, and the docs site gained a JupyterLite-powered "Try it Now" in-browser notebook -- no install required.  Additional work on docs, build system, and testing.  

Significant amount of tech debt identified, these are now tracked in Issues. Happy for help from **first time contributors**, projects from bite size to a full meal.

### Features

* bundle spatialgeometry as pure-Python (removes external dep) ([1b522e6](https://github.com/petercorke/robotics-toolbox-python/commit/1b522e65f1e50ca37d95d036cad7f405a7c71b5d))
* change collision backend from PyBullet to Coal ([f99c154](https://github.com/petercorke/robotics-toolbox-python/commit/f99c154f0685b5c13661603e40dffaa4df6d22e7))
* **docs:** add JupyterLite in-browser "Try it Now" deployment ([d383f55](https://github.com/petercorke/robotics-toolbox-python/commit/d383f55e13c14928adccd0cedd2b8b76c2270239))
* **dynamics:** add armature inertia and friction to Robot.rne() ([1d55edf](https://github.com/petercorke/robotics-toolbox-python/commit/1d55edfc7d3341c050600ef085f384906d755e1e))
* **et:** accept a string joint descriptor in param (jindex/flip parsing) ([ab0f794](https://github.com/petercorke/robotics-toolbox-python/commit/ab0f794e30ff4ce443994428108466909a9eddc7))
* **et:** rename axis -&gt; kind, add ax for the x/y/z-only concept ([8ffc493](https://github.com/petercorke/robotics-toolbox-python/commit/8ffc493719ebf846873e5395fb365bc35f8bcf34))
* **et:** rename eta -&gt; param, keep eta as a deprecated alias ([7f0df78](https://github.com/petercorke/robotics-toolbox-python/commit/7f0df78c137be82418a08b2d8716954598fd7539))
* **ets:** rigorous dh/mdh split() convention, sum() support, terser tests ([a0674e7](https://github.com/petercorke/robotics-toolbox-python/commit/a0674e79a367cc03e7bc908bd74b152f92a11b13))
* **models:** add Panda DH link centre-of-mass parameters ([fc2fe20](https://github.com/petercorke/robotics-toolbox-python/commit/fc2fe20d0a352b9baab565cc124f9f7765b75de7))
* refactor URDF loading, add Jaco model, fix collision and deepcopy ([d8f9289](https://github.com/petercorke/robotics-toolbox-python/commit/d8f9289d19e9955afcc32b9b856f59ec11705a65))
* robot model listing cleanup, extended display, sorting option ([40c4882](https://github.com/petercorke/robotics-toolbox-python/commit/40c4882bdf5f0249e8bf473707480eece35ab932))
* **robot:** add BaseRobot.graph() for multi-format link graph text ([fca095c](https://github.com/petercorke/robotics-toolbox-python/commit/fca095c4b0980e6a9d1fc8669ca0ebcff4013013))
* **robot:** add Robot.fkine_geometry(q) for pure per-part FK ([8dbb44e](https://github.com/petercorke/robotics-toolbox-python/commit/8dbb44e233eff617c37660c857f0dbaa2ba6a62b))
* **rtbtool:** add --test smoke-test flag, fix SG version bug ([ae3ba73](https://github.com/petercorke/robotics-toolbox-python/commit/ae3ba73874523fe3cd6f3dccd6445407ea813dab))
* **rtbtool:** add 'tool' extra for IPython/pygments, fail with clear message when missing ([b791bc8](https://github.com/petercorke/robotics-toolbox-python/commit/b791bc8455aa6cb9aeb20c1c86d6c1b417488d2d))
* **swift:** support teach() teach panel for Swift-backed robots ([1be412d](https://github.com/petercorke/robotics-toolbox-python/commit/1be412da2bcdc6214c99920f17a71df34fda83ee))


### Bug Fixes

* 343 array ordering causing fkine traj error ([083cdc4](https://github.com/petercorke/robotics-toolbox-python/commit/083cdc434b79e829631d3ab90df1160a0e8f60b4))
* 355 add qlim setters ([bae653b](https://github.com/petercorke/robotics-toolbox-python/commit/bae653b09a837f48b92859bd32dc064e91146c2d))
* 361 flipped joint causing Jacobian error ([f0ec2f8](https://github.com/petercorke/robotics-toolbox-python/commit/f0ec2f8d34840fcd6beea2e5084c7c9301d116f3))
* 366 symbolic printing handled ([eeedeae](https://github.com/petercorke/robotics-toolbox-python/commit/eeedeae2d521e2c4ac2682ddad36c270dcf1f748))
* 377 add setitem method to ETS class ([3c0e463](https://github.com/petercorke/robotics-toolbox-python/commit/3c0e4637d528f2de3bf7f544e3920fb8bfbc08d1))
* bdsim blocks and tests ([43e89cf](https://github.com/petercorke/robotics-toolbox-python/commit/43e89cfa01ba7990f654fad9865a75a1ab476d98))
* **ci:** stopgap the JupyterLite piplite index wheel-overwrite bug ([11a3f29](https://github.com/petercorke/robotics-toolbox-python/commit/11a3f29e1bae55a0dc3c27cc7e10848360e3e0b0))
* **ci:** stopgap the JupyterLite piplite index wheel-overwrite bug ([695af7b](https://github.com/petercorke/robotics-toolbox-python/commit/695af7b8ebdd3221bf563f67662b235c2238d159))
* deprecate Bicycle block's vlim/slim kwargs instead of erroring ([8d7449c](https://github.com/petercorke/robotics-toolbox-python/commit/8d7449c8c2672c6ec817d70f370d73cec8c17c67))
* deprecate RangeBearingSensor's animate kwarg instead of erroring ([#599](https://github.com/petercorke/robotics-toolbox-python/issues/599)) ([1885394](https://github.com/petercorke/robotics-toolbox-python/commit/1885394bcf64de1f7195660ad59ae0911a6052b6))
* **deps:** add coal/trimesh to dev extras ([#528](https://github.com/petercorke/robotics-toolbox-python/issues/528)) ([3bb5847](https://github.com/petercorke/robotics-toolbox-python/commit/3bb5847642e6e6f9b500016128e470b8d647dd0d))
* **deps:** add missing robot_descriptions dependency ([#534](https://github.com/petercorke/robotics-toolbox-python/issues/534)) ([66c40d4](https://github.com/petercorke/robotics-toolbox-python/commit/66c40d43365a269669488d5bebf958d16fc32931))
* **deps:** don't require coal on Windows, fix stale install docs ([#531](https://github.com/petercorke/robotics-toolbox-python/issues/531)) ([667084a](https://github.com/petercorke/robotics-toolbox-python/commit/667084ad82ee7e5e8d20c41cc8ca2450f594d41e))
* **deps:** pin robot_descriptions&lt;3.0, UR5 broke on the unpinned major bump ([48715b3](https://github.com/petercorke/robotics-toolbox-python/commit/48715b349344d47a0680fcc8d9ef63f19b2d5f9c))
* **deps:** pin robot_descriptions&lt;3.0, UR5 broke on the unpinned major bump ([b00b46b](https://github.com/petercorke/robotics-toolbox-python/commit/b00b46b33623f3ea1cccef9b2d8503a9d7d07af7))
* **docs:** fix intro.rst syntax errors, dangling section, and stale claims ([#549](https://github.com/petercorke/robotics-toolbox-python/issues/549)) ([29ee42b](https://github.com/petercorke/robotics-toolbox-python/commit/29ee42b71b4d905fbb714fa6a308ae09fd2e0293))
* **docs:** pick a collision-checking example that actually collides ([3240822](https://github.com/petercorke/robotics-toolbox-python/commit/3240822318608d981b6cba46db670fe5eb543b0e))
* **docs:** resolve reST syntax and structural Sphinx build warnings ([#540](https://github.com/petercorke/robotics-toolbox-python/issues/540)) ([cd4e997](https://github.com/petercorke/robotics-toolbox-python/commit/cd4e9970a0e213e58b33637a38bade543b89fc0e))
* docstrings in robot models ([912cc4b](https://github.com/petercorke/robotics-toolbox-python/commit/912cc4b3dc7a095cc3103cf0cf48c62e0fb86c1e))
* drop stale ax= kwarg from ArmPlot's robot.plot() call ([#590](https://github.com/petercorke/robotics-toolbox-python/issues/590)) ([09dba28](https://github.com/petercorke/robotics-toolbox-python/commit/09dba2804c2b6f002367fa5370e0b4401298b1a9))
* **dynamics:** auto-detect symbolic models, fix rne_python()'s per-call dtype selection ([c9db413](https://github.com/petercorke/robotics-toolbox-python/commit/c9db41370450e3ad2ba881d77c2a0620ab8a6838))
* **dynamics:** fix three bugs in rne_python()'s modified-DH branch ([6b28899](https://github.com/petercorke/robotics-toolbox-python/commit/6b2889909fcd59b0d19a7f8d27adf881780b1fa7))
* **dynamics:** guard Robot.rne() against standard-DH DHRobot instances ([13e62a7](https://github.com/petercorke/robotics-toolbox-python/commit/13e62a72680195369ba9c8ab88d7afb5f3bf1991))
* **dynamics:** Robot.rne() dropped the rotational inertia tensor ([8434fc9](https://github.com/petercorke/robotics-toolbox-python/commit/8434fc92ef542e201f707bae04d45363c3c61873))
* **dynamics:** rotate gravity by robot base in rne(), move handling into ne.c ([9a800e0](https://github.com/petercorke/robotics-toolbox-python/commit/9a800e05e3d2431ccccc813cab80393be5308c65))
* **examples:** repair ik_speed.py after fknm/frne refactor and API renames ([227f854](https://github.com/petercorke/robotics-toolbox-python/commit/227f854d7915541e9795a6de05c2309c958801d3))
* forward name/base/tool/etc through Robot's clone-from-Robot constructor ([16b8974](https://github.com/petercorke/robotics-toolbox-python/commit/16b8974b967c67fa1d355f1ee3294d0cda67f0d2))
* **ik:** pin seed and tighten tol in flaky ikine tests, document quadratic tol ([bc3770a](https://github.com/petercorke/robotics-toolbox-python/commit/bc3770a73b73eabe08c069770f575bcd72bb80fa))
* **ik:** return pre-step q on convergence, not unverified post-step q ([3606c3b](https://github.com/petercorke/robotics-toolbox-python/commit/3606c3b55ef66c32fe2b0809475427d2fc3b0090))
* install swift/spatialgeometry from jhavl upstream, not petercorke's stale forks ([#601](https://github.com/petercorke/robotics-toolbox-python/issues/601)) ([4fd2dff](https://github.com/petercorke/robotics-toolbox-python/commit/4fd2dff70f2dc5926caa6dbcfaa46c57d3f66bc9))
* Mesh filename handling ([#443](https://github.com/petercorke/robotics-toolbox-python/issues/443)) ([4f0d3ff](https://github.com/petercorke/robotics-toolbox-python/commit/4f0d3ffc88081b2a27c2dd944cfe5a9d991362ec))
* **mobile:** DistanceTransformPlanner.next() calls undefined Error() ([e876f58](https://github.com/petercorke/robotics-toolbox-python/commit/e876f58f899159611c4cf1a26c883c2fc33dfd59))
* **mobile:** gate driveto()'s per-step print behind verbose flag ([aac83e1](https://github.com/petercorke/robotics-toolbox-python/commit/aac83e187dc02e48e21ef7c0a949dac968864d69))
* **mobile:** gate driveto()'s per-step print behind verbose flag ([ef4853a](https://github.com/petercorke/robotics-toolbox-python/commit/ef4853af8b0b73146483f41c20e79f91fd7eebd0))
* **mobile:** gate driveto()'s per-step print behind verbose flag ([2115bb3](https://github.com/petercorke/robotics-toolbox-python/commit/2115bb30b0910fa28501c1b98e5a841e2af0232c))
* **mobile:** guard VehicleDriverBase against workspace=None, fix RandomPath docstring example ([#541](https://github.com/petercorke/robotics-toolbox-python/issues/541)) ([901cabb](https://github.com/petercorke/robotics-toolbox-python/commit/901cabb03f906e090e050e118d37da92baabdd9f))
* **models:** add missing URDFRobot.py and Jaco.py ([839ab74](https://github.com/petercorke/robotics-toolbox-python/commit/839ab74ec6285e20641991c2b00da38a9d70c974))
* **models:** distinguish not-found vs renamed robot_descriptions models ([#537](https://github.com/petercorke/robotics-toolbox-python/issues/537)) ([68da27f](https://github.com/petercorke/robotics-toolbox-python/commit/68da27f022835083662270a5998d874b89714568))
* **models:** fall back to robot_descriptions' new naming, hide wrapped dependency on failure ([#529](https://github.com/petercorke/robotics-toolbox-python/issues/529)) ([d8441a3](https://github.com/petercorke/robotics-toolbox-python/commit/d8441a30389f36ba5162c69feeade880d2069e3f))
* **models:** fix 3 more addconfiguration() length mismatches surfaced on current main ([41ef991](https://github.com/petercorke/robotics-toolbox-python/commit/41ef99170fceee69c96de785d7176e95ad80cb6f))
* **models:** fix qz configuration size mismatch in px150/rx150/rx200 ([a1f10c8](https://github.com/petercorke/robotics-toolbox-python/commit/a1f10c80d9390149f87d9bc1764d9a0ace9e19d5))
* **models:** make KinovaGen3 load via robot_descriptions, add XACRO_ARGS support ([#546](https://github.com/petercorke/robotics-toolbox-python/issues/546)) ([d746ce4](https://github.com/petercorke/robotics-toolbox-python/commit/d746ce48a16f4ec04db24f431094bbaac4498299))
* **models:** patch broken upstream RD files for Valkyrie and Fetch ([#543](https://github.com/petercorke/robotics-toolbox-python/issues/543)) ([acb8c6b](https://github.com/petercorke/robotics-toolbox-python/commit/acb8c6bc6e9c5f55021a7c41e0289d6a096055af))
* **models:** remove FetchCamera, not used in RVC3 and its URDF is unshippable ([#545](https://github.com/petercorke/robotics-toolbox-python/issues/545)) ([17c67d2](https://github.com/petercorke/robotics-toolbox-python/commit/17c67d25eefee0d671727c461625970c7cbc85cb))
* **models:** rename list()'s type= param to mtype=, matching stale docs ([#542](https://github.com/petercorke/robotics-toolbox-python/issues/542)) ([c1402f6](https://github.com/petercorke/robotics-toolbox-python/commit/c1402f6ca62d87c6fed8d4eeb96ea5872a41dacb))
* **models:** validate addconfiguration() input, fix 5 broken named configs ([19dae9a](https://github.com/petercorke/robotics-toolbox-python/commit/19dae9a26fc9f11237d68fb0a03e824a02880837))
* **models:** validate addconfiguration() input, fix broken named configs ([756e101](https://github.com/petercorke/robotics-toolbox-python/commit/756e101611b9e9fe83194e40642dde706eca54fa))
* package eigdemo/tripleangledemo/twistdemo entry points correctly ([#600](https://github.com/petercorke/robotics-toolbox-python/issues/600)) ([b19eaa6](https://github.com/petercorke/robotics-toolbox-python/commit/b19eaa6f9c3006f22e88f1828b1cca3fab2fa7e4))
* pin spatialgeometry/swift-sim to their real published versions ([7da1125](https://github.com/petercorke/robotics-toolbox-python/commit/7da11251a27cdf8fe7944b351a29a6a2cc9e95cf))
* remove vendored spatialgeometry, restore real dependency ([bedf504](https://github.com/petercorke/robotics-toolbox-python/commit/bedf504da18a57d0e4a7c19fcee054faa8013064))
* resolve all 9 Sphinx build warnings ([#610](https://github.com/petercorke/robotics-toolbox-python/issues/610)) ([0437ae6](https://github.com/petercorke/robotics-toolbox-python/commit/0437ae66a4df0dc5e95bd4ac04f91bfaec89d493))
* Robot._to_dict() uses set_alpha(), deprecated by spatialgeometry ([3f73c26](https://github.com/petercorke/robotics-toolbox-python/commit/3f73c26e7596a7ad1317e3eecc78c1e0cb7a47cb))
* **rtb-data:** add missing rtb-data package config and data files ([aaff234](https://github.com/petercorke/robotics-toolbox-python/commit/aaff234fa8f16a459f6d1e78c74678d963fab847))
* rtbtool's %precision line spuriously echoes its own return value ([#592](https://github.com/petercorke/robotics-toolbox-python/issues/592)) ([40198dc](https://github.com/petercorke/robotics-toolbox-python/commit/40198dc7527fe6288875e23329245e59e8d944c7))
* **rtbtool:** don't let spatialgeometry's Path shadow pathlib.Path ([f061415](https://github.com/petercorke/robotics-toolbox-python/commit/f061415bf9195f35d74d7c19575af2c9656a94e3))
* **rtbtool:** fix arg-parsing bugs and converge CLI with mvtbtool ([#557](https://github.com/petercorke/robotics-toolbox-python/issues/557)) ([b111856](https://github.com/petercorke/robotics-toolbox-python/commit/b1118564076c5a6becd9b68cae62217b2f6b4355))
* ShapePlot uses SG Shape.base (removed) instead of .T for pose ([#595](https://github.com/petercorke/robotics-toolbox-python/issues/595)) ([638d673](https://github.com/petercorke/robotics-toolbox-python/commit/638d67359d1bf945d57f37924661371fb16e306c))
* **swift:** use label kwarg/property, not deprecated desc ([17bcfad](https://github.com/petercorke/robotics-toolbox-python/commit/17bcfadd6cd9c806486a3e8c79a03cfae661208d))
* **swift:** use Slider/Label/Button's label kwarg, not deprecated desc ([bc9ded4](https://github.com/petercorke/robotics-toolbox-python/commit/bc9ded4bd4cd1227592d802c63ffd0cbf0950a16))
* **test:** give test_isspherical its own link instances per robot ([8dfd09a](https://github.com/petercorke/robotics-toolbox-python/commit/8dfd09a36d64bcb362ce343cb0b0f0d53e3e0a5a))
* **tests:** fix Python 3.10-only mock.patch AttributeError in fknm fallback tests ([#539](https://github.com/petercorke/robotics-toolbox-python/issues/539)) ([38be2c4](https://github.com/petercorke/robotics-toolbox-python/commit/38be2c4761f0ccc5c6ae5d395a05bbb6566b40df))
* **test:** specify encoding=utf-8 when reading BaseRobot.py/pyproject.toml ([#548](https://github.com/petercorke/robotics-toolbox-python/issues/548)) ([ea4acff](https://github.com/petercorke/robotics-toolbox-python/commit/ea4acff35015534c1fea0fc45584ec198f6d0e79))
* **tests:** skip test_collision.py cases that require coal ([#538](https://github.com/petercorke/robotics-toolbox-python/issues/538)) ([fc51bc2](https://github.com/petercorke/robotics-toolbox-python/commit/fc51bc2e52f9f664faa079de3c980023bad2107f))
* track swift's SwiftElement.py -&gt; Elements.py rename ([eecc6d9](https://github.com/petercorke/robotics-toolbox-python/commit/eecc6d9ffabdb86e8e35c823f6dae6ea5cfd9e51))
* URDF parser ([39f14a0](https://github.com/petercorke/robotics-toolbox-python/commit/39f14a0469d8cb3140c677c76f436735e3d5fcc0))
* use spatialgeometry's correctly-spelled _propagate_scene_tree() ([d0fad96](https://github.com/petercorke/robotics-toolbox-python/commit/d0fad962b07c31a853fa8c90af0578a7376fc698))
* use spatialgeometry's new update() method, not the deprecated alias ([972ff0d](https://github.com/petercorke/robotics-toolbox-python/commit/972ff0d65bda1c63f00367dd062d82c761a117b8))
* **xacro:** use quoteattr for attribute serialization (Python 3.14) ([34bb4e2](https://github.com/petercorke/robotics-toolbox-python/commit/34bb4e2ffeadad7689754863433213830d86e36e)), closes [#511](https://github.com/petercorke/robotics-toolbox-python/issues/511)


### Performance Improvements

* **dynamics:** add symbolic-aware C dispatch check, safety net, batch trajectories into one C call ([c2bd405](https://github.com/petercorke/robotics-toolbox-python/commit/c2bd405a7b07b2118a6f6cacb70055d7ff61a8cc))


### Documentation

* add AGENTS.md stub linking to rvc-ecosystem conventions ([#591](https://github.com/petercorke/robotics-toolbox-python/issues/591)) ([870c840](https://github.com/petercorke/robotics-toolbox-python/commit/870c840363b9588df3b4735379a76cef84bd5a24))
* add desiderata (design principles) ([104671e](https://github.com/petercorke/robotics-toolbox-python/commit/104671e590ff0285b9161b16c5a4adf1815cabed))
* adopt rvc-ecosystem Sphinx footer and extension conventions ([#606](https://github.com/petercorke/robotics-toolbox-python/issues/606)) ([755614a](https://github.com/petercorke/robotics-toolbox-python/commit/755614a8229da5448482b1bd561545a265cc3ea0))
* describe robot_descriptions integration and catalog() ([#608](https://github.com/petercorke/robotics-toolbox-python/issues/608)) ([f759fa4](https://github.com/petercorke/robotics-toolbox-python/commit/f759fa4579ae8fe00ab62da15af614af9c3356a3))
* **desiderata:** move joint vel/accel limits under actuator params ([1325bb3](https://github.com/petercorke/robotics-toolbox-python/commit/1325bb321b1635508cfd579c413cd808c151369a))
* **dynamics:** housekeeping and write up the full investigation in rne.md ([c00af8c](https://github.com/petercorke/robotics-toolbox-python/commit/c00af8ccc05f49a0c2a1b9257b58665424112dbd))
* fix dead wiki/Common-Issues links, point at wiki/FAQ ([#609](https://github.com/petercorke/robotics-toolbox-python/issues/609)) ([fae8c27](https://github.com/petercorke/robotics-toolbox-python/commit/fae8c278c76fdee775e258e2666d798869e27d57))
* fix repeated typos in IK/DHRobot/RobotKinematics/mobile docstrings ([8780374](https://github.com/petercorke/robotics-toolbox-python/commit/87803745112b24a120427d3edc035beb10a7c7b6))
* **fix:** resolve stale jointset references and joints() forward-ref warning ([298f4c0](https://github.com/petercorke/robotics-toolbox-python/commit/298f4c017412524510bf38bb84249015c3a5a70a))
* **fix:** resolve stale jointset references and joints() forward-ref warning ([4531d93](https://github.com/petercorke/robotics-toolbox-python/commit/4531d93b80911e2283944fa1bbbd6d569486572c))
* **fix:** update stale roboticstoolbox.robot.ETS references to ets package ([2e9f2cd](https://github.com/petercorke/robotics-toolbox-python/commit/2e9f2cd8b698667d9825b9a7944f670ef00a9a1c))
* **intro:** restructure into arm/mobile robots, spatial math, history ([8d438d0](https://github.com/petercorke/robotics-toolbox-python/commit/8d438d07b73a829f6592017c7183c693ace51547))
* make sphinx invocation portable for CI ([5854c42](https://github.com/petercorke/robotics-toolbox-python/commit/5854c42e3c014495f4a5e6526c9edbd6cc693807))
* modernize issue/PR templates to match MVTB's community-health rollout ([#607](https://github.com/petercorke/robotics-toolbox-python/issues/607)) ([0b9f0c5](https://github.com/petercorke/robotics-toolbox-python/commit/0b9f0c50cb17c8ece6c7cccdbe8c9e8fcb2815e4))
* point graphics-block docstrings at |GraphicsBlockOptions| ([2c487c7](https://github.com/petercorke/robotics-toolbox-python/commit/2c487c74d93aa168f5841f41e16f461255aac314))
* README update ([e691441](https://github.com/petercorke/robotics-toolbox-python/commit/e691441e94e5e1b60870931eb943d415454d9d7a))
* **readme:** rework top section, references, TOC, and wasm wheel docs ([f169546](https://github.com/petercorke/robotics-toolbox-python/commit/f169546a21251939fc08b1309914df292a67c476))
* record 1.3.1 in release-please manifest and CHANGELOG ([#532](https://github.com/petercorke/robotics-toolbox-python/issues/532)) ([dfb38c2](https://github.com/petercorke/robotics-toolbox-python/commit/dfb38c21418c355a633f04e2c3ea9d907f1d7521))
* restructure intro.rst, add JupyterLite deployment, rework README ([53222a4](https://github.com/petercorke/robotics-toolbox-python/commit/53222a4709ffcd397a3cca25a36aff2a7c1d5a6c))
* **tech-debt:** consolidate rtb-data entries, document LBR/Valkyrie/Fetch findings ([#544](https://github.com/petercorke/robotics-toolbox-python/issues/544)) ([8ac586d](https://github.com/petercorke/robotics-toolbox-python/commit/8ac586d26af67e4f9dbb5bf2eda679b34b51b2f2))
* **tech-debt:** note rtb-data auto-publish idea ([c3cfcb2](https://github.com/petercorke/robotics-toolbox-python/commit/c3cfcb2fa15eab85016182157ae64298c718ff39))
* **tech-debt:** record deferred IKProtocol idea, ahead of the /ets split ([73fee37](https://github.com/petercorke/robotics-toolbox-python/commit/73fee374ea7657ae3973faebc7a056c23d0bf035))
* **tech-debt:** record fknm nanobind leak on background-thread creation ([c5b80b3](https://github.com/petercorke/robotics-toolbox-python/commit/c5b80b35f406978c5636b521ff9b6d36936bb613))
* **tech-debt:** record post-merge CI findings (fknm 3.10, collision skip gap, flaky IK) ([#535](https://github.com/petercorke/robotics-toolbox-python/issues/535)) ([174d9da](https://github.com/petercorke/robotics-toolbox-python/commit/174d9da597c089429fb4fe34f51b4d32bd55cb86))
* **tech-debt:** record vendored spatialgeometry vs external [@future](https://github.com/future) install ([0ed3f81](https://github.com/petercorke/robotics-toolbox-python/commit/0ed3f81d51b489feb8c219a7c7f0838c32b93ef7))
* **test:** clarify test_isspherical's fix is a workaround, not a rule ([6994fad](https://github.com/petercorke/robotics-toolbox-python/commit/6994fad65d9649079d5671399f4f175058cf2129))
* track docs/source/_templates/layout.html ([#603](https://github.com/petercorke/robotics-toolbox-python/issues/603)) ([c2be132](https://github.com/petercorke/robotics-toolbox-python/commit/c2be132085076faa096154aad8139cad9048e983))
* update Swift RRMC example to the AssemblyHandle API ([fe892e8](https://github.com/petercorke/robotics-toolbox-python/commit/fe892e879e7367fd51df0f5092c57c9dcde318cd))


### Build System

* **deps:** bump actions/checkout from 6 to 7 ([#522](https://github.com/petercorke/robotics-toolbox-python/issues/522)) ([65feed3](https://github.com/petercorke/robotics-toolbox-python/commit/65feed3f702a81371362b06ed98b76e3b236c73a))
* **deps:** bump actions/deploy-pages from 4 to 5 ([c384502](https://github.com/petercorke/robotics-toolbox-python/commit/c3845022afbb6678165af8f28ab0f4d5178fa232))
* **deps:** bump actions/download-artifact from 4 to 8 ([69be126](https://github.com/petercorke/robotics-toolbox-python/commit/69be126d3c0b2f24b8a7272d00299e35a3f8222f))
* **deps:** bump actions/setup-python from 6 to 7 ([2e31d0f](https://github.com/petercorke/robotics-toolbox-python/commit/2e31d0fcd946b5c86f7ff284e3922cbe92d23645))
* **deps:** bump actions/upload-pages-artifact from 3 to 5 ([#521](https://github.com/petercorke/robotics-toolbox-python/issues/521)) ([874e4a5](https://github.com/petercorke/robotics-toolbox-python/commit/874e4a5937a8edcaca81aa89a297cf1e116c7766))
* **deps:** bump amannn/action-semantic-pull-request from 5 to 6 ([#525](https://github.com/petercorke/robotics-toolbox-python/issues/525)) ([9acab8b](https://github.com/petercorke/robotics-toolbox-python/commit/9acab8b9d909e8d0aa3a0a5b3a3c49d472ed751e))
* **deps:** bump codecov/codecov-action from 6 to 7 ([33690a0](https://github.com/petercorke/robotics-toolbox-python/commit/33690a049dad24275a04c0e589985988a5c70c17))
* **deps:** bump googleapis/release-please-action from 4 to 5 ([#524](https://github.com/petercorke/robotics-toolbox-python/issues/524)) ([a5e59ed](https://github.com/petercorke/robotics-toolbox-python/commit/a5e59ed9a0a0369119676b54bc201af5bd187938))
* **deps:** bump pypa/cibuildwheel from 3.4.1 to 4.1.1 ([4a0b377](https://github.com/petercorke/robotics-toolbox-python/commit/4a0b377d1e045941316d278acdc2f55fae02d5ba))
* **deps:** bump softprops/action-gh-release from 2 to 3 ([#523](https://github.com/petercorke/robotics-toolbox-python/issues/523)) ([1e6183f](https://github.com/petercorke/robotics-toolbox-python/commit/1e6183f6d2291926dd7d1e205e7fc334aaece99b))
* exclude rtb-data subdir from sdist ([e45363d](https://github.com/petercorke/robotics-toolbox-python/commit/e45363d16458712321c630902ac736a2be283a42))
* remove rules for pushing to PyPI, now done by GH actions. ([569fedd](https://github.com/petercorke/robotics-toolbox-python/commit/569fedd1cd026f1a8ecd1122851a9e10d0c0632a))
* setup proper release process ([d0e0559](https://github.com/petercorke/robotics-toolbox-python/commit/d0e0559a4c59e15d7eb87d9d34cedb11a9eb3fb6))

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
