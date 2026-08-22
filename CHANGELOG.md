# Changelog

## [1.4.0](https://github.com/petercorke/robotics-toolbox-python/compare/v1.3.1...v1.4.0) (2026-08-22)

**Highlights**: using significantly reworked versions of [Swift](https://github.com/jhavl/swift) (v2.0.0) and [Spatial Geometry](https://github.com/jhavl/spatialgeometry) (v1.3.0) -- lots of bug fixes, new features and documentation.  Collision detection changed from PyBullet (no longer well maintained to [Coal](https://github.com/coal-library/coal).  Integrates with [Robot Descriptions](https://github.com/robot-descriptions) for access to over 150 URDF models, loaded dynamically and cached locally -- this means that RTB can now ship with a very minimum set of mesh models.  Significant work on RNE models (DH Python, DH C, ETS Featherstone) all consistent and checked against analytic model.  Additional work on docs, build system, and testing.  

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
