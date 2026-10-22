import matplotlib
import matplotlib.animation

matplotlib.use("Agg")

import numpy as np
import pytest

import roboticstoolbox as rtb


def _filter(verbose: bool = False) -> rtb.ParticleFilter:
    robot = rtb.Bicycle(workspace=10, dt=0.1)
    robot.control = rtb.RandomPath(workspace=10, seed=0)
    landmarks = rtb.LandmarkMap(5, workspace=10, seed=0)
    sensor = rtb.RangeBearingSensor(
        robot, landmarks, covar=np.diag([0.1, 0.01]), seed=0
    )
    return rtb.ParticleFilter(
        robot,
        sensor,
        R=np.diag([0.1, 0.1, 0.01]),
        L=np.diag([0.1, 0.1]),
        nparticles=50,
        verbose=verbose,
        seed=0,
    )


def test_map_is_the_map_of_the_sensor() -> None:
    pf = _filter()
    assert pf.map is pf.sensor.map


@pytest.mark.parametrize("verbose", [False, True])
def test_verbose_is_what_was_asked(verbose: bool) -> None:
    assert _filter(verbose=verbose).verbose is verbose


def test_run_keeps_a_history_of_named_steps() -> None:
    pf = _filter()
    pf.run(T=1)
    assert len(pf.history) == 10
    assert pf.history[0]._fields == ("t", "odo", "xest", "std", "weights")
    assert pf.get_xyt().shape == (10, 3)
    assert pf.get_std().shape == (10, 3)
    assert pf.get_t().shape == (10,)


@pytest.mark.skipif(
    not matplotlib.animation.FFMpegWriter.isAvailable(), reason="needs ffmpeg"
)
def test_mp4_is_written(tmp_path) -> None:
    out = tmp_path / "pf.mp4"
    _filter().run_animation(T=0.5, format="mp4", file=str(out))
    assert out.stat().st_size > 0


@pytest.mark.parametrize("format", ["gif", "mp4"])
def test_a_file_format_needs_a_file_name(format: str) -> None:
    with pytest.raises(ValueError, match="file name"):
        _filter().run_animation(T=0.5, format=format)


def test_gif_is_written(tmp_path) -> None:
    # used self.dt, which does not exist, so this raised AttributeError
    out = tmp_path / "pf.gif"
    _filter().run_animation(T=0.5, format="gif", file=str(out))
    assert out.stat().st_size > 0


def test_unknown_format_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown format"):
        _filter().run_animation(T=0.5, format="avi")


def test_x0_vector_starts_every_particle_there() -> None:
    pf = _filter()
    pf._init(x0=[1.0, 2.0, 0.5])
    assert pf.x.shape == (50, 3)
    assert np.all(pf.x == [1.0, 2.0, 0.5])


def test_x0_matrix_gives_each_particle_its_own_state() -> None:
    pf = _filter()
    x0 = np.random.default_rng(1).uniform(-5, 5, size=(50, 3))
    keep = x0.copy()
    pf._init(x0=x0)
    assert np.array_equal(pf.x, keep)
    # the filter updates its particles in place, the caller's array is not them
    pf.x[:, 0] += 1
    assert np.array_equal(x0, keep)


def test_x0_of_the_constructor_is_the_default() -> None:
    robot = rtb.Bicycle(workspace=10)
    landmarks = rtb.LandmarkMap(5, workspace=10, seed=0)
    sensor = rtb.RangeBearingSensor(robot, landmarks, seed=0)
    pf = rtb.ParticleFilter(
        robot, sensor, np.eye(3) * 0.01, np.eye(2), nparticles=20, x0=[3, 4, 0]
    )
    pf._init()
    assert np.all(pf.x == [3, 4, 0])


@pytest.mark.parametrize("x0", [[1, 2], np.zeros((49, 3)), np.zeros((50, 2))])
def test_x0_of_the_wrong_shape_is_an_error(x0) -> None:
    with pytest.raises(ValueError, match="x0 must have shape"):
        _filter()._init(x0=x0)


def test_run_from_a_given_x0() -> None:
    pf = _filter()
    pf.run(T=0.5, x0=[0, 0, 0])
    assert len(pf.history) == 5


def test_run_animation_starts_from_x0() -> None:
    pf = _filter()
    anim = pf.run_animation(T=0.2, x0=[1.0, 2.0, 0.0])
    anim._init_draw()
    assert np.all(pf.x == [1.0, 2.0, 0.0])
