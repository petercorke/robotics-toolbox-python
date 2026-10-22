import matplotlib

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
