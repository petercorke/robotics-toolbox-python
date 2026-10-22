"""
The ``view`` option: start a 3D plot in an axis-aligned view of a plane.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import numpy.testing as nt  # noqa: E402
import pytest  # noqa: E402
from mpl_toolkits.mplot3d import proj3d  # noqa: E402

import roboticstoolbox as rtb  # noqa: E402
from roboticstoolbox import ET2  # noqa: E402
from roboticstoolbox.backends.PyPlot import PyPlot  # noqa: E402
from roboticstoolbox.tools.plot import plane_view, set_plane_view  # noqa: E402

pytestmark = pytest.mark.timeout(30)

#: (view, elev, azim)
VIEWS = [("xy", 90, -90), ("xz", 0, -90), ("yz", 0, 0)]


@pytest.fixture(autouse=True)
def close_figures():
    plt.close("all")
    yield
    plt.close("all")


@pytest.fixture
def puma():
    return rtb.models.DH.Puma560()


def axes3d():
    fig = plt.figure()
    return fig.add_subplot(111, projection="3d", proj_type="ortho")


def screen_direction(ax, vector):
    """Where a unit vector along a world axis points on the screen, from the origin."""
    ax.figure.canvas.draw()
    proj = ax.get_proj()
    x0, y0, _ = proj3d.proj_transform(0, 0, 0, proj)
    x1, y1, _ = proj3d.proj_transform(*vector, proj)
    return np.array([x1 - x0, y1 - y0])


class TestPlaneView:
    @pytest.mark.parametrize("name, elev, azim", VIEWS)
    def test_angles(self, name, elev, azim):
        assert plane_view(name) == {"elev": elev, "azim": azim}

    def test_any_case(self):
        assert plane_view("XZ") == plane_view("xz")
        assert plane_view("Yz") == plane_view("yz")

    def test_the_result_can_be_changed_without_changing_the_next(self):
        plane_view("xy")["elev"] = 0
        assert plane_view("xy")["elev"] == 90

    @pytest.mark.parametrize("bad", ["zz", "xyz", "x", "", "top", " xy", 3, None])
    def test_unknown(self, bad):
        with pytest.raises(ValueError, match="unknown view.*xy, xz, yz"):
            plane_view(bad)

    @pytest.mark.parametrize(
        "name, right, up",
        [
            ("xy", (1, 0, 0), (0, 1, 0)),
            ("xz", (1, 0, 0), (0, 0, 1)),
            ("yz", (0, 1, 0), (0, 0, 1)),
        ],
    )
    def test_the_first_axis_is_to_the_right_and_the_second_up(self, name, right, up):
        ax = axes3d()
        set_plane_view(ax, name)
        r, u = screen_direction(ax, right), screen_direction(ax, up)
        assert r[0] > 0 and abs(r[1]) < 1e-9  # to the right, level
        assert u[1] > 0 and abs(u[0]) < 1e-9  # up, upright

    @pytest.mark.parametrize("name, elev, azim", VIEWS)
    def test_the_third_axis_is_end_on(self, name, elev, azim):
        ax = axes3d()
        set_plane_view(ax, name)
        third = {"xy": (0, 0, 1), "xz": (0, 1, 0), "yz": (1, 0, 0)}[name]
        nt.assert_allclose(screen_direction(ax, third), [0, 0], atol=1e-9)

    def test_the_view_can_be_changed_afterwards(self):
        ax = axes3d()
        set_plane_view(ax, "xy")
        ax.view_init(elev=20, azim=30)  # what dragging the plot does
        assert (ax.elev, ax.azim) == (20, 30)


class TestLaunch:
    def test_default_view_is_unchanged(self):
        env = PyPlot()
        env.launch()
        default = axes3d()
        assert (env.ax.elev, env.ax.azim) == (default.elev, default.azim)

    @pytest.mark.parametrize("name, elev, azim", VIEWS)
    def test_view(self, name, elev, azim):
        env = PyPlot()
        env.launch(view=name)
        assert (env.ax.elev, env.ax.azim) == (elev, azim)

    def test_bad_view_leaves_no_figure(self):
        env = PyPlot()
        with pytest.raises(ValueError, match="unknown view"):
            env.launch(view="top")
        assert plt.get_fignums() == []

    def test_on_given_axes(self):
        ax = axes3d()
        env = PyPlot()
        env.launch(ax=ax, view="yz")
        assert env.ax is ax
        assert (ax.elev, ax.azim) == (0, 0)


class TestPlot:
    @pytest.mark.parametrize("name, elev, azim", VIEWS)
    def test_view(self, puma, name, elev, azim):
        env = puma.plot(puma.qr, backend="pyplot", view=name, block=False)
        assert (env.ax.elev, env.ax.azim) == (elev, azim)

    def test_the_robot_is_drawn_in_that_plane(self, puma):
        env = puma.plot(puma.qr, backend="pyplot", view="xz", block=False)
        tip = np.array(env.robots[0].links[0].get_data_3d()).T[-1]
        # the screen position of the tip follows x to the right and z up
        env.fig.canvas.draw()
        x, y, _ = proj3d.proj_transform(*tip, env.ax.get_proj())
        x0, y0, _ = proj3d.proj_transform(0, 0, 0, env.ax.get_proj())
        assert (x - x0) * tip[0] > 0 or abs(tip[0]) < 1e-6
        assert (y - y0) * tip[2] > 0 or abs(tip[2]) < 1e-6

    def test_a_trajectory_does_not_reset_the_view(self, puma):
        qt = np.array([puma.qr + (puma.qn - puma.qr) * s for s in np.linspace(0, 1, 4)])
        env = puma.plot(qt, backend="pyplot", view="xy", block=False)
        assert (env.ax.elev, env.ax.azim) == (90, -90)

    def test_the_user_can_rotate_it_afterwards(self, puma):
        env = puma.plot(puma.qr, backend="pyplot", view="xy", block=False)
        env.ax.view_init(elev=10, azim=20)
        env.step(0.001)
        assert (env.ax.elev, env.ax.azim) == (10, 20)

    def test_with_env_the_view_of_the_scene_changes(self, puma):
        env = puma.plot(puma.qr, backend="pyplot", view="xy", block=False)
        env2 = puma.plot(puma.qn, backend="pyplot", env=env, view="yz", block=False)
        assert env2 is env
        assert (env.ax.elev, env.ax.azim) == (0, 0)
        assert len(env.robots) == 2

    def test_with_env_and_no_view_it_is_left_alone(self, puma):
        env = puma.plot(puma.qr, backend="pyplot", view="xz", block=False)
        puma.plot(puma.qn, backend="pyplot", env=env, block=False)
        assert (env.ax.elev, env.ax.azim) == (0, -90)

    def test_on_given_axes(self, puma):
        ax = axes3d()
        env = puma.plot(puma.qr, backend="pyplot", ax=ax, view="xy", block=False)
        assert env.ax is ax
        assert (ax.elev, ax.azim) == (90, -90)

    def test_bad_view_is_an_error_before_anything_is_drawn(self, puma):
        with pytest.raises(ValueError, match="unknown view"):
            puma.plot(puma.qr, backend="pyplot", view="up", block=False)
        assert plt.get_fignums() == []

    def test_a_planar_backend_cannot(self):
        planar = rtb.Robot2(ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1), name="planar")
        with pytest.raises(ValueError, match="cannot set a view"):
            planar.plot([0.1, 0.2], backend="pyplot2", view="xy", block=False)
        assert plt.get_fignums() == []


class TestTeach:
    @pytest.mark.parametrize("name, elev, azim", VIEWS)
    def test_view(self, puma, name, elev, azim):
        env = puma.teach(puma.qr, backend="pyplot", view=name, block=False)
        assert (env.ax.elev, env.ax.azim) == (elev, azim)

    def test_default_view_is_unchanged(self, puma):
        env = puma.teach(puma.qr, backend="pyplot", block=False)
        default = axes3d()
        assert (env.ax.elev, env.ax.azim) == (default.elev, default.azim)

    def test_bad_view_leaves_no_figure(self, puma):
        with pytest.raises(ValueError, match="unknown view"):
            puma.teach(puma.qr, backend="pyplot", view="up", block=False)
        assert plt.get_fignums() == []

    def test_a_planar_backend_cannot(self):
        planar = rtb.Robot2(ET2.R() * ET2.tx(1) * ET2.R() * ET2.tx(1), name="planar")
        with pytest.raises(ValueError, match="cannot set a view"):
            planar.teach([0.1, 0.2], backend="pyplot2", view="xy", block=False)
