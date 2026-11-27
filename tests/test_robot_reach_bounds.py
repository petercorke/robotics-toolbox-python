"""Reach bounds include compound transforms and signed joint travel."""

import io

import matplotlib
import numpy as np
import pytest
from spatialmath import SE2, SE3

import roboticstoolbox as rtb
from roboticstoolbox.models.URDF.URDFRobot import URDF_file

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_compound_translation_contributes_to_reach(planar, compiled):
    if planar:
        ets = rtb.ET2.SE2(SE2(3, 4, 0.7)) * rtb.ET2.R()
        robot_type = rtb.Robot2
    else:
        ets = rtb.ET.SE3(SE3(3, 4, 0) * SE3.RPY(0.2, -0.4, 0.7)) * rtb.ET.Rz()
        robot_type = rtb.Robot
    robot = robot_type(ets.compile() if compiled else ets)

    assert robot.reach == pytest.approx(5)
    for angle in [-2, 0, 1.3]:
        assert np.linalg.norm(robot.fkine([angle]).t) <= robot.reach + 1e-12


@pytest.mark.parametrize("planar", [False, True])
@pytest.mark.parametrize("limits", [(-2, -1), (-3, 1), (1, 2)])
@pytest.mark.parametrize("offset", [-4, 0, 4])
@pytest.mark.parametrize("flip", [False, True])
def test_prismatic_reach_bounds_signed_travel(planar, limits, offset, flip):
    et_type, ets_type, robot_type = (
        (rtb.ET2, rtb.ETS2, rtb.Robot2) if planar else (rtb.ET, rtb.ETS, rtb.Robot)
    )
    joint = et_type.tx(qlim=limits, flip=flip)
    joint.offset = offset
    robot = robot_type(ets_type(joint))
    travel = [(offset - q if flip else offset + q) for q in limits]

    assert robot.reach == pytest.approx(max(abs(t) for t in travel))
    for q in np.linspace(*limits, 9):
        assert np.linalg.norm(robot.fkine([q]).t) <= robot.reach + 1e-12


@pytest.mark.parametrize("link_type", [rtb.PrismaticDH, rtb.PrismaticMDH])
@pytest.mark.parametrize("limits", [(-2, -1), (-3, 1), (1, 2)])
@pytest.mark.parametrize("offset", [-4, 0, 4])
@pytest.mark.parametrize("flip", [False, True])
def test_dh_prismatic_reach_bounds_signed_travel(link_type, limits, offset, flip):
    link = link_type(
        a=-0.5, alpha=0.6, theta=0.3, qlim=limits, offset=offset, flip=flip
    )
    robot = rtb.DHRobot([link])
    travel = [(offset - q if flip else offset + q) for q in limits]

    assert robot.reach == pytest.approx(0.5 + max(abs(t) for t in travel))
    for q in np.linspace(*limits, 9):
        assert np.linalg.norm(robot.fkine([q]).t) <= robot.reach + 1e-12


@pytest.mark.parametrize("link_type", [rtb.RevoluteDH, rtb.RevoluteMDH])
def test_dh_negative_constant_d_is_a_length(link_type):
    robot = rtb.DHRobot([link_type(a=-1, d=-2, alpha=0.6)])

    assert robot.reach == 3
    for q in [-2, 0, 1.3]:
        assert np.linalg.norm(robot.fkine([q]).t) <= robot.reach


@pytest.mark.parametrize("planar", [False, True])
def test_reach_uses_longest_branch(planar):
    if planar:
        root = rtb.Link2(rtb.ET2.SE2(SE2(3, 4)))
        first = rtb.Link2(rtb.ET2.tx(qlim=[-3, -1]), parent=root)
        second = rtb.Link2(rtb.ET2.SE2(SE2(4, 0)) * rtb.ET2.R(), parent=root)
        robot = rtb.Robot2([root, first, second])
    else:
        root = rtb.Link(rtb.ET.SE3(SE3(3, 4, 0)))
        first = rtb.Link(rtb.ET.tx(qlim=[-3, -1]), parent=root)
        second = rtb.Link(rtb.ET.SE3(SE3(4, 0, 0)) * rtb.ET.Rz(), parent=root)
        robot = rtb.Robot([root, first, second])

    assert robot.reach == 9
    frames = robot.fkine_all([-3, 0.7])
    for link in robot.ee_links:
        assert np.linalg.norm(frames[link.number].t) <= robot.reach


def test_urdf_joint_origins_contribute_to_reach():
    # Use the real URDF loader without network models or mesh downloads.
    links, name, _ = URDF_file(
        io.StringIO(
            """<robot name="reach_test">
              <link name="base"/><link name="arm"/><link name="tip"/>
              <joint name="hinge" type="revolute">
                <parent link="base"/><child link="arm"/>
                <origin xyz="3 4 0" rpy="0.2 -0.4 0.7"/>
                <axis xyz="0 0 1"/>
                <limit lower="-2" upper="2" effort="1" velocity="1"/>
              </joint>
              <joint name="tip_mount" type="fixed">
                <parent link="arm"/><child link="tip"/>
                <origin xyz="0 0 -2" rpy="0.1 0.2 0.3"/>
              </joint>
            </robot>"""
        )
    )
    robot = rtb.Robot(links, name=name)

    assert robot.reach == pytest.approx(7)
    for q in np.linspace(-2, 2, 9):
        assert np.linalg.norm(robot.fkine([q]).t) <= robot.reach + 1e-12


@pytest.mark.parametrize("compound", [False, True])
def test_planar_plot_limits_include_robot(compound):
    from roboticstoolbox.backends.PyPlot import PyPlot2

    if compound:
        robot = rtb.Robot2(rtb.ET2.SE2(SE2(3, 4)) * rtb.ET2.R())
        q = [0.5]
    else:
        robot = rtb.Robot2(rtb.ETS2(rtb.ET2.tx(qlim=[-2, -1])))
        q = [-2]
    env = PyPlot2()
    try:
        env.launch()
        env.add(robot, q=q, eeframe=False, readonly=True)
        env.step()
        endpoint = robot.fkine(q).t
        for coordinate, (lower, upper) in zip(
            endpoint, [env.ax.get_xlim(), env.ax.get_ylim()]
        ):
            assert lower < coordinate < upper
    finally:
        env.close()
        plt.close("all")


def test_existing_elementary_length_sum_and_cache():
    robot = rtb.Robot(rtb.ET.tz(2) * rtb.ET.Rz() * rtb.ET.tx(-3))
    assert robot.reach == 5
    assert robot.reach == 5
    robot.base = SE3(100, 200, 300)
    robot.tool = SE3(20, 30, 40)
    assert robot.reach == 5


@pytest.mark.parametrize("planar", [False, True])
def test_unbounded_prismatic_includes_fixed_offset(planar):
    et_type, ets_type, robot_type = (
        (rtb.ET2, rtb.ETS2, rtb.Robot2) if planar else (rtb.ET, rtb.ETS, rtb.Robot)
    )
    joint = et_type.tx()
    joint.offset = -2
    assert robot_type(ets_type(joint)).reach == 2


@pytest.mark.parametrize("link_type", [rtb.PrismaticDH, rtb.PrismaticMDH])
def test_unbounded_dh_prismatic_includes_fixed_offset(link_type):
    robot = rtb.DHRobot([link_type(a=-0.5, offset=-2)])
    assert robot.reach == 2.5
