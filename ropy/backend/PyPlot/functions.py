#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import time
import ropy as rp
import numpy as np
from spatialmath.base.argcheck import getvector, verifymatrix
from ropy.backend.PyPlot.EllipsePlot import EllipsePlot
from matplotlib.widgets import Slider


def _plot(
        robot, block, q, dt, limits=None,
        vellipse=False, fellipse=False,
        jointaxes=True, eeframe=True, shadow=True, name=True):

    # Make an empty 3D figure
    env = rp.backend.PyPlot()

    trajn = 1

    if q is None:
        q = robot.q

    try:
        q = getvector(q, robot.n, 'col')
        robot.q = q
    except ValueError:
        trajn = q.shape[1]
        verifymatrix(q, (robot.n, trajn))

    # Add the robot to the figure in readonly mode
    if trajn == 1:
        env.launch(robot.name + ' Plot', limits)
    else:
        env.launch(robot.name + ' Trajectory Plot', limits)

    env.add(
        robot, readonly=True,
        jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

    if vellipse:
        vell = robot.vellipse(centre='ee')
        env.add(vell)

    if fellipse:
        fell = robot.fellipse(centre='ee')
        env.add(fell)

    if trajn != 1:
        for i in range(trajn):
            robot.q = q[:, i]
            env.step()
            time.sleep(dt/1000)

    # Keep the plot open
    if block:           # pragma: no cover
        env.hold()

    return env


def _plot2(
        robot, block, q, dt, limits=None,
        vellipse=False, fellipse=False,
        eeframe=True, name=True):

    # Make an empty 2D figure
    env = rp.backend.PyPlot2()

    trajn = 1

    if q is None:
        q = robot.q

    try:
        q = getvector(q, robot.n, 'col')
        robot.q = q
    except ValueError:
        trajn = q.shape[1]
        verifymatrix(q, (robot.n, trajn))

    # Add the robot to the figure in readonly mode
    if trajn == 1:
        env.launch(robot.name + ' Plot', limits)
    else:
        env.launch(robot.name + ' Trajectory Plot', limits)

    env.add(
        robot, readonly=True,
        eeframe=eeframe, name=name)

    if vellipse:
        vell = robot.vellipse(centre='ee')
        env.add(vell)

    if fellipse:
        fell = robot.fellipse(centre='ee')
        env.add(fell)

    if trajn != 1:
        for i in range(trajn):
            robot.q = q[:, i]
            env.step()
            time.sleep(dt/1000)

    # Keep the plot open
    if block:           # pragma: no cover
        env.hold()

    return env


def _teach(
        robot, block, order='xyz', limits=None,
        jointaxes=True, eeframe=True, shadow=True, name=True):

    # Add text to the plots
    def text_trans(text):  # pragma: no cover
        T = robot.fkine()
        t = np.round(T.t, 3)
        r = np.round(T.rpy(), 3)
        text[0].set_text("x: {0}".format(t[0]))
        text[1].set_text("y: {0}".format(t[1]))
        text[2].set_text("z: {0}".format(t[2]))
        text[3].set_text("r: {0}".format(r[0]))
        text[4].set_text("p: {0}".format(r[1]))
        text[5].set_text("y: {0}".format(r[2]))

    # Update the robot state in mpl and the text
    def update(val, text):  # pragma: no cover
        for i in range(robot.n):
            robot.q[i] = sjoint[i].val * np.pi/180

        text_trans(text)

        # Step the environment
        env.step(0)

    # Make an empty 3D figure
    env = rp.backend.PyPlot()

    # Add the robot to the figure in readonly mode
    env.launch('Teach ' + robot.name, limits=limits)
    env.add(
        robot, readonly=True,
        jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

    fig = env.fig

    fig.subplots_adjust(left=0.25)
    text = []

    x1 = 0.04
    x2 = 0.22
    yh = 0.04
    ym = 0.5 - (robot.n * yh) / 2 + 0.17/2

    axjoint = []
    sjoint = []

    qlim = np.copy(robot.qlim) * 180/np.pi

    if np.all(qlim == 0):
        qlim[0, :] = -180
        qlim[1, :] = 180

    # Set the pose text
    T = robot.fkine()
    t = np.round(T.t, 3)
    r = np.round(T.rpy(), 3)

    fig.text(
        0.02,  1 - ym + 0.25, "End-effector Pose",
        fontsize=9, weight="bold", color="#4f4f4f")
    text.append(fig.text(
        0.03, 1 - ym + 0.20, "x: {0}".format(t[0]),
        fontsize=9, color="#2b2b2b"))
    text.append(fig.text(
        0.03, 1 - ym + 0.16, "y: {0}".format(t[1]),
        fontsize=9, color="#2b2b2b"))
    text.append(fig.text(
        0.03, 1 - ym + 0.12, "z: {0}".format(t[2]),
        fontsize=9, color="#2b2b2b"))
    text.append(fig.text(
        0.15, 1 - ym + 0.20, "r: {0}".format(r[0]),
        fontsize=9, color="#2b2b2b"))
    text.append(fig.text(
        0.15, 1 - ym + 0.16, "p: {0}".format(r[1]),
        fontsize=9, color="#2b2b2b"))
    text.append(fig.text(
        0.15, 1 - ym + 0.12, "y: {0}".format(r[2]),
        fontsize=9, color="#2b2b2b"))
    fig.text(
        0.02,  1 - ym + 0.06, "Joint angles",
        fontsize=9, weight="bold", color="#4f4f4f")

    for i in range(robot.n):
        ymin = (1 - ym) - i * yh
        axjoint.append(fig.add_axes([x1, ymin, x2, 0.03], facecolor='#dbdbdb'))

        sjoint.append(
            Slider(
                axjoint[i], 'q' + str(i),
                qlim[0, i], qlim[1, i], robot.q[i] * 180/np.pi))

        sjoint[i].on_changed(lambda x: update(x, text))

    # Keep the plot open
    if block:           # pragma: no cover
        env.hold()

    return env


def _teach2(
        robot, block, order='xyz', limits=None,
        eeframe=True, name=True):

    # Add text to the plots
    def text_trans(text):  # pragma: no cover
        T = robot.fkine()
        t = np.round(T.t, 3)
        r = np.round(T.rpy(), 3)
        text[0].set_text("x: {0}".format(t[0]))
        text[1].set_text("y: {0}".format(t[1]))
        text[2].set_text("yaw: {0}".format(r[2]))

    # Update the robot state in mpl and the text
    def update(val, text):  # pragma: no cover
        for i in range(robot.n):
            robot.q[i] = sjoint[i].val * np.pi/180

        text_trans(text)

        # Step the environment
        env.step(0)

    # Make an empty 3D figure
    env = rp.backend.PyPlot2()

    # Add the robot to the figure in readonly mode
    env.launch('Teach ' + robot.name, limits=limits)
    env.add(
        robot, readonly=True,
        eeframe=eeframe, name=name)

    fig = env.fig

    fig.subplots_adjust(left=0.38)
    text = []

    x1 = 0.04
    x2 = 0.22
    yh = 0.04
    ym = 0.5 - (robot.n * yh) / 2 + 0.17/2

    axjoint = []
    sjoint = []

    qlim = np.copy(robot.qlim) * 180/np.pi

    if np.all(qlim == 0):
        qlim[0, :] = -180
        qlim[1, :] = 180

    # Set the pose text
    T = robot.fkine()
    t = np.round(T.t, 3)
    r = np.round(T.rpy(), 3)

    fig.text(
        0.02,  1 - ym + 0.25, "End-effector Pose",
        fontsize=9, weight="bold", color="#4f4f4f")
    text.append(fig.text(
        0.03, 1 - ym + 0.20, "x: {0}".format(t[0]),
        fontsize=9, color="#2b2b2b"))
    text.append(fig.text(
        0.03, 1 - ym + 0.16, "y: {0}".format(t[1]),
        fontsize=9, color="#2b2b2b"))
    text.append(fig.text(
        0.15, 1 - ym + 0.20, "yaw: {0}".format(r[0]),
        fontsize=9, color="#2b2b2b"))
    fig.text(
        0.02,  1 - ym + 0.06, "Joint angles",
        fontsize=9, weight="bold", color="#4f4f4f")

    for i in range(robot.n):
        ymin = (1 - ym) - i * yh
        axjoint.append(fig.add_axes([x1, ymin, x2, 0.03], facecolor='#dbdbdb'))

        sjoint.append(
            Slider(
                axjoint[i], 'q' + str(i),
                qlim[0, i], qlim[1, i], robot.q[i] * 180/np.pi))

        sjoint[i].on_changed(lambda x: update(x, text))

    # Keep the plot open
    if block:           # pragma: no cover
        env.hold()

    return env


def _fellipse(robot, q=None, opt='trans', centre=[0, 0, 0]):

    ell = EllipsePlot(robot, 'f', opt, centre=centre)
    return ell


def _vellipse(robot, q=None, opt='trans', centre=[0, 0, 0]):

    ell = EllipsePlot(robot, 'v', opt, centre=centre)
    return ell


def _plot_ellipse(
        ellipse, block=True, limits=None,
        jointaxes=True, eeframe=True, shadow=True, name=True):

    if not isinstance(ellipse, EllipsePlot):
        raise TypeError(
            'ellipse must be of type ropy.backend.PyPlot.EllipsePlot')

    env = rp.backend.PyPlot()

    # Add the robot to the figure in readonly mode
    env.launch(ellipse.robot.name + ' ' + ellipse.name, limits=limits)

    env.add(
        ellipse,
        jointaxes=jointaxes, eeframe=eeframe, shadow=shadow, name=name)

    # Keep the plot open
    if block:           # pragma: no cover
        env.hold()

    return env
