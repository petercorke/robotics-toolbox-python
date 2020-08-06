#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import spatialmath as sm
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def text_trans(text):
    T = panda.fkine()
    t = np.round(T.t, 3)
    r = np.round(T.rpy(), 3)
    text[0].set_text("x: {0}".format(t[0]))
    text[1].set_text("y: {0}".format(t[1]))
    text[2].set_text("z: {0}".format(t[2]))
    text[3].set_text("r: {0}".format(r[0]))
    text[4].set_text("p: {0}".format(r[1]))
    text[5].set_text("y: {0}".format(r[2]))


def update(val, text):
    for i in range(panda.n):
        panda.q[i] = sjoint[i].val * np.pi/180

    text_trans(text)
    env.step(0)


panda = rp.PandaMDH()
panda.q = panda.qr

env = rp.PyPlot()
env.launch('Teach ' + panda.name)
env.add(panda, readonly=True)

ax = env.ax
fig = env.fig

fig.subplots_adjust(left=0.25)
text = []

x1 = 0.04
x2 = 0.22
yh = 0.04
ym = 0.5 - (panda.n * yh) / 2 + 0.17/2

axjoint = []
sjoint = []

qlim = np.copy(panda.qlim) * 180/np.pi

if not np.all(qlim != 0):
    qlim[0, :] = -180
    qlim[1, :] = 180

# Set the pose text
T = panda.fkine()
t = np.round(T.t, 3)
r = np.round(T.rpy(unit='deg'), 3)

fig.text(
    0.02,  1 - ym + 0.25, "End-effector Pose",
    fontsize=9, weight="bold", color="#4f4f4f")
text.append(fig.text(
    0.03, 1 - ym + 0.20, "x: {0}".format(t[0]), fontsize=9, color="#2b2b2b"))
text.append(fig.text(
    0.03, 1 - ym + 0.16, "y: {0}".format(t[1]), fontsize=9, color="#2b2b2b"))
text.append(fig.text(
    0.03, 1 - ym + 0.12, "z: {0}".format(t[2]), fontsize=9, color="#2b2b2b"))
text.append(fig.text(
    0.15, 1 - ym + 0.20, "r: {0}".format(r[0]), fontsize=9, color="#2b2b2b"))
text.append(fig.text(
    0.15, 1 - ym + 0.16, "p: {0}".format(r[1]), fontsize=9, color="#2b2b2b"))
text.append(fig.text(
    0.15, 1 - ym + 0.12, "y: {0}".format(r[2]), fontsize=9, color="#2b2b2b"))
fig.text(
    0.02,  1 - ym + 0.06, "Joint angles",
    fontsize=9, weight="bold", color="#4f4f4f")

for i in range(panda.n):
    ymin = (1 - ym) - i * yh
    axjoint.append(plt.axes([x1, ymin, x2, 0.03], facecolor='#dbdbdb'))

    sjoint.append(
        Slider(
            axjoint[i], 'q' + str(i),
            qlim[0, i], qlim[1, i], panda.q[i] * 180/np.pi))

    sjoint[i].on_changed(lambda x: update(x, text))

env.hold()
