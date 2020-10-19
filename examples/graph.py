#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
import numpy as np
import time
import qpsolvers as qp
import pybullet as p
import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('ggplot')
matplotlib.rcParams['font.size'] = 4.5
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['xtick.major.size'] = 1.5
matplotlib.rcParams['ytick.major.size'] = 1.5
matplotlib.rcParams['axes.labelpad'] = 1
plt.rc('grid', linestyle="-", color='#dbdbdb')

obj1 = pickle.load(open("neo1.p", "rb"))
obj2 = pickle.load(open("neo2.p", "rb"))
obj3 = pickle.load(open("neo3.p", "rb"))


# ----------------------------------------------------------------------------
fig1, ax1 = plt.subplots()
fig1.set_size_inches(2.5, 1.5)

ax1.set(xlabel='Time (s)', ylabel='Distance (m)')
ax1.grid()
plt.grid(True)
ax1.set_xlim(xmin=0, xmax=14.5)
ax1.set_ylim(ymin=0, ymax=0.6)
plt.subplots_adjust(left=0.13, bottom=0.18, top=0.95, right=1)

pld0 = ax1.plot(
    obj1['s0'][0], obj1['s0'][1], label='Distance to Obstacle 1')

dummy = ax1.plot(
    0, 0)

pld2 = ax1.plot(
    obj1['s2'][0], obj1['s2'][1], label='Distance to Goal')

# plm = ax1.plot(
#     obj['m'][0], obj['m'][1], label='Manipuability')

ax1.legend()
ax1.legend(loc="upper right")

fig1.savefig('neo1.eps')



# ----------------------------------------------------------------------------
fig2, ax2 = plt.subplots()
fig2.set_size_inches(2.5, 1.5)

ax2.set(xlabel='Time (s)', ylabel='Distance (m)')
ax2.grid()
plt.grid(True)
ax2.set_xlim(xmin=0, xmax=14.5)
ax2.set_ylim(ymin=0, ymax=0.6)
plt.subplots_adjust(left=0.13, bottom=0.18, top=0.95, right=1)

pld0 = ax2.plot(
    obj2['s0'][0], obj2['s0'][1], label='Distance to Obstacle 1')

pld1 = ax2.plot(
    obj2['s1'][0], obj2['s1'][1], label='Distance to Obstacle 2')

pld2 = ax2.plot(
    obj2['s2'][0], obj2['s2'][1], label='Distance to Goal')

# plm = ax2.plot(
#     obj['m'][0], obj['m'][1], label='Manipuability')

ax2.legend()
ax2.legend(loc="upper right")

fig2.savefig('neo2.eps')


# ----------------------------------------------------------------------------
fig3, ax3 = plt.subplots()
fig3.set_size_inches(2.5, 1.5)

ax3.set(xlabel='Time (s)', ylabel='Distance (m)')
ax3.grid()
plt.grid(True)
ax3.set_xlim(xmin=0, xmax=14.5)
ax3.set_ylim(ymin=0, ymax=0.6)
plt.subplots_adjust(left=0.13, bottom=0.18, top=0.95, right=1)

pld0 = ax3.plot(
    obj3['s0'][0], obj3['s0'][1], label='Distance to Obstacle 1')

pld1 = ax3.plot(
    obj3['s1'][0], obj3['s1'][1], label='Distance to Obstacle 2')

pld2 = ax3.plot(
    obj3['s2'][0], obj3['s2'][1], label='Distance to Goal')

# plm = ax3.plot(
#     obj['m'][0], obj['m'][1], label='Manipuability')

ax3.legend()
ax3.legend(loc="upper right")

fig3.savefig('neo3.eps')

# ----------------------------------------------------------------------------
fig4, ax4 = plt.subplots()
fig4.set_size_inches(2.5, 1.5)

ax4.set(xlabel='Time (s)', ylabel='Manipulability')
ax4.grid()
plt.grid(True)
ax4.set_xlim(xmin=0, xmax=14.5)
ax4.set_ylim(ymin=0, ymax=0.13)
plt.subplots_adjust(left=0.13, bottom=0.18, top=0.95, right=1)

pld0 = ax4.plot(
    obj1['m'][0], obj1['m'][1], label='Trajectory a')

pld1 = ax4.plot(
    obj2['m'][0], obj2['m'][1], label='Trajectory b')

pld2 = ax4.plot(
    obj3['m'][0], obj3['m'][1], label='Trajectory c')

ax4.legend()
ax4.legend(loc="lower right")

fig4.savefig('neom.eps')

plt.show()
