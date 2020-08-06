#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
import time
from spatialmath.base.argcheck import getvector, verifymatrix


def _plot(robot, block, q, dt):

    # Make an empty 3D figure
    env = rp.PyPlot()

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
        env.launch(robot.name + ' Plot')
    else:
        env.launch(robot.name + ' Trajectory Plot')

    env.add(robot, readonly=True)

    if trajn != 1:
        for i in range(trajn):
            robot.q = q[:, i]
            print(robot.q)
            env.step()
            time.sleep(dt/1000)

    # Keep the plot open
    if block:
        env.hold()

    return env
