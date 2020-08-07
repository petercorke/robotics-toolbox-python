#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
from ropy.backend.PyPlot.EllipsePlot import EllipsePlot


def _vellipse(robot, q=None, opt='trans', centre=[0, 0, 0]):

    ell = EllipsePlot(robot, opt, centre=centre)
    return ell


def _plot_vellipse(ellipse, block=True, limits=None):

    if not isinstance(ellipse, EllipsePlot):
        raise TypeError(
            'ellipse must be of type ropy.backend.PyPlot.EllipsePlot')

    env = rp.backend.PyPlot()

    # Add the robot to the figure in readonly mode
    env.launch(ellipse.robot.name + ' Velocity Ellipse', limits=limits)

    env.add(ellipse)

    # Keep the plot open
    if block:           # pragma: no cover
        env.hold()

    return env
