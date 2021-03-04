#!/usr/bin/env python
"""
@author: Peter Corke
@author: Jesse Haviland
"""

# 2/8/95  changed D3 to 150.05mm which is closer to data from Lee, AKB86 and
# Tarn fixed errors in COG for links 2 and 3
# 29/1/91 to agree with data from Armstrong etal.  Due to their use
#  of modified D&H params, some of the offsets Ai, Di are
#  offset, and for links 3-5 swap Y and Z axes.
# 14/2/91 to use Paul's value of link twist (alpha) to be consistant
#  with ARCL.  This is the -ve of Lee's values, which means the
#  zero angle position is a righty for Paul, and lefty for Lee.

# all parameters are in SI units: m, radians, kg, kg.m2, N.m, N.m.s etc.

# from math import pi
import numpy as np
from roboticstoolbox import DHRobot, PrismaticDH, models
from spatialmath import SE3
from spatialmath import base
from math import pi


class P8(DHRobot):
    """
    Class that models a Puma robot on an XY base

    ``P8()`` is an object which models an 8-axis robot comprising a Puma 560
    robot on an XY base.  Joints 0 and 1 are the base, joints 2-7 are the robot
    arm.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration


    .. note::
        - SI units are used.

    .. codeauthor:: Peter Corke

    :seealso: :func:`models.DH.Puma560`
    """  # noqa

    def __init__(self):

        # create the base
        links = [
            PrismaticDH(alpha=-pi / 2, qlim=[-1, 1]),
            PrismaticDH(theta=-pi / 2, alpha=pi / 2, qlim=[-1, 1])
        ]

        # stick the Puma on top
        puma = models.DH.Puma560()
        links.extend(puma.links)

        super().__init__(
            links,
            name="P8",
            keywords=('mobile', 'redundant'),
            base=SE3.Ry(pi/2)
        )

        self.addconfiguration("qz", np.zeros((8,)))

if __name__ == '__main__':    # pragma nocover

    robot = P8()
    print(robot)


    # robot.plot([.5, -0.6, 0, 0, 0, 0, 0, 0], block=True)


