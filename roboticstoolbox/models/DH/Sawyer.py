#!/usr/bin/env python
"""
@author: Peter Corke
"""

import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class Sawyer(DHRobot):
    """
    Class that models a Sawyer manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``Sawyer()`` is an object which models a Rethink Sawyer robot and
    describes its kinematic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Sawyer()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration

    :references:

        -`Layeghi, Daniel. “Dynamic and Kinematic Modelling of the Sawyer Arm ” Google Sites, 20 Nov. 2017 <https://sites.google.com/site/daniellayeghi/daily-work-and-writing/major-project-2>`_

    .. note:: SI units of metres are used.

    """

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi
            zero = 0.0

        deg = pi / 180
        mm = 1e-3
        
        # kinematic parameters
        a = np.r_[81, 0, 0, 0, 0, 0, 0] * mm
        d = np.r_[317, 192.5, 400, 168.5, 400, 136.3, 133.75] * mm
        alpha = [-pi/2, -pi/2, -pi/2, -pi/2, -pi/2, -pi/2, 0]

        links = []

        for j in range(6):
            link = RevoluteDH(
                d=d[j],
                a=a[j],
                alpha=alpha[j]
            )
            links.append(link)
    
        super().__init__(
            links,
            name="Sawyer",
            manufacturer="Rethink Robotics",
            keywords=('redundant', 'symbolic',),
            symbolic=symbolic
        )
    
        # zero angles
        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0]))


if __name__ == '__main__':    # pragma nocover

    sawyer = Sawyer(symbolic=False)
    print(sawyer)
