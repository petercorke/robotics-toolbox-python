#!/usr/bin/env python

from math import pi, sin, cos
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class Jaco(DHRobot):
    """
    Class that models a  Kinova Jaco manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``Jaco()`` is an object which models a Kinova Jaco robot and
    describes its kinematic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Puma560()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the x-direction
    - qn, arm is at a nominal non-singular configuration

    .. note::
        - SI units are used.

    :references:
        - "DH Parameters of Jaco" Version 1.0.8, July 25, 2013.

    :seealso: :func:`Mico`
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
        # robot length values (metres)
        D1 = 0.2755
        D2 = 0.4100
        D3 = 0.2073
        D4 = 0.0743
        D5 = 0.0743
        D6 = 0.1687
        e2 = 0.0098
        
        # alternate parameters
        aa = 30 * deg
        ca = cos(aa)
        sa = sin(aa)
        c2a = cos(2 * aa)
        s2a = sin(2 * aa)
        d4b = D3 + sa /s2a * D4
        d5b = sa / s2a * D4 + sa / s2a * D5
        d6b = sa / s2a * D5 + D6
        
        # and build a serial link manipulator
        
        # offsets from the table on page 4, "Mico" angles are the passed joint
        # angles.  "DH Algo" are the result after adding the joint angle offset.

        super().__init__(
                    [
                        RevoluteDH(alpha=pi/2,  a=0,  d=D1,   flip=True),
                        RevoluteDH(alpha=pi,    a=D2, d=0,    offset=-pi/2),
                        RevoluteDH(alpha=pi/2,  a=0,  d=-e2,  offset=pi/2),
                        RevoluteDH(alpha=2*aa,  a=0,  d=-d4b),
                        RevoluteDH(alpha=2*aa,  a=0,  d=-d5b, offset=-pi),
                        RevoluteDH(alpha=pi,    a=0,  d=-d6b, offset=100*deg)
                    ],
                    name='Jaco', 
                    manufacturer='Kinova',
                    keywords=('symbolic',)
                )

        self.addconfiguration('qz', np.r_[0, 0, 0, 0, 0, 0]) # zero angles
        self.addconfiguration('qr', np.r_[270, 180, 180, 0, 0, 0]*deg) # vertical pose as per Fig 2

if __name__ == '__main__':    # pragma nocover

    jaco = Jaco(symbolic=False)
    print(jaco)
