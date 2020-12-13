#!/usr/bin/env python
"""
@author: Peter Corke
"""


# from math import pi
import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3


class Hyper(DHRobot):
    """
    Create model of a hyper redundant planar manipulator

    :param N: number of links, defaults to 10
    :type N: int, optional
    :param a: link length, defaults total ``1/N`` giving a reach of 1
    :type a: int or symbolic, optional
    :param symbolic: [description], defaults to False
    :type symbolic: bool, optional

    - ``Hyper()`` is an object which describes the kinematics of a serial link
      manipulator with 10 joints which moves in the xy-plane, using standard DH
      conventions. At zero angles it forms a straight line along the x-axis. 

    - ``Hyper(N)`` as above, but models a robot with ``N`` joints.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Hyper()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration

    :References: 
    
      - "A divide and conquer articulated-body algorithm for parallel O(log(n))
      calculation of rigid body dynamics, Part 2",
      Int. J. Robotics Research, 18(9), pp 876-892. 

    :seealso: :func:`Coil`, :func:`Ball`

    .. codeauthor:: Peter Corke
    """  # noqa

    def __init__(self, N=10, a=None, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi
            zero = 0.0

        if a is None:
            a = 1 / N

        links = []
        for i in range(N):
            links.append(RevoluteDH(a=a, alpha=zero))

        super().__init__(
            links,
            name="Hyper" + str(N),
            keywords=('symbolic',),
            symbolic=symbolic
        )

        # zero angles, straight
        self.addconfiguration("qz", np.zeros((N,)))

if __name__ == '__main__':    # pragma nocover

    hyper = Hyper(N=10, symbolic=False)
    print(hyper)

    #print(hyper.fkine(hyper.qz))
