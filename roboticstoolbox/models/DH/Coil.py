#!/usr/bin/env python
"""
@author: Peter Corke
"""

import numpy as np
from roboticstoolbox import DHRobot, RevoluteDH


class Coil(DHRobot):
    """
    Create model of a coil manipulator

    :param N: number of links, defaults to 10
    :type N: int, optional
    :param symbolic: [description], defaults to False
    :type symbolic: bool, optional

    The coil robot is an *abstract* robot with an arbitrary number of joints
    that folds into a helix shape.  At zero joint angles it is straight along
    the x-axis, and as the joint angles are increased (equally) it wraps up into
    a 3D helix shape.

    - ``Coil()`` is an object which describes the kinematic characteristics of
      a coil robot using standard DH conventions.

    - ``Coil(N)`` as above, but models a robot with ``N`` joints.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Coil()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration

    :references:
        - "A divide and conquer articulated-body algorithm for parallel O(log(n))
          calculation of rigid body dynamics, Part 2",
          Int. J. Robotics Research, 18(9), pp 876-892.

    :seealso: :func:`Hyper`, :func:`Ball`

    .. codeauthor:: Peter Corke
    """  # noqa

    def __init__(self, N=10, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym

            pi = sym.pi()
        else:
            from math import pi

        a = 1 / N

        links = []
        for i in range(N):
            links.append(RevoluteDH(a=a, alpha=5 * pi / N))

        super().__init__(
            links, name="Coil" + str(N), keywords=("symbolic",), symbolic=symbolic
        )

        self.qr = np.array(np.ones(N) * 10 * pi / N)
        self.qz = np.zeros(N)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    coil = Coil(N=10, symbolic=False)
    print(coil)

    # print(coil.fkine(coil.qz))
    # print(coil.fkine(coil.qf))
