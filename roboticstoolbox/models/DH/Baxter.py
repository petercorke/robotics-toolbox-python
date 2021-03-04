import numpy as np
from math import pi
from roboticstoolbox import DHRobot, RevoluteDH
from spatialmath import SE3
from spatialmath import base

class Baxter(DHRobot):
    """
    Class that models a Baxter manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``Baxter()`` is an object which models the left arm of the two 7-joint
    arms of a Rethink Robotics Baxter robot using standard DH conventions.

    ``Baxter(which)`` as above but models the specified arm and ``which`` is
    either 'left' or 'right'.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.Baxter()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, vertical 'READY' configuration
    - qs, arm is stretched out in the X direction
    - qd, lower arm horizontal as per data sheet

    .. note:: SI units are used.

    .. warning:: The base transform is set to reflect the pose of the arm's
        shoulder with respect to the base.  Changing the base attribute of the
        arm will overwrite this, IT DOES NOT CHANGE THE POSE OF BAXTER's base.
        Instead do ``baxter.base = newbase * baxter.base``.

    :References:
        - "Kinematics Modeling and Experimental Verification of Baxter Robot"
          Z. Ju, C. Yang, H. Ma, Chinese Control Conf, 2015.

    .. codeauthor:: Peter Corke
    """  # noqa

    def __init__(self, arm='left'):

        links = [
                RevoluteDH(d=0.27,        a=0.069, alpha=-pi/2),
                RevoluteDH(d=0,           a=0,     alpha=pi/2, offset=pi/2),
                RevoluteDH(d=0.102+0.262, a=0.069, alpha=-pi/2),
                RevoluteDH(d=0,           a=0,     alpha=pi/2),
                RevoluteDH(d=0.103+0.271, a=0.010, alpha=-pi/2),
                RevoluteDH(d=0,           a=0,     alpha=pi/2),
                RevoluteDH(d=0.28,        a=0,     alpha=0)
        ]

        super().__init__(
            links,
            name=f"Baxter-{arm}",
            manufacturer="Rethink Robotics",
        )

        # zero angles, L shaped pose
        self.addconfiguration("qz", np.array([0, 0, 0, 0, 0, 0, 0]))

        # ready pose, arm up
        self.addconfiguration("qr", np.array([0, -pi/2, -pi/2, 0, 0, 0, 0]))

        # straight and horizontal
        self.addconfiguration("qs", np.array([0, 0, -pi/2, 0, 0, 0, 0]))

        # nominal table top picking pose
        self.addconfiguration("qn", np.array([0, pi/4, pi/2, 0, pi/4, 0, 0]))

        if arm == 'left':
            self.base = SE3(0.064614, 0.25858, 0.119) * SE3.Rz(pi/4)
        else:
            self.base = SE3(0.063534, -0.25966, 0.119) * SE3.Rz(-pi/4)


if __name__ == '__main__':    # pragma nocover

    baxter = Baxter('left')
    print(baxter.name, baxter.base)
    # baxter.plot(baxter.qz, block=False)

    # baxter = Baxter('right')
    # print(baxter)

    # baxter.plot(baxter.qz)