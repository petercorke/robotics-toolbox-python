"""
@author: Gautam Sinha, Indian Institute of Technology, Kanpur
  (original MATLAB version)
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox import DHRobot, RevoluteDH
from math import pi


class KR5(DHRobot):
    """
    Class that models a Kuka KR5 manipulator

    ``KR5()`` is a class which models a Kuka KR5 robot and
    describes its kinematic characteristics using standard DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.KR5()
        >>> print(robot)

    Defined joint configurations are:

      - qk1, nominal working position 1
      - qk2, nominal working position 2
      - qk3, nominal working position 3

    .. note::
      - SI units of metres are used.
      - Includes an 11.5cm tool in the z-direction

    :references:
      - https://github.com/4rtur1t0/ARTE/blob/master/robots/KUKA/KR5_arc/parameters.m

    .. codeauthor:: Gautam Sinha, Indian Institute of Technology, Kanpur (original MATLAB version)
    .. codeauthor:: Samuel Drew
    .. codeauthor:: Peter Corke
    """  # noqa

    def __init__(self):
        deg = pi / 180

        # Updated values form ARTE git. Old values left as comments

        L1 = RevoluteDH(a=0.18, d=0.4,
                        alpha=-pi/2,  # alpha=pi / 2,
                        qlim=[-155 * deg, 155 * deg]
                        )
        L2 = RevoluteDH(a=0.6, d=0,  # d=0.135,
                        alpha=0,  # alpha=pi,
                        qlim=[-180 * deg, 65 * deg]
                        )
        L3 = RevoluteDH(a=0.12,
                        d=0,  # d=0.135,
                        alpha=pi/2,  # alpha=-pi / 2,
                        qlim=[-15 * deg, 158 * deg]
                        )
        L4 = RevoluteDH(a=0.0,
                        d=-0.62,  # d=0.62,
                        alpha=-pi/2,  # alpha=pi / 2,
                        qlim=[-350 * deg, 350 * deg]
                        )
        L5 = RevoluteDH(a=0.0,
                        d=0.0,
                        alpha=pi/2,  # alpha=-pi / 2,
                        qlim=[-130 * deg, 130 * deg]
                        )
        L6 = RevoluteDH(a=0,
                        d=-0.115,
                        alpha=pi,
                        qlim=[-350 * deg, 350 * deg]
                        )

        L = [L1, L2, L3, L4, L5, L6]

        # Create SerialLink object
        super().__init__(
            L,
            # meshdir="KUKA/KR5_arc",
            name='KR5',
            manufacturer='KUKA',
            meshdir="meshes/KUKA/KR5_arc")

        self.addconfiguration("qz", [0, 0, 0, 0, 0, 0])
        self.addconfiguration(
          "qk1", [pi / 4, pi / 3, pi / 4, pi / 6, pi / 4, pi / 6])
        self.addconfiguration(
          "qk2", [pi / 4, pi / 3, pi / 6, pi / 3, pi / 4, pi / 6])
        self.addconfiguration(
          "qk3", [pi / 6, pi / 3, pi / 6, pi / 3, pi / 6, pi / 3])


if __name__ == '__main__':   # pragma nocover
    robot = KR5()
    print(robot)
