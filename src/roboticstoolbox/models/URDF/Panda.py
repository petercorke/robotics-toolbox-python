#!/usr/bin/env python

import numpy as np
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot
from spatialmath import SE3


class Panda(URDFRobot):
    """
    Class that imports a Panda URDF model

    ``Panda()`` is a class which imports a Franka-Emika Panda robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Panda()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :param use_robot_descriptions: if ``True``, load the Panda URDF from the
        `robot_descriptions <https://github.com/robot-descriptions/robot_descriptions.py>`_
        package instead of the toolbox's own bundled ``qut_frankie_description``
        xacro (the default, ``False``). The bundled model has real collision
        geometry (a hand-built capsule approximation) but no inertial
        (mass/CoM/inertia) data at all -- ``rne()``/``inertia()``/``coriolis()``/
        ``gravload()`` are all silently zero. The ``robot_descriptions`` model
        has real inertial data, but its collision geometry is plain meshes,
        which are roughly an order of magnitude slower to collision-check
        against than the bundled model's capsules -- noticeable in a
        real-time reactive-avoidance loop (see ``examples/neo.py``). See the
        wiki's `Panda models <https://github.com/petercorke/robotics-toolbox-python/wiki/Panda-models>`_
        page for the full comparison and rationale.
    :type use_robot_descriptions: bool

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self, use_robot_descriptions: bool = False):

        if use_robot_descriptions:
            super().__init__(
                "panda",
                manufacturer="Franka Emika",
                gripper_link_index=9,
            )
        else:
            super().__init__(
                "qut_frankie_description/robots/panda_arm_hand.urdf.xacro",
                manufacturer="Franka Emika",
                gripper_link_index=9,
            )

        self.grippers[0].tool = SE3(0, 0, 0.1034)

        self.qdlim = np.array(
            [2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100, 3.0, 3.0]
        )

        self.qr = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])
        self.qz = np.zeros(7)

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    r = Panda()

    r.qz

    for link in r.grippers[0].links:
        print(link)
