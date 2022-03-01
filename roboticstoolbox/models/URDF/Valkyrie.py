#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.ERobot import ERobot


class Valkyrie(ERobot):
    """
    Class that imports a NASA Valkyrie URDF model

    ``Valkyrie()`` is a class which imports a NASA Valkyrie robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Valkyrie()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - qr, vertical 'READY' configuration

    :reference:
        - https://github.com/gkjohnson/nasa-urdf-robots

    .. codeauthor:: Peter Corke
    """

    def __init__(self, variant="A"):

        if not variant in "ABCD":
            raise ValueError("variant must be in the range A-D")

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            f"val_description/model/robots/valkyrie_{variant}.urdf"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="NASA",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # self.addconfiguration_attr("qz", np.array([0, 0, 0, 0, 0, 0, 0]))
        # self.addconfiguration_attr("qr", np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4]))


if __name__ == "__main__":  # pragma nocover

    robot = Valkyrie("B")
    print(robot)
    env = robot.plot(np.zeros((robot.n,)))
    env.hold()
