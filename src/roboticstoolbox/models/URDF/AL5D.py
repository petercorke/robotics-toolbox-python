#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot
from math import pi


class AL5D(Robot):
    """
    Class that imports a AL5D URDF model

    ``AL5D()`` is a class which imports a Lynxmotion AL5D robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.AL5D()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, 'L' shaped configuration
    - up, robot poiting upwards

    .. codeauthor:: Tassos Natsakis
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "al5d_description/urdf/al5d_robot.urdf"
        )

        super().__init__(
            links,
            name=name,
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        self.manufacturer = "Lynxmotion"

        # zero angles, upper arm pointing up, lower arm straight ahead
        self.addconfiguration("qz", np.array([0, 0, 0, 0]))

        # reference pose robot pointing upwards
        self.addconfiguration("up", np.array([0.0000, 0.0000, 1.5707, 0.0000]))


if __name__ == "__main__":  # pragma nocover

    robot = AL5D()
    print(robot)
