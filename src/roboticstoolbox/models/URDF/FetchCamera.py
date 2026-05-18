#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot

# from spatialmath import SE3


class FetchCamera(Robot):
    """
    Class that imports a FetchCamera URDF model

    ``FetchCamera()`` is a class which imports a FetchCamera robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Fetch()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration
    - qr, zero joint angle configuration

    .. codeauthor:: Kerry He
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):

        links, name, urdf_string, urdf_filepath = self.URDF_read(
            "fetch_description/robots/fetch_camera.urdf"
        )

        super().__init__(
            links,
            name=name,
            manufacturer="Fetch",
            gripper_links=links[6],
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # self.grippers[0].tool = SE3(0, 0, 0.1034)
        self.qdlim = np.array([4.0, 4.0, 0.1, 1.57, 1.57])

        self.qz = np.array([0, 0, 0, 0, 0])
        self.qr = np.array([0, 0, 0, 0, 0])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover

    robot = FetchCamera()
    print(robot)

    for link in robot.links:
        print(link.name)
        print(link.isjoint)
        print(len(link.collision))

    print()

    for link in robot.grippers[0].links:
        print(link.name)
        print(link.isjoint)
        print(len(link.collision))
