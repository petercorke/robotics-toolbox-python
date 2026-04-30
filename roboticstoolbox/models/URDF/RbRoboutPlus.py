#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Robot import Robot
from roboticstoolbox.robot.Link import Link
from roboticstoolbox.robot.ETS import ETS
from roboticstoolbox.robot.ET import ET
from spatialmath import SE3


class RbRoboutPlus(Robot):
    """
    RbRoboutPlus omnidirectional manipulator from Robotnik.
    
    This class imports an RbRoboutPlus robot definition from a URDF file.
    The RbRoboutPlus is an omnidirectional mobile manipulator with:
    - 3-DOF omnidirectional base for mobility (x, y, theta)
    - 6-DOF arm for manipulation
    - Gripper end-effector

    ``RbRoboutPlus(xacro_path)`` is a class which imports the robot model
    and describes its kinematic and graphical characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.RbRoboutPlus()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, all joints at 0
    - qr, ready configuration with arm in upright pose

    .. codeauthor:: Jesse Haviland
    .. sectionauthor:: Peter Corke
    """

    def __init__(self, xacro_path):
        """
        Initialize the RbRoboutPlus robot by loading URDF and setting parameters.
        
        Parameters
        ----------
        xacro_path : str
            Path to the xacro files directory containing the robot definition
        """
        
        # Load the URDF model from xacro file
        # The URDF_read method parses the xacro file and returns robot links and metadata
        links, _, urdf_string, urdf_filepath = self.URDF_read(
            "robots/rbrobout_plus.urdf.xacro",
            tld=xacro_path,
        )

        # Initialize the parent Robot class with the loaded links and metadata
        super().__init__(
            links,
            name="RbRoboutPlus",
            manufacturer="Robotnik",
            urdf_string=urdf_string,
            urdf_filepath=urdf_filepath,
        )

        # Optional: Set end-effector tool offset (currently disabled)
        # self.grippers[0].tool = SE3(0, 0, 0.1034)

        # Define joint velocity limits (rad/s) for each joint
        # First 3 joints: base mobility (x, y, theta) - 4.0 rad/s
        # Joints 4-7: arm shoulder/elbow joints - 2.175 rad/s
        # Joints 8-10: arm wrist joints - 2.61 rad/s
        self.qdlim = np.array(
            [
                4.0,      # Base x velocity limit
                4.0,      # Base y velocity limit
                4.0,      # Base rotation velocity limit
                2.1750,   # Shoulder pan joint
                2.1750,   # Shoulder lift joint
                2.1750,   # Elbow joint
                2.1750,   # Wrist 1 joint
                2.6100,   # Wrist 2 joint
                2.6100,   # Wrist 3 joint
            ]
        )

        # Ready configuration - arm in upright position
        self.qr = np.array([0, 0, 0, 0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

        # Zero configuration - all joints at zero position
        self.qz = np.zeros(10)

        # Register named configurations for easy access
        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)




if __name__ == "__main__":  # pragma nocover
    """
    Main module execution block for testing and debugging.
    This code only runs when the file is executed directly, not when imported.
    """
    pass

    # Example usage (currently commented out):
    # r = RbRoboutPlus("/path/to/xacro/files")
    # print(r)  # Display robot information

    # To inspect gripper links:
    # for link in r.grippers[0].links:
    #     print(link)
