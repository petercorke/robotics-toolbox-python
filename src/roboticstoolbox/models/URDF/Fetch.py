#!/usr/bin/env python

import re

import numpy as np
from roboticstoolbox.models.URDF.URDFRobot import URDFRobot

# from spatialmath import SE3


def _patch_fetch_urdf(text: str) -> str:
    """Work around a broken upstream file served by robot_descriptions.

    Applies to: robot_descriptions v2.0.0's ``fetch_description``, which
    clones ``openai/roboschool`` at commit ``c8ee2812`` and exposes
    ``roboschool/models_robot/fetch_description/robots/fetch.urdf``.

    That file is not well-formed XML: it has a single
    ``<sensor:camera>...</sensor:camera>`` block (inside a trailing
    ``<gazebo reference="head_camera_rgb_optical_frame">`` element) using the
    ``sensor:`` namespace prefix, but no ``xmlns:sensor`` is ever declared
    anywhere in the document — old pre-SDF Gazebo camera-sensor syntax
    (circa ROS Fuerte/Groovy) that was never valid XML to begin with.
    ``roboschool`` is an archived (read-only) GitHub repo, so this can't be
    fixed upstream. The block is pure Gazebo simulation config (image
    format/size/FOV/clip planes for a camera plugin) with no bearing on
    kinematics, dynamics, or geometry, so it's safe to drop entirely.

    If a future robot_descriptions update points at a fixed fork or file,
    this patch becomes a no-op (the regex simply won't match) — safe to
    remove once confirmed unnecessary.
    """
    return re.sub(
        r"\s*<gazebo reference=\"head_camera_rgb_optical_frame\">\s*"
        r"<sensor:camera.*?</sensor:camera>\s*</gazebo>\n?",
        "\n",
        text,
        flags=re.DOTALL,
    )


class Fetch(URDFRobot):
    """
    Class that imports a Fetch URDF model

    ``Fetch()`` is a class which imports a Fetch robot definition
    from a URDF file.  The model describes its kinematic and graphical
    characteristics.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.URDF.Fetch()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration, arm is stretched out in the x-direction
    - qr, tucked arm configuration

    .. codeauthor:: Kerry He
    .. sectionauthor:: Peter Corke
    """

    def __init__(self):
        super().__init__(
            "fetch",
            manufacturer="Fetch",
            gripper_link_index=11,
            patch=_patch_fetch_urdf,
        )

        self.qdlim = np.array([4.0, 4.0, 0.1, 1.25, 1.45, 1.57, 1.52, 1.57, 2.26, 2.26])

        self.qz = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.qr = np.array([0, 0, 0.05, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0])

        self.addconfiguration("qr", self.qr)
        self.addconfiguration("qz", self.qz)


if __name__ == "__main__":  # pragma nocover
    robot = Fetch()
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
