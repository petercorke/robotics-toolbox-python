#!/usr/bin/env python

import numpy as np
import argparse
from roboticstoolbox.robot.Link import *
from spatialmath.base import argcheck


class SerialLink:
    """
    A superclass for arm type robots

    Note: Link subclass elements passed in must be all standard, or all 
          modified, DH parameters.
    
    Attributes:
    --------
        name : string
            Name of the robot
        manufacturer : string
            Manufacturer of the robot
        base : float np.ndarray(4,4)
            Locaation of the base
        tool : float np.ndarray(4,4)
            Location of the tool
        links : List[n]
            Series of links which define the robot
        mdh : int
            0 if standard D&H, else 1
        n : int
            Number of joints in the robot
        T : float np.ndarray(4,4)
            The current pose of the robot
        q : float np.ndarray(1,n)
            The current joint angles of the robot
        Je : float np.ndarray(6,n)
            The manipulator Jacobian matrix maps joint velocity to end-effector
            spatial velocity in the ee frame
        J0 : float np.ndarray(6,n)
            The manipulator Jacobian matrix maps joint velocity to end-effector
            spatial velocity in the 0 frame
        He : float np.ndarray(6,n,n)
            The manipulator Hessian matrix maps joint acceleration to end-effector
            spatial acceleration in the ee frame
        H0 : float np.ndarray(6,n,n)
            The manipulator Hessian matrix maps joint acceleration to end-effector
            spatial acceleration in the 0 frame

    """

    def __init__(self, *argv):

        self.links = []
        self.n = 0

        # format input into argparse
        argstr = ""
        known = ['name', 'comment', 'manufacturer', 'base', 'tool', 'gravity', 'offest',
                 'qlim', 'plotopt', 'plotopt3d', 'ikine', 'modified', 'configs']
        for arg in argv:
            if arg in known:
                argstr += "--" + arg + " "
            else:
                argstr += str(arg) + " "

        # Create a new Link based on parameters
        # parse all possible options
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', help="set robot name property",
                            default= 'noname')
        parser.add_argument("--comment", help="set robot comment property",
                            default='')
        parser.add_argument("--manufacturer", help="set robot manufacturer property",
                            default='')
        parser.add_argument("--base", help="set base transformation matrix property",
                            type=np.ndarray, default=np.eye(4, 4))
        parser.add_argument("--tool", help="set tool transformation matrix property",
                            type=np.ndarray, default=np.eye(4, 4))
        parser.add_argument("--gravity", help="set gravity vector property",
                            type=array, default=np.array([[0],[0],[9.81]]))
        parser.add_argument("--offset", help="Set the robot offset property",
                            type=float, default=0)
        parser.add_argument("--qlim", help="set q limit properties",
                            type=float, default=0)
        parser.add_argument("--plotopt", help="set default options for .plot()",
                            type=float, default=0)
        parser.add_argument("--plotopt3d", help="set default options for .plot3d()",
                            type=float, default=0)
        parser.add_argument("--ikine", help="link centre of gravity (3x1)",
                            type=float, default=0)
        parser.add_argument("--modified", help="joint limit",
                            type=bool, default=False)
        parser.add_argument("--configs", help="provide a cell array of predefined configurations, as name, value pairs",
                            type= float, default=0)
        (opt, arg) = parser.parse_known_args(argstr.split())

        if len(arg) == 1:
            # at least on argument, either a robot or link array

            L = argcheck.getvector(arg[0])

            if isinstance(L[0], Link):
                # passed an array of link objects
                self.links = L
            elif L.shape[1] >= 4 and (isinstance(L[0], float)):
                # passed a legacy DH matrix
                dh_dyn = L
                L = []
                for i, row in enumerate(dh_dyn):
                    if opt.modified:
                        L[i] = Link(dh_dyn[i, :], 'modified')
                    else:
                        L[i] = Link(dh_dyn[i, :])
                self.links = L

            elif isinstance(L[0], SerialLink):
                # passed a SerialLink object
                if len(L) == 1:
                    # clone the passed robot anf the attached links
                    self = L

                    self.links = L.links
                    self.name = self.name+" -copy"
                else:
                    # compound the robots in the vector
                    self = L[0]

                    for i in range(1, len(L)):
                        self.links = self.links.append(L[i].links)
                        self.name = self.name+" + "+L[i].name

                    end = len(L) - 1
                    self.tool = L[end].tool         # tool of composite robot from the last one
                    self.gravity = L[0].gravity     # gravity of composite robot from the first one

            else:
                print('unknown type passed to robot')
            self.n = len(self.links)

