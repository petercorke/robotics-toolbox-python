#!/usr/bin/env python

from pathlib import PurePath
import numpy as np
from roboticstoolbox.robot.Link import *
from spatialmath.pose3d import *
from scipy.optimize import minimize
import graphics as gph
from roboticstoolbox.robot.trajectory import *


class SerialLink:

    def __init__(self, links, name=None, base=None, tool=None, toolmesh=None, stl_files=None, q=None, param=None, manufacturer=None, comment=None, meshdir=None, configurations={}):
        """
        Creates a SerialLink object.
        :param links: a list of links that will constitute SerialLink object.
        :param name: name property of the object.
        :param base: base transform applied to the SerialLink object.
        :param stl_files: STL file names to associate with links. Only works for pre-implemented models in model module.
        :param q: initial angles for link joints.
        :param colors: colors of STL files.
        """
        self.g_canvas = None
        self.roplot = None
        self.pipeline = None
        self.links = links
        if q is None:
            self.q = np.array([0 for each in links])
        if base is None:
            self.base = np.eye(4, 4)
        else:
            assert (type(base) is np.ndarray) and (base.shape == (4, 4))
            self.base = base
        if tool is None:
            self.tool = np.eye(4, 4)
        else:
            assert (type(tool) is np.ndarray) and (tool.shape == (4, 4))
            self.tool = tool

        if toolmesh:
            self.toolmesh = PurePath(__file__).parent.parent / 'models' / 'meshes' / toolmesh
        else:
            self.toolmesh = None

        # Following arguments initialised by plot function and animate functions only
        if stl_files is None:
            # Default stick figure model code goes here
            pass
        else:
            self.stl_files = stl_files
        if name is None:
            self.name = ''
        else:
            self.name = name
        if param is None:
            # If model deosn't pass params, then use these default ones
            self.param = {
                "cube_axes_x_bounds": np.array([[-1.5, 1.5]]),
                "cube_axes_y_bounds": np.array([[-1.5, 1.5]]),
                "cube_axes_z_bounds": np.array([[-1.5, 1.5]]),
                "floor_position": np.array([[0, -1.5, 0]])
            }
        else:
            self.param = param
        self.manufacturer = manufacturer
        self.comment = comment
        self.meshdir = meshdir
        self.configurations = configurations

    def __iter__(self):
        return (each for each in self.links)

    def __repr__(self):
        s = ''
        for joint, link in enumerate(self.links):
            s += "{:2d}: {:s}\n".format(joint+1, str(link))
        return s

    def config(self, name):
        return self.configurations[name]
    

    @property
    def length(self):
        """
        length property
        :return: int
        """
        return len(self.links)

    def fkine(self, jointconfig, unit='rad', alltout=False):
        """
        Calculates forward kinematics for a list of joint angles.
        :param jointconfig: stance is list of joint angles.
        :param unit: unit of input angles. 'rad' or 'deg'.
        :param alltout: request intermediate transformations
        :return: homogeneous transformation matrix.
        """
        if type(jointconfig) == list:
            jointconfig = argcheck.getvector(jointconfig)
        if unit == 'deg':
            jointconfig = jointconfig * pi / 180
        if jointconfig.size == self.length:
            if alltout:
                allt = [SE3(self.base)]
            t = SE3(self.base)
            for i in range(self.length):
                t *= self.links[i].A(jointconfig[i])
                if alltout:
                    allt.append(t)
            #TODO tool isn't shown in allt
            t *= SE3(self.tool)
        else:
            assert jointconfig.shape[1] == self.length, "joinconfig must have {self.length} columns"
            if alltout:
                allt = []
            t = []
            for k in range(jointconfig.shape[0]):
                qk = jointconfig[k, :]
                jointpose = SE3(self.base)
                if alltout:
                    armpose = [jointpose]
                for i in range(self.length):
                    jointpose *= self.links[i].A(qk[i])
                    if alltout:
                        armpose.append(jointpose)
                t.append(jointpose * SE3(self.tool))
                if alltout:
                    allt.append(armpose)

        if alltout:
            return allt
        else:
            return t

    def ikine(self, T, q0=None, unit='rad'):
        """
        Calculates inverse kinematics for homogeneous transformation matrix using numerical optimisation method.
        :param T: homogeneous transformation matrix.
        :param q0: initial list of joint angles for optimisation.
        :param unit: preferred unit for returned joint angles. Allowed values: 'rad' or 'deg'.
        :return: a list of 6 joint angles.
        """
        assert T.shape == (4, 4)
        if type(T) == SE3:
            T = T.A
        bounds = [(link.qlim[0], link.qlim[1]) for link in self]
        reach = 0
        for link in self:
            reach += abs(link.a) + abs(link.d)
        omega = np.diag([1, 1, 1, 3 / reach])
        if q0 is None:
            q0 = np.zeros((1, self.length))

        def objective(x):
            return (
                np.square(((np.linalg.lstsq(T, self.fkine(x).A, rcond=-1)[0]) - np.eye(4, 4)) * omega)).sum()

        sol = minimize(objective, x0=q0, bounds=bounds)
        if unit == 'deg':
            return sol.x * 180 / pi
        else:
            return sol.x

    def plot(self, jointconfig, unit='rad'):
        """
        Creates a 3D plot of the robot in your web browser
        :param jointconfig: takes an array or list of joint angles
        :param unit: unit of angles. radians if not defined
        :return: a vpython robot object.
        """

        if type(jointconfig) == list:
            jointconfig = argcheck.getvector(jointconfig)
        if unit == 'deg':
            jointconfig = jointconfig * pi / 180
        if jointconfig.size == self.length:
            poses = self.fkine(jointconfig, unit, alltout=True)

        if self.roplot is None:
            # No current plot, create robot plot

                self.g_canvas = gph.GraphicsCanvas3D()
                print("canvas created")

                self.roplot = gph.GraphicalRobot(self.g_canvas, self.name, self)

                return
        else:
            # Move existing plot
            self.roplot.set_joint_poses(poses)
            return

    def animate(self, q1, q2, unit='rad', frames=10, fps=5):
        """
        animates an existing plot.
        :param q1: starting joint config
        :param q2: end joint config
        :param unit: unit of angles #TODO
        :param frames: steps between joint configurations
        :param fps: frames per second. used by graphics library
        """
        assert (self.roplot is not None), "plot has not been created. Use plot() first."
        traj = jtraj(q1, q2, frames)
        tposes = self.fkine(traj, alltout=True)
        self.roplot.animate(tposes, fps)