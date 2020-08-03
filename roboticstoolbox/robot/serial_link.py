#!/usr/bin/env python

import numpy as np
from roboticstoolbox.robot.Link import *
from spatialmath.pose3d import *
from scipy.optimize import minimize
import graphics as gph


class SerialLink:

    def __init__(self, links, name=None, base=None, tool=None, stl_files=None, q=None, param=None, manufacturer=None, comment=None, meshdir=None, configurations={}):
        """
        Creates a SerialLink object.
        :param links: a list of links that will constitute SerialLink object.
        :param name: name property of the object.
        :param base: base transform applied to the SerialLink object.
        :param stl_files: STL file names to associate with links. Only works for pre-implemented models in model module.
        :param q: initial angles for link joints.
        :param colors: colors of STL files.
        """
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
        if alltout:
            allt = [SE3(self.base)]
        if jointconfig.size == self.length:
            t = SE3(self.base)
            for i in range(self.length):
                t *= self.links[i].A(jointconfig[i])
                if alltout:
                    allt.append(t)
            #TODO tool isn't shown in allt
            t *= SE3(self.tool)
        else:
            assert jointconfig.shape[1] == self.length, "joinconfig must have {self.length} columns"
            t = list(range(0, jointconfig.shape[0]))
            for k in range(jointconfig.shape[0]):
                qk = jointconfig[k, :]
                tt = SE3(self.base)
                for i in range(self.length):
                    tt *= self.links[i].A(qk[i])
                t[k] = tt * SE3(self.tool)
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

            g_canvas = gph.GraphicsCanvas3D()
            print("canvas created")

            roplot = gph.GraphicalRobot(g_canvas, self.name)

            for i in range(1, len(poses)):

                # calculate length of joint
                x1 = poses[i-1].t[0]
                x2 = poses[i].t[0]
                y1 = poses[i-1].t[1]
                y2 = poses[i].t[1]
                z1 = poses[i-1].t[2]
                z2 = poses[i].t[2]
                length = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1))

                # add joint to robot
                roplot.append_link(self.links[i-1].jointtype, poses[i], length)

            return
