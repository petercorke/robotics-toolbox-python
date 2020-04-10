# Created by: Aditya Dua
# 30 September 2017
from __future__ import print_function
from abc import ABC
import math
from math import pi
import numpy as np
import vtk
from . import transforms
from .graphics import VtkPipeline
from .graphics import axesCube
from .graphics import axesCubeFloor
from .graphics import vtk_named_colors
import pkg_resources
from scipy.optimize import minimize


class SerialLink:
    """
    SerialLink object class.
    """

    def __init__(self, links, name=None, base=None, tool=None, stl_files=None, q=None, colors=None, param=None):
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
            self.q = np.matrix([0 for each in links])
        if base is None:
            self.base = np.asmatrix(np.eye(4, 4))
        else:
            assert (type(base) is np.matrix) and (base.shape == (4, 4))
            self.base = base
        if tool is None:
            self.tool = np.asmatrix(np.eye(4, 4))
        else:
            assert (type(tool) is np.matrix) and (tool.shape == (4, 4))
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
        if colors is None:
            self.colors = vtk_named_colors(["Grey"] * len(stl_files))
        else:
            self.colors = colors
        if param is None:
            # If model deosn't pass params, then use these default ones
            self.param = {
                "cube_axes_x_bounds": np.matrix([[-1.5, 1.5]]),
                "cube_axes_y_bounds": np.matrix([[-1.5, 1.5]]),
                "cube_axes_z_bounds": np.matrix([[-1.5, 1.5]]),
                "floor_position": np.matrix([[0, -1.5, 0]])
            }
        else:
            self.param = param

    def __iter__(self):
        return (each for each in self.links)

    @property
    def length(self):
        """
        length property
        :return: int
        """
        return len(self.links)

    def fkine(self, stance, unit='rad', apply_stance=False, actor_list=None, timer=None):
        """
        Calculates forward kinematics for a list of joint angles.
        :param stance: stance is list of joint angles.
        :param unit: unit of input angles.
        :param apply_stance: If True, then applied tp actor_list.
        :param actor_list: Passed to apply transformations computed by fkine.
        :param timer: internal use only (for animation).
        :return: homogeneous transformation matrix.
        """
        if type(stance) is np.ndarray:
            stance = np.asmatrix(stance)
        if unit == 'deg':
            stance = stance * pi / 180
        if timer is None:
            timer = 0
        t = self.base
        for i in range(self.length):
            if apply_stance:
                actor_list[i].SetUserMatrix(transforms.np2vtk(t))
            t = t * self.links[i].A(stance[timer, i])
        t = t * self.tool
        if apply_stance:
            actor_list[self.length].SetUserMatrix(transforms.np2vtk(t))
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
        bounds = [(link.qlim[0], link.qlim[1]) for link in self]
        reach = 0
        for link in self:
            reach += abs(link.a) + abs(link.d)
        omega = np.diag([1, 1, 1, 3 / reach])
        if q0 is None:
            q0 = np.asmatrix(np.zeros((1, self.length)))

        def objective(x):
            return (
                np.square(((np.linalg.lstsq(T, self.fkine(x))[0]) - np.asmatrix(np.eye(4, 4))) * omega)).sum()

        sol = minimize(objective, x0=q0, bounds=bounds)
        if unit == 'deg':
            return np.asmatrix(sol.x * 180 / pi)
        else:
            return np.asmatrix(sol.x)

    def plot(self, stance, unit='rad'):
        """
        Plots the SerialLink object in a desired stance.
        :param stance: list of joint angles for SerialLink object.
        :param unit: unit of input angles.
        :return: null.
        """

        assert type(stance) is np.matrix

        if unit == 'deg':
            stance = stance * (pi / 180)

        self.pipeline = VtkPipeline()
        self.pipeline.reader_list, self.pipeline.actor_list, self.pipeline.mapper_list = self.__setup_pipeline_objs()

        self.fkine(stance, apply_stance=True, actor_list=self.pipeline.actor_list)

        cube_axes = axesCubeFloor(self.pipeline.ren,
                                  self.param.get("cube_axes_x_bounds"),
                                  self.param.get("cube_axes_y_bounds"),
                                  self.param.get("cube_axes_z_bounds"),
                                  self.param.get("floor_position"))

        self.pipeline.add_actor(cube_axes)

        for each in self.pipeline.actor_list:
            each.SetScale(self.scale)

        self.pipeline.render()

    def __setup_pipeline_objs(self):
        """
        Internal function to initialise vtk objects.
        :return: reader_list, actor_list, mapper_list
        """
        reader_list = [0] * len(self.stl_files)
        actor_list = [0] * len(self.stl_files)
        mapper_list = [0] * len(self.stl_files)
        for i in range(len(self.stl_files)):
            reader_list[i] = vtk.vtkSTLReader()
            loc = pkg_resources.resource_filename("robopy", '/'.join(('media', self.name, self.stl_files[i])))
            reader_list[i].SetFileName(loc)
            mapper_list[i] = vtk.vtkPolyDataMapper()
            mapper_list[i].SetInputConnection(reader_list[i].GetOutputPort())
            actor_list[i] = vtk.vtkActor()
            actor_list[i].SetMapper(mapper_list[i])
            actor_list[i].GetProperty().SetColor(self.colors[i])  # (R,G,B)

        return reader_list, actor_list, mapper_list

    @staticmethod
    def _setup_file_names(num):
        file_names = []
        for i in range(0, num):
            file_names.append('link' + str(i) + '.stl')

        return file_names

    def animate(self, stances, unit='rad', frame_rate=25, gif=None):
        """
        Animates SerialLink object over nx6 dimensional input matrix, with each row representing list of 6 joint angles.
        :param stances: nx6 dimensional input matrix.
        :param unit: unit of input angles. Allowed values: 'rad' or 'deg'
        :param frame_rate: frame_rate for animation. Could be any integer more than 1. Higher value runs through stances faster.
        :return: null
        """
        if unit == 'deg':
            stances = stances * (pi / 180)

        self.pipeline = VtkPipeline(total_time_steps=stances.shape[0] - 1, gif_file=gif)
        self.pipeline.reader_list, self.pipeline.actor_list, self.pipeline.mapper_list = self.__setup_pipeline_objs()
        self.fkine(stances, apply_stance=True, actor_list=self.pipeline.actor_list)
        self.pipeline.add_actor(axesCube(self.pipeline.ren))

        def execute(obj, event):
            nonlocal stances
            self.pipeline.timer_tick()

            self.fkine(stances, apply_stance=True, actor_list=self.pipeline.actor_list, timer=self.pipeline.timer_count)
            self.pipeline.iren = obj
            self.pipeline.iren.GetRenderWindow().Render()

        self.pipeline.iren.AddObserver('TimerEvent', execute)
        self.pipeline.animate()


class Link(ABC):
    """
    Link object class.
    """

    def __init__(self, j, theta, d, a, alpha, offset=None, kind='', mdh=0, flip=None, qlim=None):
        """
        initialises the link object.
        :param j:
        :param theta:
        :param d:
        :param a:
        :param alpha:
        :param offset:
        :param kind: 'r' or 'p' as input. 'r' for Revolute. 'p' for Prismatic.
        :param mdh:
        :param flip:
        :param qlim:
        """
        self.theta = theta
        self.d = d
        # self.j = j
        self.a = a
        self.alpha = alpha
        self.offset = offset
        self.kind = kind
        self.mdh = mdh
        self.flip = flip
        self.qlim = qlim

    def A(self, q):
        sa = math.sin(self.alpha)
        ca = math.cos(self.alpha)
        if self.flip:
            q = -q + self.offset
        else:
            q = q + self.offset
        st = 0
        ct = 0
        d = 0
        if self.kind == 'r':
            st = math.sin(q)
            ct = math.cos(q)
            d = self.d
        elif self.kind == 'p':
            st = math.sin(self.theta)
            ct = math.cos(self.theta)
            d = q

        se3_np = 0
        if self.mdh == 0:
            se3_np = np.matrix([[ct, -st * ca, st * sa, self.a * ct],
                                [st, ct * ca, -ct * sa, self.a * st],
                                [0, sa, ca, d],
                                [0, 0, 0, 1]])

        return se3_np


class Revolute(Link):
    """
    Revolute object class.
    """

    def __init__(self, j, theta, d, a, alpha, offset, qlim):
        """
        Initialised revolute object.
        :param j:
        :param theta:
        :param d:
        :param a:
        :param alpha:
        :param offset:
        :param qlim:
        """
        super().__init__(j=j, theta=theta, d=d, a=a, alpha=alpha, offset=offset, kind='r', qlim=qlim)
        pass


class Prismatic(Link):
    """
    Prismatic object class.
    """

    def __init__(self, j, theta, d, a, alpha, offset, qlim):
        """
        Initialises prismatic object.
        :param j:
        :param theta:
        :param d:
        :param a:
        :param alpha:
        :param offset:
        :param qlim:
        """
        super().__init__(j=j, theta=theta, d=d, a=a, alpha=alpha, offset=offset, kind='p', qlim=qlim)
        pass

    pass
