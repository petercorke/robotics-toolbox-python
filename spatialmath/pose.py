# Created by: Aditya Dua
# 13 June, 2017
# Pose module has class implementations of SO2, SO3, SE2 and SE3 type matrices

import numpy as np
import math
from . import check_args
from .super_pose import SuperPose
from random import uniform, randint
from . import transforms
import vtk
from . import graphics
from .graphics import VtkPipeline


# TODO Implement argument checking for all poses
# -----------------------------------------------------------------------------------------
class SO2(SuperPose):
    # --------------------------------------------------------------------------------------

    def __init__(self, args_in=None, unit='rad', null=False):

        check_args.unit_check(unit)
        check_args.so2_input_types_check(args_in)
        self._unit = unit
        self._list = []
        angles_deg = []

        if null:  # Usually only internally used to create empty objects
            pass
        elif args_in is None:
            self._list.append(np.asmatrix(np.eye(2, 2)))
        elif isinstance(args_in, int) or isinstance(args_in, float):
            if unit == 'deg':
                args_in = args_in * math.pi / 180
            self._list.append(np.matrix([[math.cos(args_in), -math.sin(args_in)],
                                         [math.sin(args_in), math.cos(args_in)]]))
        elif isinstance(args_in, SO2):
            check_args.so2_valid(args_in)
            for each_matrix in args_in:
                self._list.append(each_matrix)
        elif isinstance(args_in, np.matrix):
            if SO2.is_valid(args_in):
                self._list.append(args_in)
        elif isinstance(args_in, list):
            check_args.so2_angle_list_check(args_in)
            if unit == "deg":
                for each_angle in args_in:
                    angles_deg.append(each_angle * math.pi / 180)
            for each_angle in angles_deg:
                self._list.append(transforms.rot2(each_angle))
        else:
            raise AttributeError("\nINVALID instantiation. Valid scenarios:-\n"
                                 "SO2(angle)\n"
                                 "SO2(list of angles)\n"
                                 "SO2(angle, unit)\n"
                                 "SO2(list of angles, unit)\n"
                                 "SO2()\n"
                                 "SO2(so2)\n"
                                 "SO2(np.matrix)\n")
        # Round all matrices to 15 decimal places
        # Removes eps values
        for i in range(len(self._list)):
            self._list[i] = np.asmatrix(self._list[i].round(15))

    @staticmethod
    def is_valid(obj):
        """Checks if a np.matrix is a valid SO2 pose."""
        if type(obj) is np.matrix \
                and obj.shape == (2, 2) \
                and abs(np.linalg.det(obj) - 1) < np.spacing([1])[0]:
            return True
        else:
            return False

    @staticmethod
    def form_trans_matrix(rot, transl):
        rot = np.r_[rot, np.matrix([0, 0])]
        rot = np.c_[rot, np.matrix([[transl[0]], [transl[1]], [1]])]
        return rot

    @staticmethod
    def check(obj):
        if type(obj) is SO2:
            err = None
            try:
                check_args.so2_valid(obj)
            except AssertionError as err:
                pass
            if err is None:
                return SO2(obj)
            else:
                raise ValueError('INVALID SO2 object.')
        elif type(obj) is SE2:
            err = None
            try:
                check_args.se2_valid(obj)
            except AssertionError as err:
                pass
            if err is None:
                return SE2(obj)
            else:
                raise ValueError('INVALID SE2 object')
        elif type(obj) is np.matrix and obj.shape == (2, 2):
            if SO2.is_valid(obj):
                return SO2(obj)
            else:
                raise ValueError('INVALID 2x2 np.matrix')
        elif type(obj) is np.matrix and obj.shape == (3, 3):
            if SE2.is_valid(obj):
                return SE2(obj)
            else:
                raise ValueError('INVALID 3x3 np.matrix')
        else:
            raise ValueError("\nINVALID argument.\n"
                             "check(obj) accepts valid:\n"
                             "- SO2\n"
                             "- SE2\n"
                             "- np.matrix (2, 2)\n"
                             "- np.matrix (3, 3)\n")

    @classmethod
    def rand(cls):
        obj = cls(uniform(0, 360), unit='deg')
        return obj

    @staticmethod
    def exp():
        # TODO !! How to ?
        pass

    @property
    def angle(self):
        """Returns angle of SO2 object matrices in unit radians"""
        angles = []
        for each_matrix in self:
            angles.append(math.atan2(each_matrix[1, 0], each_matrix[0, 0]))
        # TODO !! Return list be default ?
        if len(angles) == 1:
            return angles[0]
        elif len(angles) > 1:
            return angles

    @property  # TODO Remove if useless
    def unit(self):
        return self._unit

    @unit.setter  # TODO Remove if useless
    def unit(self, val):
        assert val == 'deg' or val == 'rad'
        self._unit = val

    @property
    def det(self):
        """Returns a list containing determinants of all matrices in a SO2 object"""
        val = []
        for i in range(len(self._list)):
            val.append(np.linalg.det(self._list[i]))
        return val

    def t_matrix(self):
        """Returns a list of transformation matrices"""
        mat = []
        for each_matrix in self:
            mat.append(SO2.form_trans_matrix(each_matrix, (0, 0)))
        return mat

    def SE2(self):
        """Returns SE2 object with same rotational component as SO2 and a zero translation component"""
        se2_pose = SE2(null=True)  # Creates empty poses with no data
        for each_matrix in self:
            se2_pose.append(transforms.r2t(each_matrix))
        return se2_pose

    def inv(self):
        """Returns inverse SO2 object"""
        new_pose = SO2(null=True)
        for each_matrix in self:
            new_pose.append(np.matrix.transpose(each_matrix))
        return new_pose

    def eig(self):
        # TODO !! How to ?
        pass

    def log(self):
        # TODO !! How to ?
        pass

    def interp(self, other, s):
        """Returns the interpolated SO2 object"""
        # TODO Refactor "angle" to "theta" everywhere
        check_args.so2_interp_check(self, other, s)
        if type(self.angle) is list:
            angle_diff = []
            for i in range(len(self._list)):
                angle_diff.append(self.angle[i] + s * (other.angle[i] - self.angle[i]))
            return SO2(angle_diff)
        else:
            angle_diff = self.angle + s * (other.angle - self.angle)
            return SO2(angle_diff)

    def new(self):
        """Returns a deep copy of SO2 object"""
        new_pose = SO2(null=True)
        for each_matrix in self:
            new_pose.append(each_matrix)
        return new_pose

    def plot(self):

        angles = self.angle
        if type(angles) == int or type(angles) == float:
            angles = [angles]
        z = [0, ] * len(angles)
        x = []
        y = []
        if type(self) is SO2:
            x = [0, ] * len(angles)
            y = [0, ] * len(angles)
        elif type(self) is SE2:
            for each in self.transl:
                x.append(each[0])
                y.append(each[1])
        pose_se3 = SE3.Rz(theta=angles, x=x, y=y, z=z)
        axes_pose = [graphics.axesActor2d() for each in pose_se3]
        vtk_mat = [transforms.np2vtk(each) for each in pose_se3]
        pipeline = VtkPipeline()
        axis_x_y = graphics.axes_x_y(pipeline.ren)

        for i in range(pose_se3.length):
            axes_pose[i].SetUserMatrix(vtk_mat[i])
            axes_pose[i].SetAxisLabels(0)
            pipeline.add_actor(axes_pose[i])

        pipeline.add_actor(axis_x_y)
        pipeline.render()


# ---------------------------------------------------------------------------------
class SE2(SO2):
    # ---------------------------------------------------------------------------------
    def __init__(self, theta=None, unit='rad', x=None, y=None, rot=None, so2=None, se2=None, null=False):
        check_args.unit_check(unit)
        check_args.se2_constructor_args_check(x, y, rot, theta, so2, se2)
        self._list = []
        self._transl = []
        self._unit = unit
        if theta is None:
            theta = 0
        if unit == 'deg':
            if isinstance(theta, list):
                for i in range(len(theta)):
                    theta[i] = theta[i] * math.pi / 180
            else:
                theta = theta * math.pi / 180

        if null:
            pass
        elif x is not None and y is not None and rot is None and se2 is None and so2 is None:
            if isinstance(x, list) and isinstance(y, list):
                for i in range(len(x)):
                    self._transl.append((x[i], y[i]))
                    angle = 0
                    if isinstance(theta, list):
                        angle = theta[i]
                    else:
                        angle = theta
                    mat = transforms.rot2(angle)
                    mat = SO2.form_trans_matrix(mat, (x[i], y[i]))
                    self._list.append(mat)
            else:
                self._transl.append((x, y))
                mat = transforms.rot2(theta)
                mat = SO2.form_trans_matrix(mat, (x, y))
                self._list.append(mat)
        elif x is not None and y is not None and rot is not None and se2 is None and so2 is None:
            if isinstance(x, list) and isinstance(y, list) and isinstance(rot, list):
                for i in range(len(x)):
                    self._transl.append((x[i], y[i]))
                    mat = SO2.form_trans_matrix(rot[i], (x[i], y[i]))
                    self._list.append(mat)
            else:
                self._transl.append((x, y))
                mat = SO2.form_trans_matrix(rot, (x, y))
                self._list.append(mat)
        elif x is None and y is None and rot is not None and se2 is None and so2 is None:
            if isinstance(rot, list):
                for i in range(len(rot)):
                    self._transl.append((0, 0))
                    mat = SO2.form_trans_matrix(rot[i], (0, 0))
                    self._list.append(mat)
            else:
                self._transl.append((0, 0))
                mat = SO2.form_trans_matrix(rot, (0, 0))
                self._list.append(mat)
        elif x is None and y is None and rot is None and se2 is not None and so2 is None:
            self._transl = se2.transl
            for each_matrix in se2:
                self._list.append(each_matrix)
        elif x is None and y is None and rot is None and se2 is None and so2 is not None:
            for i in range(so2.length):
                self._transl.append((0, 0))
            for each_matrix in so2:
                mat = SO2.form_trans_matrix(each_matrix, (0, 0))
                self._list.append(mat)
        elif x is None and y is None and rot is None and se2 is None and so2 is None and isinstance(theta, list):
            for i in range(len(theta)):
                mat = transforms.rot2(theta[i])
                mat = SO2.form_trans_matrix(mat, (0, 0))
                self._list.append(mat)
                self._transl.append((0, 0))
        elif x is None and y is None and rot is None and se2 is None and so2 is None and theta != 0:
            mat = transforms.rot2(theta)
            mat = SO2.form_trans_matrix(mat, (0, 0))
            self._list.append(mat)
            self._transl.append((0, 0))
        elif x is None and y is None and rot is None and se2 is None and so2 is None and theta == 0:
            self._list.append(np.asmatrix(np.eye(3, 3)))
            self._transl.append((0, 0))
        else:
            raise AttributeError("\nINVALID instantiation. Valid scenarios:-\n"
                                 "- SE2(x, y)\n"
                                 "- SE2(x, y, rot)\n"
                                 "- SE2(x, y, theta)\n"
                                 "- SE2(se2)\n"
                                 "- SE2(so2)\n"
                                 "- SE2(theta)\n"
                                 "- SE2(rot)\n")
        # Round all matrices to 15 decimal places
        # Removes eps values
        for i in range(len(self._list)):
            self._list[i] = np.asmatrix(self._list[i].round(15))

    @property  # transl_vec is dependent on this !
    def transl(self):
        return self._transl

    # @transl.setter
    # def transl(self, value):
    #     assert isinstance(value, tuple) and len(value) == 2
    #     self._transl = value

    @property
    def transl_vec(self):
        """Returns list of translation vectors of SE2 object"""
        mat = []
        for each in self.transl:
            mat.append(np.matrix([[each[0]], [each[1]]]))
        return mat

    @staticmethod
    def is_valid(obj):
        # TODO
        # Check if obj is 3x3 matrix and det is 1
        pass

    def SE3(self):
        # TODO
        pass

    def t_matrix(self):
        """Returns list of translation matrices of SE2 object"""
        if self.length == 1:
            return self._list[0]
        elif self.length > 1:
            return self._list

    def inv(self):
        """Returns an inverse SE2 object of same length"""
        new_transl = []
        new_rot = []
        for i in range(len(self._list)):
            # Get rotation matrix. Transpose it. Then append in new_rot
            rot_transposed = np.matrix.transpose(transforms.t2r(self._list[i]))
            new_rot.append(rot_transposed)
            transl_mat = -rot_transposed * self.transl_vec[i]
            new_transl.append(transl_mat)

        new_x = []
        new_y = []
        for each in new_transl:  # Get all x and y translations from list of new translation vectors
            new_x.append(each[0, 0])
            new_y.append(each[1, 0])

        return SE2(x=new_x, y=new_y, rot=new_rot)

    def xyt(self, unit='rad'):
        """Return list of 3x1 dimension vectors containing x, y translation components and theta"""
        check_args.unit_check(unit)
        val = []
        assert len(self._transl) == len(self.angle)
        for i in range(len(self._list)):
            x = self._transl[i][0]
            y = self._transl[i][1]
            theta = 0
            if unit == 'deg':
                theta = self.angle[i] * 180 / math.pi
            elif unit == 'rad':
                theta = self.angle[i]
            val.append(np.matrix([[x], [y], [theta]]))
        return val

    def log(self):
        # TODO
        pass

    @classmethod
    def rand(cls):
        x = uniform(-2, 3)
        y = uniform(-2, 3)
        theta = uniform(0, 360)
        obj = cls(x=x, y=y, theta=theta, unit='deg')
        return obj


# ------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
class SO3(SuperPose):
    # -------------------------------------------------------------------------------
    def __init__(self, args_in=None, null=False):
        """
        Initialises SO3 object with identity matrix
        :param null: Creates empty objects with no matrices. Mostly for internal use only.
        """

        self.pipeline = None
        check_args.so3_constructor_args_check(args_in)
        # TODO make sure all list elements are of same data type. !!! Throw TypeError if not

        self._list = []

        if args_in is None and null is True:
            pass
        elif args_in is None and null is False:
            self._list.append(np.asmatrix(np.eye(3, 3)))
        elif type(args_in) is list:
            SO3.np(args_in)
        elif type(args_in) is SE3:
            SO3.se3(args_in)
        elif type(args_in) is SO3:
            SO3.so3(args_in)
        elif type(args_in) is np.matrix:
            self.np(args_in)
        else:
            raise AttributeError("\n INVALID instantiation. Valid scenarios:\n"
                                 "- SO3()\n"
                                 "- SO3(np.matrix_3x3)\n"
                                 "- SO3([np.matrix, np.matrix, np.matrix])\n"
                                 "- SO3(se3)\n"
                                 "- SO3([se3, se3, se3])\n"
                                 "- SO3(so3)\n"
                                 "- SO3([so3, so3, so3])\n")
        # Round all matrices to 15 decimal places
        # Removes eps values
        for i in range(len(self._list)):
            self._list[i] = np.asmatrix(self._list[i].round(15))

    @classmethod
    def so3(cls, args_in):
        assert type(args_in) is SO3
        return cls(null=True).__fill(args_in.data)

    @classmethod
    def se3(cls, args_in):
        assert type(args_in) is SE3
        obj = cls(null=True)
        for each in args_in:
            obj._list.append(np.delete(np.delete(each, [3], axis=0), [3], axis=1))
        return obj

    @classmethod
    def np(cls, args_in):
        assert type(args_in) is np.matrix or type(args_in) is list
        if type(args_in) is np.matrix:
            args_in = [args_in]
        return cls(null=True).__fill(args_in)

    @classmethod
    def Rx(cls, theta, unit="rad"):
        theta = cls.__RxRyRz(theta, unit)
        rot = [transforms.rotx(each) for each in theta]
        return cls(null=True).__fill(rot)

    @classmethod
    def Ry(cls, theta, unit="rad"):
        theta = cls.__RxRyRz(theta, unit)
        rot = [transforms.roty(each) for each in theta]
        return cls(null=True).__fill(rot)

    @classmethod
    def Rz(cls, theta, unit="rad"):
        theta = cls.__RxRyRz(theta, unit)
        rot = [transforms.rotz(each) for each in theta]
        return cls(null=True).__fill(rot)

    @classmethod
    def rand(cls):
        ran = randint(1, 3)
        if ran == 1:
            rot = transforms.rotx(uniform(0, 360), unit='deg')
            return cls(null=True).__fill([transforms.rotx(uniform(0, 360), unit='deg')])
        elif ran == 2:
            return cls(null=True).__fill([transforms.roty(uniform(0, 360), unit='deg')])
        elif ran == 3:
            return cls(null=True).__fill([transforms.rotz(uniform(0, 360), unit='deg')])

    @classmethod
    def eul(cls, theta, unit="rad"):
        if unit == 'deg':
            theta = [(each * math.pi / 180) for each in theta]
        z1_rot = transforms.rotz(theta[0])
        y_rot = transforms.roty(theta[1])
        z2_rot = transforms.rotz(theta[2])
        zyz = z1_rot * y_rot * z2_rot
        return cls(null=True).__fill([zyz])

    @classmethod
    def rpy(cls, thetas, order='zyx', unit='rad'):
        return cls(null=True).__fill([transforms.rpy2r(thetas=thetas, order=order, unit=unit)])

    @classmethod
    def oa(cls, o, a):
        # TODO
        pass

    @staticmethod
    def __RxRyRz(theta, unit):
        check_args.unit_check(unit)
        if type(theta) is float or type(theta) is int:
            theta = [theta]
        if type(theta) is list:
            if unit == 'deg':
                theta = [(each * math.pi / 180) for each in theta]
                return theta
            else:
                return theta
        else:
            raise AttributeError("\nInvalid argument type.\n"
                                 "theta must be of type: \n"
                                 "float, \n"
                                 "int, or \n"
                                 "list of float or int")

    def __fill(self, data):
        for each in data:
            self._list.append(each)
        return self

    def to_se3(self):
        pose_se3 = SE3(null=True)
        for each in self:
            pose_se3.append(transforms.r2t(each))
        return pose_se3

    def plot(self):
        pose_se3 = self
        if type(self) is SO3:
            pose_se3 = self.to_se3()
        pipeline = VtkPipeline()
        axes = [vtk.vtkAxesActor() for i in range(self.length)]
        vtk_mat = [transforms.np2vtk(each) for each in pose_se3]
        for i in range(len(axes)):
            axes[i].SetUserMatrix(vtk_mat[i])
            axes[i].SetAxisLabels(0)
            pipeline.add_actor(axes[i])

        pipeline.add_actor(graphics.axesCube(pipeline.ren))
        pipeline.render(ui=False)
        pipeline.screenshot()
        pipeline.iren.Initialize()
        pipeline.iren.Start()

    def animate(self, other=None, duration=5, gif=None):
        from .quaternion import UnitQuaternion
        assert duration > 0
        q1 = []
        q2 = []
        if other is not None:
            assert type(other) is SO3
            assert self.length == other.length
            for i in range(self.length):
                q1.append(UnitQuaternion.rot(self.data[i]))
                q2.append(UnitQuaternion.rot(other.data[i]))
        else:
            for i in range(self.length):
                q1.append(UnitQuaternion())
                q2.append(UnitQuaternion.rot(self.data[i]))

        self.pipeline = VtkPipeline(total_time_steps=duration*60, gif_file=gif)
        axis_list = []
        for i in range(self.length):
            axis_list.append(vtk.vtkAxesActor())
            axis_list[i].SetAxisLabels(0)
            axis_list[i].SetUserMatrix(transforms.np2vtk(q1[i].q2tr()))
            self.pipeline.add_actor(axis_list[i])

        cube_axes = graphics.axesCube(self.pipeline.ren)
        self.pipeline.add_actor(cube_axes)

        def execute(obj, event):
            nonlocal axis_list
            self.pipeline.timer_tick()
            print(self.pipeline.timer_count)

            for i in range(len(axis_list)):
                axis_list[i].SetUserMatrix(
                    transforms.np2vtk(
                        q1[i].interp(
                            q2[i], r=1 / self.pipeline.total_time_steps * self.pipeline.timer_count).q2tr()))
            self.pipeline.iren.GetRenderWindow().Render()

        self.pipeline.iren.AddObserver('TimerEvent', execute)
        self.pipeline.animate()

    def rotation(self):
        return self.mat

    def t_matrix(self):
        """
        Returns transformation matrices associated with the pose object.
        Return data type is list or np.matrix depending on number of transformation matrices present.
        """
        mat = []
        for each in self:
            mat.append(transforms.r2t(each))

        if self.length == 1:
            return mat[0]
        elif self.length > 1:
            return mat

    def det(self):
        det_list = []
        for each in self:
            det_list.append(np.linalg.det(each))
        if self.length == 1:
            return det_list[0]
        elif self.length > 1:
            return det_list

    def norm_vec(self):
        vec = []
        for each in self:
            vec.append(each[:, 0])  # Return first column, x-axis unit vector.

        if self.length == 1:
            return vec[0]
        elif self.length > 1:
            return vec

    def orient_vec(self):
        vec = []
        for each in self:
            vec.append(each[:, 1])  # Return first column, x-axis unit vector.

        if self.length == 1:
            return vec[0]
        elif self.length > 1:
            return vec

    def approach_vec(self):
        vec = []
        for each in self:
            vec.append(each[:, 2])  # Return first column, x-axis unit vector.

        if self.length == 1:
            return vec[0]
        elif self.length > 1:
            return vec

    def trnorm(self):
        # TODO
        pass

    def toangvec(self):
        # TODO
        pass

    def new(self):
        new_pose = SO3(null=True)
        for each in self:
            new_pose.append(each)
        return new_pose

    def exp(self):
        # TODO - maybe a static method
        pass

    @staticmethod
    def check(self):
        # TODO - maybe a static method
        pass

    def angvec(self):
        # TODO - maybe a static method
        pass

    def torpy(self):
        # TODO
        pass

    def toeul(self):
        # TODO
        pass

    def interp(self):
        # TODO
        pass

    def log(self):
        # TODO
        pass

    def inv(self):
        inv_mat = []
        for each in self:
            inv_mat.append(np.transpose(each))
        return SO3.np(inv_mat)

    def eig(self):
        vec = []
        mat = []
        for each in self:
            v, m = np.linalg.eig(each)
            vec.append(v)
            mat.append(m)
        vec = [np.transpose(np.asmatrix(each)) for each in vec]
        if self.length == 1:
            return vec[0], mat[0]
        elif self.length > 1:
            return vec, mat

    def ctraj(self, T1, N):
        from .util import ctraj
        from .common import ishomog
        from .transforms import r2t
        assert ishomog(T1, (3, 3)) or ishomog(T1, (4, 4)) or (type(self) == type(T1))

        T0 = self.data
        if type(T1) is SO3 or type(T1) is SE3:
            T1 = T1.data

        # Allow one to one, one to many, or many to one case.
        assert (len(T0) == len(T1) == 1) \
               or (len(T0) == 1 and len(T1) > 1) \
               or (len(T0) > 1 and len(T1) == 1), " Allows only one to one, one to many, or many to one case."

        if type(self) is SO3:
            for i in range(len(T0)):
                T0[i] = r2t(T0[i])

            for i in range(len(T1)):
                T1[i] = r2t(T1[i])

        if len(T0) == 1:
            T0 = T0[0]
        if len(T1) == 1:
            T1 = T1[0]

        return ctraj(T0, T1, N)


# ---------------------------------------------------------------------------------
class SE3(SO3):
    # ---------------------------------------------------------------------------------

    def __init__(self, x=None, y=None, z=None, rot=None, so3=None, se3=None, null=False):
        self._list = []
        self._transl = []
        if null:
            pass
        elif x is not None and y is not None and z is not None and rot is None and so3 is None and se3 is None:
            if (type(x) is int or type(x) is float) and (type(y) is int or type(y) is float) and (
                    type(z) is int or type(z) is float):
                x = [x]
                y = [y]
                z = [z]
            if isinstance(x, list) and isinstance(y, list) and isinstance(z, list):
                # Assert they are all same length
                for i in range(len(x)):
                    self._transl.append((x[i], y[i], z[i]))
                    rot = transforms.rotx(0)
                    self._list.append(SE3.form_trans_matrix(rot, (x[i], y[i], z[i])))
        elif x is not None and y is not None and z is not None and rot is not None and so3 is None and se3 is None:
            if (type(x) is int or type(x) is float) and \
                    (type(y) is int or type(y) is float) and \
                    (type(z) is int or type(z) is float) and \
                    (type(rot) is np.matrix):
                x = [x]
                y = [y]
                z = [z]
                rot = [rot]
            if isinstance(x, list) and isinstance(y, list) and isinstance(z, list) and isinstance(rot, list):
                for i in range(len(x)):
                    self._transl.append((x[i], y[i], z[i]))
                    self._list.append(SE3.form_trans_matrix(rot[i], (x[i], y[i], z[i])))
        elif x is not None and y is not None and z is not None and rot is None and so3 is not None and se3 is None:
            if (type(x) is int or type(x) is float) and \
                    (type(y) is int or type(y) is float) and \
                    (type(z) is int or type(z) is float) and \
                    (type(so3) is SO3):
                x = [x]
                y = [y]
                z = [z]
            if isinstance(x, list) and isinstance(y, list) and isinstance(z, list):
                for i in range(len(x)):
                    self._transl.append((x[i], y[i], z[i]))
                    self._list.append(SE3.form_trans_matrix(so3.data[i], (x[i], y[i], z[i])))
        elif x is None and y is None and z is None and rot is not None and so3 is None and se3 is None:
            if type(rot) is np.matrix:
                rot = [rot]
            if type(rot) is list:
                for i in range(len(rot)):
                    self._transl.append((0, 0, 0))
                    self._list.append(SE3.form_trans_matrix(rot[i], (0, 0, 0)))
        elif x is None and y is None and z is None and rot is None and so3 is not None and se3 is None:
            for each in so3:
                self._transl.append((0, 0, 0))
                self._list.append(SE3.form_trans_matrix(each, (0, 0, 0)))
        elif x is None and y is None and z is None and rot is None and so3 is None and se3 is not None:
            for i in range(se3.length):
                self._transl.append(se3.transl[i])
                self._list.append(se3.data[i])
        elif x is None and y is None and z is None and rot is None and so3 is None and se3 is None:
            self._list.append(np.asmatrix(np.eye(4, 4)))
            self._transl.append((0, 0, 0))
        else:
            raise AttributeError("\nINVALID instantiation. Valid scenarios:-\n"
                                 "- SE3(x, y, z)\n"
                                 "- SE3(x, y, z, rot)\n"
                                 "- SE3(x, y, z, so3)\n"
                                 "- SE3(so3)\n"
                                 "- SE3(se3)\n"
                                 "- SE3(rot)\n")

        for i in range(len(self._list)):
            self._list[i] = np.asmatrix(self._list[i].round(15))

    @property  # transl_vec is dependent on this !
    def transl(self):
        return self._transl

    @classmethod
    def Rx(cls, theta, unit="rad", x=None, y=None, z=None):
        so3 = SO3.Rx(theta, unit)
        return cls(x=x, y=y, z=z, so3=so3)

    @classmethod
    def Ry(cls, theta, unit="rad", x=None, y=None, z=None):
        so3 = SO3.Ry(theta, unit)
        return cls(x=x, y=y, z=z, so3=so3)

    @classmethod
    def Rz(cls, theta, unit="rad", x=None, y=None, z=None):
        so3 = SO3.Rz(theta, unit)
        return cls(x=x, y=y, z=z, so3=so3)

    @classmethod
    def se3(cls, args_in):
        assert (type(args_in) is SE3)
        pose_se3 = cls(null=True)
        for each in args_in:
            pose_se3.append(each)

        return pose_se3

    @classmethod
    def np(cls, arg_in):
        from .common import ishomog
        assert (type(arg_in) is np.matrix) or (type(arg_in) is list)
        se3 = cls(null=True)
        if type(arg_in) is list:
            for each in arg_in:
                assert ishomog(each, (4, 4))
                se3.append(each)
        else:
            assert ishomog(arg_in, (4, 4))
            se3.append(arg_in)
        return se3

    @staticmethod
    def form_trans_matrix(rot, transl):
        rot = np.r_[rot, np.matrix([0, 0, 0])]
        rot = np.c_[rot, np.matrix([[transl[0]], [transl[1]], [transl[2]], [1]])]
        return rot

    @classmethod
    def rand(cls):
        ran = randint(1, 3)
        theta = uniform(0, 360)
        x = uniform(-2, 2)
        y = uniform(-2, 2)
        z = uniform(-2, 2)
        if ran == 1:
            return cls.Rx(theta, unit='deg', x=x, y=y, z=z)
        elif ran == 2:
            return cls.Ry(theta, unit='deg', x=x, y=y, z=z)
        elif ran == 3:
            return cls.Rz(theta, unit='deg', x=x, y=y, z=z)

# ------------------------------------------------------------------------------------
