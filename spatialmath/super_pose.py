# Created by: Aditya Dua
# 13 June, 2017

import numpy as np
from . import check_args
from abc import ABC, abstractmethod
import numpy.testing as npt
from . import pose


class SuperPose(ABC):
    @property
    def length(self):
        """
        Property to return number of matrices in pose object
        :return: int
        """
        return len(self._list)

    @property
    def data(self):
        """
        Always returns a list containing the matrices of the pose object.
        :return: A list of matrices.
        """
        return self._list

    @property
    def mat(self):
        """
        Property to return the matrices of pose object.
        :return: Returns np.matrix type if only one matrix is present. Else returns a list of np.matrix.
        """
        if len(self._list) == 1:
            return self._list[0]
        elif len(self._list) > 1:
            return self._list

    @property
    def isSE(self):
        """
        Checks if object is of type SE2 or SE3 or none of them.
        :return: bool
        """
        if isinstance(self, pose.SE2) or isinstance(self, pose.SE3):
            return True
        else:
            return False

    @property
    def dim(self):
        """
        Returns dimensions of first matrix in pose object.
        Assumed that all matrices have same dimension.
        :return: tuple
        """
        return self._list[0].shape

    # TODO !! issym, simplify

    def is_equal(self, other):
        if (type(self) is type(other)) and (self.length == other.length):
            for i in range(self.length):
                try:
                    npt.assert_almost_equal(self.data[i], other.data[i])
                except AssertionError:
                    return False
            return True

    def append(self, item):
        check_args.super_pose_appenditem(self, item)
        if type(item) is np.matrix:
            self._list.append(item)
        else:
            for each_matrix in item:
                self._list.append(each_matrix)

    def tr_2_rt(self):
        assert isinstance(self, pose.SE2) or isinstance(self, pose.SE3)

    def t_2_r(self):
        assert isinstance(self, pose.SE2) or isinstance(self, pose.SE3)
        for each_matrix in self:
            pass  # TODO

    def isrot(self):
        return (self.dim == (3, 3)) and (not self.isSE)

    def isrot2(self):
        return (self.dim == (2, 2)) and (not self.isSE)

    def ishomog(self):
        return self.dim == (4, 4) and self.isSE

    def ishomog2(self):
        return self.dim == (3, 3) and self.isSE

    def render(self):
        pass

    def trprint(self):
        pass  # TODO

    def trplot(self):
        pass  # TODO

    def trplot2(self):
        pass  # TODO

    def tranimate(self):
        pass  # TODO

    def __mul__(self, other):
        check_args.super_pose_multiply_check(self, other)
        if isinstance(other, SuperPose):
            new_pose = type(self)(null=True)  # Creates empty poses with no data
            if self.length == other.length:
                for i in range(self.length):
                    mat = self.data[i] * other.data[i]
                    new_pose.append(mat)
                return new_pose
            else:
                for each_self_matrix in self:
                    for each_other_matrix in other:
                        mat = each_self_matrix * each_other_matrix
                        new_pose.append(mat)
            return new_pose
        elif isinstance(other, np.matrix):
            mat = []
            for each_matrix in self:
                mat.append(each_matrix * other)
            # TODO !! Return np.matrix or pose object ?
            if len(mat) == 1:
                return mat[0]
            elif len(mat) > 1:
                return mat

    def __truediv__(self, other):
        check_args.super_pose_divide_check(self, other)
        new_pose = type(self)(null=True)  # Creates empty poses with no data
        if self.length == other.length:
            for i in range(self.length):
                new_pose.append(self.data[i] * np.linalg.inv(other.data[i]))
        elif self.length == 1:
            for i in range(other.length):
                new_pose.append(np.linalg.inv(self.data[0]) * other.data[i])
        elif other.length == 1:
            for i in range(self.length):
                new_pose.append(self.data[i] * np.linalg.inv(other.data[0]))
        return new_pose

    def __add__(self, other):
        check_args.super_pose_add_sub_check(self, other)
        mat = []
        for i in range(self.length):
            mat.append(self.data[i] + other.data[i])
        # TODO !! Return np.matrix or pose object ?
        if len(mat) == 1:
            return mat[0]
        elif len(mat) > 1:
            return mat

    def __sub__(self, other):
        check_args.super_pose_add_sub_check(self, other)
        mat = []
        for i in range(self.length):
            mat.append(self.data[i] - other.data[i])
        # TODO !! Return np.matrix or pose object ?
        if len(mat) == 1:
            return mat[0]
        elif len(mat) > 1:
            return mat

    def __getitem__(self, item):
        new_pose = type(self)(null=True)
        new_pose.append(self._list[item])
        return new_pose

    def __iter__(self):
        return (each for each in self._list)

    def __repr__(self):
        if len(self.data) >= 1:
            str = '-----------------------------------------\n'
            for each in self._list:
                array = np.asarray(each)
                str = str + np.array2string(array) \
                      + '\n-----------------------------------------\n'
            return str.rstrip("\n")  # Remove trailing newline character
        else:
            return 'No matrix found'
