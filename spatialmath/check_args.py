# Created by: Aditya Dua
# 1 June, 2017

"""test_args module contains test for input arguments.
It checks if input argument passed by user is valid or not.
If any invalid data is found,
the called function in test_args returns false"""
import numpy as np
from . import pose
from . import super_pose


def is_mat_list(list_matrices):
    """is_mat_list checks(arg1) checks if arg1
    is a list containing numpy matrix data type elements or not.
    If not, False is returned."""
    flag = True
    if isinstance(list_matrices, list):
        for matrix in list_matrices:
            if not isinstance(matrix, np.matrix):
                flag = False
                # TODO Check for matrix dimensions?
    else:
        flag = False
    return flag


def unit_check(unit):
    if unit == 'rad' or unit == 'deg':
        pass
    else:
        raise AssertionError("Invalid unit value passed. Must be 'rad' or 'deg' only.")


def is_vector(item):
    assert isinstance(item, np.matrix)
    assert (item.shape[0] == 1 and item.shape[1] > 1) \
           or (item.shape[0] > 1 and item.shape[1] == 1)


# ------------------------------------------------------
#               TRANSFORMS CHECKS
# ------------------------------------------------------

def rpy2r(theta, order):
    pass  # TODO Implement this


def np2vtk(mat):
    pass  # TODO


def tr2angvec(tr, unit):
    pass  # TODO


def tr2eul(tr, unit, flip):
    pass  # TODO


def tr2rpy(tr, unit, order):
    pass  # TODO


# ------------------------------------------------------
#                SUPER POSE CHECKS
# ------------------------------------------------------
def valid_pose(obj):
    # TODO -- Check if its a valid pose object
    assert isinstance(obj, super_pose.SuperPose)


def super_pose_appenditem(obj, item):
    valid_pose(obj)
    if isinstance(item, super_pose.SuperPose):
        assert type(obj) is type(item)
    elif isinstance(item, np.matrix):
        # TODO valid matrix check ?
        pass
    else:
        raise AssertionError('Invalid data type of item to append. '
                             'Data types allowed: numpy matrix and super_pose.SuperPose')


def super_pose_multiply_check(obj, other):
    if isinstance(other, super_pose.SuperPose):
        assert type(obj) is type(other)
        assert (obj.length == other.length) \
               or (obj.length == 1 and other.length > 1) \
               or (obj.length > 1 and other.length == 1)
    elif isinstance(other, np.matrix):
        assert other.shape[1] == 1  # Should be vector
        assert obj.shape[1] == other.shape[0]  # Inner Dimensions must match


def super_pose_divide_check(obj, other):
    assert type(obj) is type(other)
    assert obj.length == other.length or obj.length == 1 or other.length == 1


def super_pose_add_sub_check(obj, other):
    valid_pose(obj)
    valid_pose(other)
    assert type(obj) is type(other)
    assert obj.length == 1 and other.length == 1
    # TODO Allow pose list ?


def super_pose_subclass_check(obj, other):
    pass


# ----------------- POSE.SO2 CHECKS -------------------------
def so2_angle_list_check(ang_list):
    for each in ang_list:
        assert isinstance(each, int) or isinstance(each, float)


def so2_valid(obj):
    assert type(obj) is pose.SO2
    for each in obj:
        assert each.shape == (2, 2)
        assert abs(np.linalg.det(each) - 1) < np.spacing([1])[0]


def so2_input_matrix(args_in):
    det = np.linalg.det(args_in)


def so2_input_types_check(args_in):
    assert isinstance(args_in, np.matrix) \
           or isinstance(args_in, list) \
           or isinstance(args_in, int) \
           or isinstance(args_in, float) \
           or isinstance(args_in, pose.SO2) \
           or args_in is None


def so2_interp_check(obj1, obj2, s):
    assert type(obj2) is pose.SO2
    assert 0 <= s <= 1
    assert obj1.length == obj2.length  # TODO Intended behaviour ?


# ----------------- POSE.SO2 CHECKS ----------------------------
# ----------------- POSE.SE2 CHECKS ----------------------------

def se2_constructor_args_check(x, y, rot, theta, so2, se2):
    pass


def se2_valid(obj):
    # TODO
    pass


# ----------------- POSE.SE2 CHECKS ----------------------------

# ----------------- POSE.SO3 CHECKS ----------------------------

def so3_constructor_args_check(args_in):
    pass

# ----------------- POSE.SO3 CHECKS ----------------------------


# ----------------- SUPER POSE CHECKS ---------------------------
