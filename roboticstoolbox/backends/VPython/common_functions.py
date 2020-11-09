#!/usr/bin/env python
"""
@author Micah Huth
"""


from vpython import radians, vector
from numpy import array
from spatialmath import SE3

"""
global variables that can be used to easily reference X, Y, and Z axis
directions.
"""
x_axis_vector = vector(1, 0, 0)
y_axis_vector = vector(0, 1, 0)
z_axis_vector = vector(0, 0, 1)


def get_pose_x_vec(se3_obj):  # pragma nocover
    """
    Convert SE3 details to VPython vector format

    :param se3_obj: SE3 pose and orientation object
    :type se3_obj: class:`spatialmath.pose3d.SE3`
    :return: VPython vector representation of the X orientation
    :rtype: class:`vpython.vector`
    """
    data = se3_obj.n
    return vector(data[0], data[1], data[2])


def get_pose_y_vec(se3_obj):  # pragma nocover
    """
    Convert SE3 details to VPython vector format

    :param se3_obj: SE3 pose and orientation object
    :type se3_obj: class:`spatialmath.pose3d.SE3`
    :return: VPython vector representation of the Y orientation
    :rtype: class:`vpython.vector`
    """
    data = se3_obj.o
    return vector(data[0], data[1], data[2])


def get_pose_z_vec(se3_obj):  # pragma nocover
    """
    Convert SE3 details to VPython vector format

    :param se3_obj: SE3 pose and orientation object
    :type se3_obj: class:`spatialmath.pose3d.SE3`
    :return: VPython vector representation of the Z orientation
    :rtype: class:`vpython.vector`
    """
    data = se3_obj.a
    return vector(data[0], data[1], data[2])


def get_pose_pos(se3_obj):  # pragma nocover
    """
    Convert SE3 details to VPython vector format

    :param se3_obj: SE3 pose and orientation object
    :type se3_obj: `spatialmath.pose3d.SE3`
    :return: VPython vector representation of the position
    :rtype: class:`vpython.vector`
    """
    data = se3_obj.t
    return vector(data[0], data[1], data[2])


def vpython_to_se3(graphic_object):  # pragma nocover
    """
    This function will take in a graphics object and output it's pose as an
    SE3 object

    :param graphic_object: A VPython graphic object
    :type graphic_object: class:`vpython.object`
    :return: SE3 representation of the pose
    :rtype: class:`spatialmath.pose3d.SE3`
    """
    # Get the x, y, z axes and position
    x_vec = graphic_object.axis
    y_vec = graphic_object.up
    z_vec = x_vec.cross(y_vec)
    pos = graphic_object.pos

    # Form a numpy array
    T = array([
        [x_vec.x, y_vec.x, z_vec.x, pos.x],
        [x_vec.y, y_vec.y, z_vec.y, pos.y],
        [x_vec.z, y_vec.z, z_vec.z, pos.z],
        [0,       0,       0,     1]
    ])

    return SE3(T)


def wrap_to_pi(angle_type, angle):  # pragma nocover
    """
    Wrap the given angle (deg or rad) to [-pi pi]

    :param angle_type: String of whether the angle is deg or rad
    :type angle_type: `str`
    :param angle: The angle to wrap
    :type angle: `float`
    :raises ValueError: Throws the error if the given string is not "deg"
        or "rad"
    :return: The wrapped angle
    :rtype: `float`
    """
    if angle_type == "deg":
        angle = angle % 360
        if angle > 180:
            angle -= 360

    elif angle_type == "rad":
        angle = angle % radians(360)
        if angle > radians(180):
            angle -= radians(360)

    else:
        raise ValueError('angle_type must be "deg" or "rad"')

    return angle


def close_localhost_session(canvas):  # pragma nocover
    """
    Terminate the local host session through JavaScript

    :param canvas: The scene to append the JS to the caption
    :type canvas:
        class:`roboticstoolbox.backends.VPython.graphics_canvas.GraphicsCanvas3D`,
        class:`roboticstoolbox.backends.VPython.graphics_canvas.GraphicsCanvas2D`
    """

    canvas.scene.append_to_caption('''
        <script type="text/javascript">
            close();
        </script>
        ''')
