from vpython import radians, vector

"""
global variables that can be used to easily reference X, Y, and Z axes directions.
"""
x_axis_vector = vector(1, 0, 0)
y_axis_vector = vector(0, 1, 0)
z_axis_vector = vector(0, 0, 1)


def wrap_to_pi(angle_type, angle):
    """
    Wrap the given angle (deg or rad) to [-pi pi]

    :param angle_type: String of whether the angle is deg or rad
    :type angle_type: `str`
    :param angle: The angle to wrap
    :type angle: `float`
    :raises ValueError: Throws the error if the given string is not "deg" or "rad"
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
