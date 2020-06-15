from vpython import radians, vector

x_axis_vector = vector(1, 0, 0)
y_axis_vector = vector(0, 1, 0)
z_axis_vector = vector(0, 0, 1)


def wrap_to_pi(angle):
    angle = angle % radians(360)
    if angle > radians(180):
        angle -= radians(360)
    return angle
