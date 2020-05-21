from vpython import radians


def wrap_to_pi(angle):
    angle = angle % radians(360)
    if angle > radians(180):
        angle -= radians(360)
    return angle
