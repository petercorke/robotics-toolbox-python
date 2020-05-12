from vpython import *
from numpy import sign


def draw_label(label_text, label_position):
    """
    Display a label at a given position, with borders and lines

    :param label_text: String of text to be written on the label.
    :type label_text: str
    :param label_position: 3D vector position to draw the label at.
    :type label_position: vpython.vector
    :return: The created label object.
    :rtype: vpython.label
    """

    # Custom settings for the label
    label_height = 10
    label_xoffset = 0
    label_yoffset = 50
    label_space = 20
    label_font = 'serif'
    label_text_colour = color.black
    label_line_color = color.black

    the_label = label(
        pos=label_position,
        text=label_text,
        height=label_height,
        xoffset=label_xoffset,
        yoffset=label_yoffset,
        space=label_space,
        font=label_font,
        color=label_text_colour,
        linecolor=label_line_color
    )

    return the_label


def draw_text(label_text, label_position):
    """
    Display a label at a given position, without borders or lines.

    :param label_text: String of text to be written on the label.
    :type label_text: str
    :param label_position: 3D vector position to draw the label at.
    :type label_position: vpython.vector
    :return: The created label object.
    :rtype: vpython.label
    """

    # Distance of camera from focus point to determine text size
    distance_from_center = mag(scene.center - scene.camera.pos)

    # Far away = smaller text, closer = larger text (up to a min (20) and max (40))
    # Typically 5->20 units away
    # (eqn and limits modified to suit display better) = -1.3333 * distance_from_center + 46.6667
    label_height = -1.3333 * distance_from_center + 36.6667  # Calculate label height
    label_height = max(min(label_height, 35), 10)  # Limit to 10->35
    label_xoffset = 0
    label_yoffset = 0
    label_space = 0
    label_font = 'serif'
    label_text_colour = color.black
    label_line_color = color.white
    label_bg_opacity = 0
    label_linewidth = 0.1

    the_label = label(
        pos=label_position,
        text=label_text,
        height=label_height,
        xoffset=label_xoffset,
        yoffset=label_yoffset,
        space=label_space,
        font=label_font,
        color=label_text_colour,
        linecolor=label_line_color,
        opacity=label_bg_opacity,
        linewidth=label_linewidth
    )

    return the_label


def create_grid_numbers(bool_camera_relative, num_squares):
    """
    Draw the grid numbers along the xyz axes.

    :param bool_camera_relative: Whether to draw the axes at the camera focus point or at (0, 0, 0).
    :type bool_camera_relative: bool
    :param num_squares: How many unit squares to draw along the axis.
    :type num_squares: int
    :return: An array of labels
    :rtype: vpython.label array
    """

    # Initial conditions
    padding = 0.25  # Padding to not draw numbers on top of lines.
    camera_axes = scene.camera.axis
    # Locate center of the axes
    if bool_camera_relative:
        x_origin, y_origin, z_origin = round(scene.center.x), round(scene.center.y), round(scene.center.z)
    else:
        x_origin, y_origin, z_origin = 0, 0, 0

    #   CAMERA AXES |  DISPLAYED GRID | XZ PLANE | XY PLANE | YZ PLANE
    #      x,y,z    |      x,y,z      |   x,z    |    x,y   |    y,z
    #  -------------+-----------------+----------+----------+----------
    #      -,-,-    |      +,+,+      |   +,+    |    +,+   |    +,+
    #      -,-,+    |      +,+,-      |   +,-    |    +,+   |    +,-
    #      -,+,-    |      +,-,+      |   +,+    |    +,-   |    -,+
    #      -,+,+    |      +,-,-      |   +,-    |    +,-   |    -,-
    #      +,-,-    |      -,+,+      |   -,+    |    -,+   |    +,+
    #      +,-,+    |      -,+,-      |   -,-    |    -,+   |    +,-
    #      +,+,-    |      -,-,+      |   -,+    |    -,-   |    -,+
    #      +,+,+    |      -,-,-      |   -,-    |    -,-   |    -,-
    # min = -num_squares or 0, around the default position
    # max = +num_squares or 0, around the default position
    # e.g. at the origin, for negative axes: -10 -> 0, positive axes: 0 -> 10
    min_x_coord = x_origin + int(-(num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2))
    max_x_coord = x_origin + int((num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2))

    min_y_coord = y_origin + int(-(num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2))
    max_y_coord = y_origin + int((num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2))

    min_z_coord = z_origin + int(-(num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2))
    max_z_coord = z_origin + int((num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2))

    nums = []

    # X plane
    for x_pos in range(min_x_coord, max_x_coord + sign(max_x_coord)):
        # Draw the corresponding unit number at each x coordinate
        nums.append(draw_text(str(x_pos), vector(x_pos + padding, y_origin + padding, z_origin)))
    # Draw the axis label at either the positive or negative side
    if (sign(camera_axes.x) * -1) > 0:
        nums.append(draw_text("X", vector(max_x_coord + 1, y_origin, z_origin)))
    else:
        nums.append(draw_text("X", vector(min_x_coord - 1, y_origin, z_origin)))

    # Y plane
    for y_pos in range(min_y_coord, max_y_coord + sign(max_y_coord)):
        # Draw the corresponding unit number at each y coordinate
        nums.append(draw_text(str(y_pos), vector(x_origin, y_pos + padding, z_origin + padding)))
    # Draw the axis label at either the positive or negative side
    if (sign(camera_axes.y) * -1) > 0:
        nums.append(draw_text("Y", vector(x_origin, max_y_coord + 1, z_origin)))
    else:
        nums.append(draw_text("Y", vector(x_origin, min_y_coord - 1, z_origin)))

    # Z plane
    for z_pos in range(min_z_coord, max_z_coord + sign(max_z_coord)):
        # Draw the corresponding unit number at each z coordinate
        nums.append(draw_text(str(z_pos), vector(x_origin, y_origin - padding, z_pos + padding)))
    # Draw the axis label at either the positive or negative side
    if (sign(camera_axes.z) * -1) > 0:
        nums.append(draw_text("Z", vector(x_origin, y_origin, max_z_coord + 1)))
    else:
        nums.append(draw_text("Z", vector(x_origin, y_origin, min_z_coord - 1)))

    return nums
