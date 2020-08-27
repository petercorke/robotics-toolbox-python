from vpython import color, label, mag, vector
from numpy import sign, arange


def draw_label(label_text, label_position, scene):
    """
    Display a label at a given position, with borders and lines

    :param label_text: String of text to be written on the label.
    :type label_text: `str`
    :param label_position: 3D vector position to draw the label at.
    :type label_position: class:`vpython.vector`
    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: The created label object.
    :rtype: class:`vpython.label`
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
        canvas=scene,
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


def draw_text(label_text, label_position, scene):
    """
    Display a label at a given position, without borders or lines.

    :param label_text: String of text to be written on the label.
    :type label_text: `str`
    :param label_position: 3D vector position to draw the label at.
    :type label_position: class:`vpython.vector`
    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :return: The created label object.
    :rtype: class:`vpython.label`
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
        canvas=scene,
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


def update_grid_numbers(focal_point, numbers_list, num_squares, scale, scene):
    """
    Draw the grid numbers along the xyz axes.

    :param focal_point: The focus point of the camera to draw the grid about
    :type focal_point: `list`
    :param numbers_list: A reference to a list of the labels that gets updated.
    :type numbers_list: `list`
    :param num_squares: How many unit squares to draw along the axis.
    :type num_squares: `int`
    :param scale: The scaled length of 1 square unit
    :type scale: `float`
    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    """

    # Initial conditions
    padding = 0.25  # Padding to not draw numbers on top of lines.
    camera_axes = scene.camera.axis
    # Locate center of the axes
    x_origin, y_origin, z_origin = focal_point[0], focal_point[1], focal_point[2]

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
    min_x_coord = x_origin + int(-(num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2)) * scale
    max_x_coord = x_origin + int((num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2)) * scale

    min_y_coord = y_origin + int(-(num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2)) * scale
    max_y_coord = y_origin + int((num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2)) * scale

    min_z_coord = z_origin + int(-(num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2)) * scale
    max_z_coord = z_origin + int((num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2)) * scale

    x_coords = arange(min_x_coord, max_x_coord + scale, scale)
    y_coords = arange(min_y_coord, max_y_coord + scale, scale)
    z_coords = arange(min_z_coord, max_z_coord + scale, scale)

    # If input is empty, append new, otherwise update current
    append = len(numbers_list) == 0
    # Dimensions don't change between updates, so indexing shall remain the same
    index = 0

    # X plane
    for x_pos in x_coords:
        # Draw the corresponding unit number at each x coordinate
        txt = str(x_pos)
        pos = vector(x_pos + padding, y_origin + padding, z_origin)
        if append:
            numbers_list.append(draw_text(txt, pos, scene))
        else:
            numbers_list[index].text = txt
            numbers_list[index].pos = pos
            index += 1
    # Draw the axis label at either the positive or negative side away from center
    # If sign = -1, draw off max side, if sign = 0 or 1, draw off negative side
    txt = "X"
    if (sign(camera_axes.x) * -1) > 0:
        pos = vector(max_x_coord + 1, y_origin, z_origin)
    else:
        pos = vector(min_x_coord - 1, y_origin, z_origin)
    if append:
        numbers_list.append(draw_text(txt, pos, scene))
    else:
        numbers_list[index].text = txt
        numbers_list[index].pos = pos
        index += 1

    # Y plane
    for y_pos in y_coords:
        # Draw the corresponding unit number at each x coordinate
        txt = str(y_pos)
        pos = vector(x_origin, y_pos + padding, z_origin + padding)
        if append:
            numbers_list.append(draw_text(txt, pos, scene))
        else:
            numbers_list[index].text = txt
            numbers_list[index].pos = pos
            index += 1
    # Draw the axis label at either the positive or negative side away from center
    # If sign = -1, draw off max side, if sign = 0 or 1, draw off negative side
    txt = "Y"
    if (sign(camera_axes.y) * -1) > 0:
        pos = vector(x_origin, max_y_coord + 1, z_origin)
    else:
        pos = vector(x_origin, min_y_coord - 1, z_origin)
    if append:
        numbers_list.append(draw_text(txt, pos, scene))
    else:
        numbers_list[index].text = txt
        numbers_list[index].pos = pos
        index += 1

    # Z plane
    for z_pos in z_coords:
        # Draw the corresponding unit number at each x coordinate
        txt = str(z_pos)
        pos = vector(x_origin, y_origin - padding, z_pos + padding)
        if append:
            numbers_list.append(draw_text(txt, pos, scene))
        else:
            numbers_list[index].text = txt
            numbers_list[index].pos = pos
            index += 1
    # Draw the axis label at either the positive or negative side away from center
    # If sign = -1, draw off max side, if sign = 0 or 1, draw off negative side
    txt = "Z"
    if (sign(camera_axes.z) * -1) > 0:
        pos = vector(x_origin, y_origin, max_z_coord + 1)
    else:
        pos = vector(x_origin, y_origin, min_z_coord - 1)
    if append:
        numbers_list.append(draw_text(txt, pos, scene))
    else:
        numbers_list[index].text = txt
        numbers_list[index].pos = pos
        index += 1
