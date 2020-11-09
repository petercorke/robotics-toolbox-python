#!/usr/bin/env python
"""
@author Micah Huth
"""


from vpython import color, label, mag, vector
from numpy import sign, arange


def get_text_size(scene):  # pragma nocover
    """
    Determine the text size based on zoom distance

    :param scene: The scene in which the camera to be used resides.
    :type scene: `vpython.canvas`
    """
    # Distance of camera from focus point to determine text size
    distance_from_center = mag(scene.center - scene.camera.pos)

    # Eq generated from data (Using 3rd order polynomial)
    #  D  | size
    #  0  |   5
    # 0.5 |   8
    #  1  |  10
    #  2  |  12
    #  3  |  14
    #  5  |  15
    val = 0.1114 * distance_from_center**3 - 1.336 * distance_from_center**2 \
        + 5.8666 * distance_from_center + 5.1711

    return min(max(val, 10), 15)  # Return val between 10 and 15


def draw_label(label_text, label_position, scene):  # pragma nocover
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


def draw_text(label_text, label_position, scene):  # pragma nocover
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
    label_height = get_text_size(scene)
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


def update_grid_numbers(
        focal_point, numbers_list, num_squares,
        scale, is_3d, scene):  # pragma nocover
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
    :param is_3d: Whether the grid is 3D or not
    :type is_3d: `bool`
    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    """

    # Initial conditions
    padding = 0.25  # Padding to not draw numbers on top of lines.
    camera_axes = scene.camera.axis
    # Locate center of the axes
    x_origin, y_origin, z_origin = focal_point[0], focal_point[1], \
        focal_point[2]

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
    min_x_coord = round(x_origin + (-(num_squares / 2) +
                                    (sign(camera_axes.x) * -1)
                                    * (num_squares / 2)) * scale, 2)
    max_x_coord = round(x_origin + ((num_squares / 2) +
                                    (sign(camera_axes.x) * -1)
                                    * (num_squares / 2)) * scale, 2)

    min_y_coord = round(y_origin + (-(num_squares / 2) +
                                    (sign(camera_axes.y) * -1)
                                    * (num_squares / 2)) * scale, 2)
    max_y_coord = round(y_origin + ((num_squares / 2) +
                                    (sign(camera_axes.y) * -1)
                                    * (num_squares / 2)) * scale, 2)

    min_z_coord = round(z_origin + (-(num_squares / 2) +
                                    (sign(camera_axes.z) * -1)
                                    * (num_squares / 2)) * scale, 2)
    max_z_coord = round(z_origin + ((num_squares / 2) +
                                    (sign(camera_axes.z) * -1)
                                    * (num_squares / 2)) * scale, 2)

    x_coords = arange(min_x_coord, max_x_coord + scale, scale)
    y_coords = arange(min_y_coord, max_y_coord + scale, scale)
    z_coords = arange(min_z_coord, max_z_coord + scale, scale)

    # If the grid has given too many objects
    if len(x_coords) > num_squares + 1:
        x_coords = x_coords[0:num_squares + 1]
    if len(y_coords) > num_squares + 1:
        y_coords = y_coords[0:num_squares + 1]
    if len(z_coords) > num_squares + 1:
        z_coords = z_coords[0:num_squares + 1]

    # Compound origins are in the middle of the bounding boxes.
    # Thus new pos will be between max and min.
    x_middle = x_coords.mean()
    y_middle = y_coords.mean()
    z_middle = z_coords.mean()

    # If input is empty, append new, otherwise update current
    append = len(numbers_list) == 0
    # Dimensions don't change between updates, so indexing shall
    # remain the same
    index = 0

    # X plane
    for x_pos in x_coords:
        # Draw the corresponding unit number at each x coordinate
        txt = "{:.2f}".format(x_pos)
        if is_3d:
            if (sign(camera_axes.y) * -1) > 0:
                pos = vector(x_pos, max_y_coord + padding, z_origin)
            else:
                pos = vector(x_pos, min_y_coord - padding, z_origin)
        else:
            pos = vector(x_pos, y_origin - padding, z_origin)
        if append:
            numbers_list.append(draw_text(txt, pos, scene))
            numbers_list[len(numbers_list)-1].height = get_text_size(scene)
        else:
            numbers_list[index].text = txt
            numbers_list[index].pos = pos
            numbers_list[index].height = get_text_size(scene)
            index += 1
    # Draw the axis label at the centre of the axes numbers
    txt = "X"
    if (sign(camera_axes.y) * -1) > 0:
        x = x_middle
        y = max_y_coord + scale * 2
        pos = vector(x, y, z_origin)
    else:
        x = x_middle
        y = min_y_coord - scale * 2
        pos = vector(x, y, z_origin)
    if append:
        numbers_list.append(draw_text(txt, pos, scene))
        numbers_list[len(numbers_list) - 1].height = get_text_size(scene)
    else:
        numbers_list[index].text = txt
        numbers_list[index].pos = pos
        numbers_list[index].height = get_text_size(scene)
        index += 1

    # Y plane
    for y_pos in y_coords:
        # Draw the corresponding unit number at each x coordinate
        txt = "{:.2f}".format(y_pos)
        if is_3d:
            if (sign(camera_axes.x) * -1) > 0:
                pos = vector(max_x_coord + padding, y_pos, z_origin)
            else:
                pos = vector(min_x_coord - padding, y_pos, z_origin)
        else:
            pos = vector(x_origin - padding, y_pos, z_origin)
        if append:
            numbers_list.append(draw_text(txt, pos, scene))
            numbers_list[len(numbers_list) - 1].height = get_text_size(scene)
        else:
            numbers_list[index].text = txt
            numbers_list[index].pos = pos
            numbers_list[index].height = get_text_size(scene)
            index += 1
    # Draw the axis label at the centre of the axes numbers
    txt = "Y"
    if (sign(camera_axes.x) * -1) > 0:
        x = max_x_coord + scale * 2
        y = y_middle
        pos = vector(x, y, z_origin)
    else:
        x = min_x_coord - scale * 2
        y = y_middle
        pos = vector(x, y, z_origin)

    if append:
        numbers_list.append(draw_text(txt, pos, scene))
        numbers_list[len(numbers_list) - 1].height = get_text_size(scene)
    else:
        numbers_list[index].text = txt
        numbers_list[index].pos = pos
        numbers_list[index].height = get_text_size(scene)
        index += 1

    if not is_3d:
        return

    # Z plane
    for z_pos in z_coords:
        # Draw the corresponding unit number at each x coordinate
        txt = "{:.2f}".format(z_pos)
        if (sign(camera_axes.x) * -1) > 0:
            pos = vector(max_x_coord + padding, y_origin, z_pos)
        else:
            pos = vector(min_x_coord - padding, y_origin, z_pos)
        if append:
            numbers_list.append(draw_text(txt, pos, scene))
            numbers_list[len(numbers_list) - 1].height = get_text_size(scene)
        else:
            numbers_list[index].text = txt
            numbers_list[index].pos = pos
            numbers_list[index].height = get_text_size(scene)
            index += 1
    # Draw the axis label at either the positive or negative side away from
    # center
    # If sign = -1, draw off max side, if sign = 0 or 1, draw off negative side
    txt = "Z"
    if (sign(camera_axes.x) * -1) > 0:
        pos = vector(max_x_coord + scale * 2, y_origin, z_middle)
    else:
        pos = vector(min_x_coord - scale * 2, y_origin, z_middle)
    if append:
        numbers_list.append(draw_text(txt, pos, scene))
        numbers_list[len(numbers_list) - 1].height = get_text_size(scene)
    else:
        numbers_list[index].text = txt
        numbers_list[index].pos = pos
        numbers_list[index].height = get_text_size(scene)
        index += 1
