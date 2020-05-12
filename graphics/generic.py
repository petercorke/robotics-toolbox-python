"""
Generic Drawing Functions

"""

from vpython import *
from numpy import sign
from math import sqrt

# TODO (NEXT TIME OPENING)
#  1. DONE - Comments
#  2. DONE - Read other TODOs
#  3. Split into diff files
#  4. Push to github

# TODO: Additions:
#  1. Keyboard input to maneuver around 3D map
#  2. Webpage buttons that:
#       a. Hide/Show labels?
#       b. Toggle between relative/static grid? if plausible
#       c. Reset camera settings?


def init_canvas(height=500, width=1000, title='', caption='', grid=True):
    """
    Set up the scene with initial conditions.
        - White background
        - Width, height
        - Title, caption
        - Axes drawn (if applicable)
    
    :param height: Height of the canvas on screen (Pixels), defaults to 500.
    :type height: int, optional
    :param width: Width of the canvas on screen (Pixels), defaults to 1000.
    :type width: int, optional
    :param title: Title of the plot. Gets displayed above canvas, defaults to ''.
    :type title: str, optional
    :param caption: Caption (subtitle) of the plot. Gets displayed below the canvas, defaults to ''.
    :type caption: str, optional
    :param grid: Whether a grid should be displayed in the plot, defaults to True.
    :type grid: bool, optional
    :return: Returns an array of the grid and the numbers displayed along the grid.
    :rtype: Array of vpython objects: [compound object, array of labels]
    """

    # Apply the settings
    scene.background = color.white
    scene.width = width
    scene.height = height
    scene.autoscale = False

    if title != '':
        scene.title = title

    if caption != '':
        scene.caption = caption

    plot_grid = draw_grid()
    if not grid:
        # If no grid, turn the objects invisible
        plot_grid[0].visible = False
        for number in plot_grid[1]:
            number.visible = False

    return plot_grid


def update_grid(input_grid):
    """
    Update the grid axes and numbers if the camera position/rotation has changed.

    :param input_grid: The current grid being displayed
    :type input_grid: Array: [compound_object, array_of_labels]
    :return: Returns an array of the grid and the numbers displayed along the grid.
    :rtype: Array: [compound_object, array_of_labels]
    """

    # Obtain the new camera settings
    new_camera_pos = scene.camera.pos
    new_camera_axes = scene.camera.axis

    # TODO: Put these funcs in a class to have camera data saved
    old_camera_pos = vector(0, 0, 0)
    old_camera_axes = vector(0, 0, 0)

    # If camera is different to previous: update
    if (not new_camera_axes.equals(old_camera_axes)) or (not new_camera_pos.equals(old_camera_pos)):
        # Update old positions
        old_camera_pos = new_camera_pos
        old_camera_axes = new_camera_axes

        # Delete old grid
        input_grid[0].visible = False
        for number in input_grid[1]:
            number.visible = False
        del input_grid

        # Return new grid
        return draw_grid()
    # Else return current grid
    else:
        return input_grid


def draw_grid():
    """
    Display grids along the x, y, z axes.

    :return: Returns an array of the grid and the numbers displayed along the grid.
    :rtype: Array: [compound_object, array_of_labels]
    """

    num_squares = 10  # Length of the grid in each direction (in units)
    relative_cam = True  # Whether the grid follows the camera rotation and movement

    the_grid = create_grid(relative_cam, num_squares)
    the_numbers = create_grid_numbers(relative_cam, num_squares)

    return [the_grid, the_numbers]


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


def draw_reference_frame_axes(origin, x_axis_vector, x_axis_rotation):
    """
    Draw x, y, z axes from the given point.
    Each axis is represented in the objects reference frame.

    :param origin: 3D vector representing the point to draw the reference from at.
    :type origin: vpython.vector
    :param x_axis_vector: 3D vector representing the direction of the positive x axis.
    :type x_axis_vector: vpython.vector
    :param x_axis_rotation: Angle in radians to rotate the frame around the x-axis.
    :type x_axis_rotation: float
    :return: Compound object of the 3 axis arrows.
    :rtype: vpython.compound
    """

    # Create Basic Frame
    # Draw X Axis
    x_arrow = arrow(pos=origin, axis=vector(1, 0, 0), length=1, color=color.red)

    # Draw Y Axis
    y_arrow = arrow(pos=origin, axis=vector(0, 1, 0), length=1, color=color.green)

    # Draw Z Axis
    z_arrow = arrow(pos=origin, axis=vector(0, 0, 1), length=1, color=color.blue)

    # Combine all to manipulate together
    # Set origin to where axis converge (instead of the middle of the resulting object bounding box)
    frame_ref = compound([x_arrow, y_arrow, z_arrow], origin=origin)

    # Rotate frame around x, y, z axes as required
    # Set x-axis along required vector, and rotate around the x-axis to corresponding angle to align last two axes
    frame_ref.axis = x_axis_vector
    frame_ref.rotate(angle=x_axis_rotation)

    return frame_ref


def create_grid(bool_camera_relative, num_squares):
    """
    Draw a grid along each 3D plane, that is closest to the camera.

    :param bool_camera_relative: Whether to draw the axes at the camera focus point or at (0, 0, 0).
    :type bool_camera_relative: bool
    :param num_squares: How many unit squares to draw along the axis.
    :type num_squares: int
    :return: Vpython compound object of the three drawn axes.
    :rtype: vpython.compound
    """

    # Initial conditions
    xz_lines = []
    xy_lines = []
    yz_lines = []
    camera_axes = scene.camera.axis
    # Locate centre of axes
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

    # XZ plane
    for x_point in range(min_x_coord, max_x_coord + 1):
        # Draw a line across for each x coord, along the same y-axis, from min to max z coord
        xz_lines.append(create_line(vector(x_point, y_origin, min_z_coord), vector(x_point, y_origin, max_z_coord)))
    for z_point in range(min_z_coord, max_z_coord + 1):
        # Draw a line across each z coord, along the same y-axis, from min to max z coord
        xz_lines.append(create_line(vector(min_x_coord, y_origin, z_point), vector(max_x_coord, y_origin, z_point)))

    # XY plane
    for x_point in range(min_x_coord, max_x_coord + 1):
        # Draw a line across each x coord, along the same z-axis, from min to max y coord
        xy_lines.append(create_line(vector(x_point, min_y_coord, z_origin), vector(x_point, max_y_coord, z_origin)))
    for y_point in range(min_y_coord, max_y_coord + 1):
        # Draw a line across each y coord, along the same z-axis, from min to max x coord
        xy_lines.append(create_line(vector(min_x_coord, y_point, z_origin), vector(max_x_coord, y_point, z_origin)))

    # YZ plane
    for y_point in range(min_y_coord, max_y_coord + 1):
        # Draw a line across each y coord, along the same x-axis, from min to max z coord
        yz_lines.append(create_line(vector(x_origin, y_point, min_z_coord), vector(x_origin, y_point, max_z_coord)))
    for z_point in range(min_z_coord, max_z_coord + 1):
        # Draw a line across each z coord, along the same x-axis, from min to max y coord
        yz_lines.append(create_line(vector(x_origin, min_y_coord, z_point), vector(x_origin, max_y_coord, z_point)))

    # Compound the lines together into respective objects
    xz_plane = compound(xz_lines)
    xy_plane = compound(xy_lines)
    yz_plane = compound(yz_lines)

    # Combine all into one object
    grid = compound([xy_plane, xz_plane, yz_plane])

    return grid


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


def create_line(pos1, pos2):
    """
    Create a line from position 1 to position 2.

    :param pos1: 3D position of one end of the line.
    :type pos1: vpython.vector
    :param pos2: 3D position of the other end of the line.
    :type pos2: vpython.vector
    """

    # Length of the line using trigonometry
    line_len = sqrt(
        (pos2.x - pos1.x) ** 2 +
        (pos2.y - pos1.y) ** 2 +
        (pos2.z - pos1.z) ** 2
    )

    # Position of the line is the midpoint (centre) between the ends
    position = (pos1 + pos2) / 2

    # Axis direction of the line (to align the box (line) to intersect the two points)
    axis_dir = pos2 - pos1

    # Return a box of thin width and height to resemble a line
    thickness = 0.01
    return box(pos=position, axis=axis_dir, length=line_len, width=thickness, height=thickness, color=color.black)


# TODO: Remove/Move to new file
def testing_references():
    le = 0.8

    # Test 1 | Position (0, 0, 0), Axis (1, 0, 0), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(0, 0, 0), vector(1, 0, 0), radians(0))
    # Actual
    arrow(pos=vector(0, 0, 0), axis=vector(1, 0, 0), length=le, color=color.purple)

    # Test 2 | Position (1, 1, 1), Axis (0, 0, 1), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(1, 1, 1), vector(0, 0, 1), radians(0))
    # Actual
    arrow(pos=vector(1, 1, 1), axis=vector(0, 0, 1), length=le, color=color.purple)

    # Test 3 | Position (2, 2, 2), Axis (1, 0, 0), 30 deg rot
    # Drawn
    draw_reference_frame_axes(vector(2, 2, 2), vector(1, 0, 0), radians(30))
    # Actual
    arrow(pos=vector(2, 2, 2), axis=vector(1, 0, 0), length=le, color=color.purple).rotate(radians(30))

    # Test 4 | Position (3, 3, 3), Axis (1, 1, 1), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(3, 3, 3), vector(1, 1, 1), radians(0))
    # Actual
    arrow(pos=vector(3, 3, 3), axis=vector(1, 1, 1), length=le, color=color.purple)

    # Test 5 | Position (4, 4, 4), Axis (1, 1, 1), 30 deg rot
    # Drawn
    draw_reference_frame_axes(vector(4, 4, 4), vector(1, 1, 1), radians(30))
    # Actual
    arrow(pos=vector(4, 4, 4), axis=vector(1, 1, 1), length=le, color=color.purple).rotate(radians(30))

    # Test 6 | Position (5, 5, 5), Axis (2, -1, 4), No Rotation
    # Drawn
    draw_reference_frame_axes(vector(5, 5, 5), vector(2, -1, 4), radians(0))
    # Actual
    arrow(pos=vector(5, 5, 5), axis=vector(2, -1, 4), length=le, color=color.purple)

    # Test 7 | Position (6, 6, 6), Axis (2, -1, 4), 30 deg rot
    # Drawn
    draw_reference_frame_axes(vector(6, 6, 6), vector(2, -1, 4), radians(30))
    # Actual
    arrow(pos=vector(6, 6, 6), axis=vector(2, -1, 4), length=le, color=color.purple).rotate(radians(30))


# TODO: Remove after testing
if __name__ == "__main__":
    print("Graphics Test")
    # testing_references()
    main_grid = init_canvas()
    while True:
        sleep(1)
        main_grid = update_grid(main_grid)
