"""
Generic Drawing Functions

"""

from vpython import *
from numpy import sign
from math import sqrt


# TODO: Additions:
#  1. Keyboard input to maneuver around 3D map
#  2. Webpage buttons that:
#       a. Hide/Show labels
#       b. Toggle between relative/static grid
#       c. Reset camera settings


def init_canvas(height=500, width=1000, title='', caption='', grid=True):
    """
    Set up the scene with initial conditions.
        - White background
        - Width, height
        - Title, caption
        - Axes drawn (if applicable)
    TODO: Add others as available
    
    @type height: int (pixels)
    @param height: Height of the canvas on screen.
    @type width: int (pixels)
    @param width: Width of the canvas on screen.
    @type title: str
    @param title: Title of the plot.
    @type caption: str
    @param caption: Caption (subtitle) of the plot.
    @type grid: bool
    @param grid: Whether a grid should be displayed in the plot.
    """
    scene.background = color.white
    scene.width = width
    scene.height = height
    scene.autoscale = False
    if title != '':
        scene.title = title
    if caption != '':
        scene.caption = caption
    if grid:
        plot_grid = draw_grid()
        draw_reference_frame_axes(vector(0, 0, 0), vector(1, 0, 0), radians(0))
    else:
        plot_grid = box().visible = False

    return plot_grid


def draw_grid():
    """
    Display grids along the x, y, z axes.
    """
    # TODO: Only update if camera has moved/zoomed/rotated
    num_squares = 10
    relative_cam = True

    the_grid = create_grid(relative_cam, num_squares)
    the_numbers = create_grid_numbers(relative_cam, num_squares)

    return [the_grid, the_numbers]


def draw_label(label_text, label_position):
    """
    Display a label at a given position
    """
    # TODO: Sanity check param input

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
    Display a label at a given position
    """
    # TODO: Sanity check param input

    distance_from_center = mag(scene.center - scene.camera.pos)

    # Far away = smaller, closer = larger (up to a min (20) and max (40))
    # Typically 5->20 units away
    # (eqn and limits modified to suit display) = -1.3333 * distance_from_center + 46.6667
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
    """

    # TODO: Remove once verified
    # Place a box to verify 90 deg angles
    L = 0.3
    new_box = box(pos=origin, axis=x_axis_vector, length=L, width=L, height=L).rotate(x_axis_rotation)

    # TODO: Ensure parameter input is sufficient

    # Create Basic Frame
    # Draw X Axis
    x_arrow = arrow(pos=origin, axis=vector(1, 0, 0), length=1, color=color.red)  # In direction of tooltip/object

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

    return


def create_grid(bool_camera_relative, num_squares):
    """
    Draw a grid along each 3D plane
    """
    # TODO: create an update_grid() that updates grid min/max relative to camera position(/rotation?).
    #  And/or ensure grid walls are at the end of visible range (relative to camera axis (find a relative origin to
    #  draw from)). May have to be in this one function?? Can't uncompound objects?
    #  Another option is to have grid in constant position relative to camera, except the numbers on the grid change
    #  relative to position (place grids against walls (like MATLAB 3D plots))

    # TODO: Change array input to create_line to vector inputs. Vector has functions to utilise (mag, etc)

    # TODO: Using scene.camera.axis, work out whether each plane should be +ve or -ve numbers
    #   AXIS |  GRID | XZ | XY | YZ
    #  -,-,- | +,+,+ | ++ | ++ | ++
    #  -,-,+ | +,+,- | +- | ++ | +-
    #  -,+,- | +,-,+ | ++ | +- | -+
    #  -,+,+ | +,-,- | +- | +- | --
    #  +,-,- | -,+,+ | -+ | -+ | ++
    #  +,-,+ | -,+,- | -- | -+ | +-
    #  +,+,- | -,-,+ | -+ | -- | -+
    #  +,+,+ | -,-,- | -- | -- | --

    # Initial conditions
    xz_lines = []
    xy_lines = []
    yz_lines = []
    if bool_camera_relative:
        x_origin, y_origin, z_origin = round(scene.center.x), round(scene.center.y), round(scene.center.z)
    else:
        x_origin, y_origin, z_origin = 0, 0, 0
    camera_axes = scene.camera.axis

    # min = -num_squares or 0 around default position
    # max = +num_squares or 0 around default position
    min_x_coord = x_origin + int(-(num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2))
    max_x_coord = x_origin + int((num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2))

    min_y_coord = y_origin + int(-(num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2))
    max_y_coord = y_origin + int((num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2))

    min_z_coord = z_origin + int(-(num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2))
    max_z_coord = z_origin + int((num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2))

    # XZ plane
    for x_point in range(min_x_coord, max_x_coord + 1):
        xz_lines.append(create_line([x_point, y_origin, min_z_coord], [x_point, y_origin, max_z_coord]))  # x-axis
    for z_point in range(min_z_coord, max_z_coord + 1):
        xz_lines.append(create_line([min_x_coord, y_origin, z_point], [max_x_coord, y_origin, z_point]))  # z-axis

    # XY plane
    for x_point in range(min_x_coord, max_x_coord + 1):
        xy_lines.append(create_line([x_point, min_y_coord, z_origin], [x_point, max_y_coord, z_origin]))  # x-axis
    for y_point in range(min_y_coord, max_y_coord + 1):
        xy_lines.append(create_line([min_x_coord, y_point, z_origin], [max_x_coord, y_point, z_origin]))  # y-axis

    # YZ plane
    for y_point in range(min_y_coord, max_y_coord + 1):
        yz_lines.append(create_line([x_origin, y_point, min_z_coord], [x_origin, y_point, max_z_coord]))  # y-axis
    for z_point in range(min_z_coord, max_z_coord + 1):
        yz_lines.append(create_line([x_origin, min_y_coord, z_point], [x_origin, max_y_coord, z_point]))  # z-axis

    # Compound the lines together into one object
    xz_plane = compound(xz_lines)
    xy_plane = compound(xy_lines)
    yz_plane = compound(yz_lines)

    # Combine all into one object
    grid = compound([xy_plane, xz_plane, yz_plane])

    return grid


def create_grid_numbers(bool_camera_relative, num_squares):
    padding = 0.25
    camera_axes = scene.camera.axis

    if bool_camera_relative:
        x_origin, y_origin, z_origin = round(scene.center.x), round(scene.center.y), round(scene.center.z)
    else:
        x_origin, y_origin, z_origin = 0, 0, 0

    # min = -num_squares or 0 around default position
    # max = +num_squares or 0 around default position
    min_x_coord = x_origin + int(-(num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2))
    max_x_coord = x_origin + int((num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2))

    min_y_coord = y_origin + int(-(num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2))
    max_y_coord = y_origin + int((num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2))

    min_z_coord = z_origin + int(-(num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2))
    max_z_coord = z_origin + int((num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2))

    nums = []

    # X plane
    for x_pos in range(min_x_coord, max_x_coord + sign(max_x_coord)):
        nums.append(draw_text(str(x_pos), vector(x_pos + padding, y_origin + padding, z_origin)))
    if (sign(camera_axes.x) * -1) > 0:
        nums.append(draw_text("X", vector(max_x_coord + 1, y_origin, z_origin)))
    else:
        nums.append(draw_text("X", vector(min_x_coord - 1, y_origin, z_origin)))

    # Y plane
    for y_pos in range(min_y_coord, max_y_coord + sign(max_y_coord)):
        nums.append(draw_text(str(y_pos), vector(x_origin, y_pos + padding, z_origin + padding)))
    if (sign(camera_axes.y) * -1) > 0:
        nums.append(draw_text("Y", vector(x_origin, max_y_coord + 1, z_origin)))
    else:
        nums.append(draw_text("Y", vector(x_origin, min_y_coord - 1, z_origin)))

    # Z plane
    for z_pos in range(min_z_coord, max_z_coord + sign(max_z_coord)):
        nums.append(draw_text(str(z_pos), vector(x_origin, y_origin - padding, z_pos + padding)))
    if (sign(camera_axes.z) * -1) > 0:
        nums.append(draw_text("Z", vector(x_origin, y_origin, max_z_coord + 1)))
    else:
        nums.append(draw_text("Z", vector(x_origin, y_origin, min_z_coord - 1)))

    return nums


def create_line(pos1, pos2):
    """
    Create a line from position 1 to position 2
    @type pos1: int array
    @param pos1: 3D position of one end of the line
    @type pos2: int array
    @param pos2: 3D position of the other end of the line
    """
    # TODO: Insert checks to ensure 3D points given (e.g. index out of bounds error if given 2D points)
    # TODO: Maybe add a colour input??

    # Length of the line using trigonometry
    line_len = sqrt(
        (pos2[0] - pos1[0]) ** 2 +
        (pos2[1] - pos1[1]) ** 2 +
        (pos2[2] - pos1[2]) ** 2
    )

    # Position of the line is the midpoint (centre) between the ends
    position = vector(
        (pos1[0] + pos2[0]) / 2,
        (pos1[1] + pos2[1]) / 2,
        (pos1[2] + pos2[2]) / 2,
    )

    # Axis direction of the line (to align the box (line) to intersect the two points
    axis_dir = vector(
        (pos2[0] - pos1[0]),
        (pos2[1] - pos1[1]),
        (pos2[2] - pos1[2])
    )

    # Return a box of thin width and height to resemble a line
    thickness = 0.01
    return box(pos=position, axis=axis_dir, length=line_len, width=thickness, height=thickness, color=color.black)


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
    globalvar_grid = init_canvas(title="Testing (TITLE)", caption="Test Environment (CAPTION)", grid=True)
    # testing_references()
    while True:
        sleep(1)
        globalvar_grid[0].visible = False
        for num in globalvar_grid[1]:
            num.visible = False
        globalvar_grid = draw_grid()
