"""
Generic Drawing Functions

"""

from vpython import *
from math import sqrt

'''
1. init conditions of scene
2. draw axes
3. draw labels
4. draw reference frames
'''


def init_canvas(height=500, width=1000, title='', caption='', grid=False):
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
    if title != '':
        scene.title = title
    if caption != '':
        scene.caption = caption
    if grid:
        plot_grid = draw_grid()

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


def draw_grid():
    """
    Display grids along the x, y, z axes.
    """
    the_grid = create_grid()
    return the_grid


def draw_label():
    """
    Display a label at a given position
    """
    return


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


def create_grid():
    """
    Draw a grid along each 3D plane
    """
    # TODO: create an update_grid() that updates grid min/max relative to camera position(/rotation?).
    #  And/or ensure grid walls are at the end of visible range (relative to camera axis (find a relative origin to
    #  draw from)). May have to be in this one function?? Can't uncompound objects?
    #  Another option is to have grid in constant position relative to camera, except the numbers on the grid change
    #  relative to position (place grids against walls (like MATLAB 3D plots))

    # TODO: Change array input to create_line to vector inputs. Vector has functions to utilise (mag, etc)

    # Initial conditions
    lines = []
    min_coord = -5
    max_coord = 5

    # XZ plane
    for start_point in range(min_coord, max_coord+1):
        lines.append(create_line([start_point, 0, min_coord], [start_point, 0, max_coord]))  # x-axis
        lines.append(create_line([min_coord, 0, start_point], [max_coord, 0, start_point]))  # z-axis

    # Compound the lines together into one object
    xz_plane = compound(lines)

    # Clone the current grid for the other two axis
    xy_plane = xz_plane.clone()
    yz_plane = xz_plane.clone()

    # Rotate them to align with their corresponding grid
    xy_plane.rotate(angle=radians(90), axis=vector(1, 0, 0), origin=vector(0, 0, 0))
    yz_plane.rotate(angle=radians(90), axis=vector(0, 0, 1), origin=vector(0, 0, 0))

    # Combine all into one object
    grid = compound([xy_plane, xz_plane, yz_plane])

    return grid


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


# TODO: Remove after testing
if __name__ == "__main__":
    print("Graphics Test")
    init_canvas(grid=True)
