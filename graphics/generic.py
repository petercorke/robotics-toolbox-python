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


def init_canvas(height=500, width=500, title='', caption='', grid=False):
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
        draw_grid()


def draw_grid():
    """
    Display grids along the x, y, z axes.

    The grid must be relative to the camera so that as the camera translates,
    all axes are still visible.
    Current zoom setting to show axis 5 units away.
    """
    # TODO: update a changeable zoom setting.

    create_grid()
    return


def draw_label():
    """
    Display a label at a given position
    """
    return


def draw_reference_frame_axes():
    """
    Draw x, y, z axes from the given point.
    Each axis is represented in the objects reference frame.
    """

    return


def create_grid():
    """
    Draw a grid along each 3D plane
    """
    # TODO: create an update_grid() that updates grid min/max relative to camera position. Ensure grid walls
    #  are at the end of visible range (relative to camera axis (find a relative origin to draw from)).
    #  May have to be in this one function?? Can't uncompound objects?

    # TODO: Reference spheres, remove later
    sphere(pos=vector(1, 0, 0), radius=0.25, color=color.red)
    sphere(pos=vector(0, 1, 0), radius=0.25, color=color.green)
    sphere(pos=vector(0, 0, 1), radius=0.25, color=color.blue)
    sphere(pos=vector(0, 0, 0), radius=0.25, color=color.black)

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
