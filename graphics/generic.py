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
    TODO: update a changeable zoom setting.
    """
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
    l1 = create_line([-5, 0, 0], [5, 0, 0])
    sphere(pos=vector(-5, 0, 0), radius=0.5, color=color.red)
    sphere(pos=vector(5, 0, 0), radius=0.5, color=color.green)
    return


def create_line(pos1, pos2):
    """
    Create a line from position 1 to position 2
    @type pos1: int array
    @param pos1: 3D position of one end of the line
    @type pos2: int array
    @param pos2: 3D position of the other end of the line
    TODO: insert checks to ensure 3D points given
    """

    # Length of the line using trigonometry
    line_len = sqrt(
        (pos2[0] - pos1[0])**2 +
        (pos2[1] - pos1[1])**2 +
        (pos2[2] - pos1[2])**2
    )

    # Position of the line is the midpoint (centre) between the ends
    position = vector(
        (pos1[0] + pos2[0]) / 2,
        (pos1[1] + pos2[1]) / 2,
        (pos1[2] + pos2[2]) / 2,
    )

    # return a box of thin width and height to resemble a line
    return box(pos=position, length=line_len, width=0.01, height=0.01, color=color.black)


if __name__ == "__main__":
    print("Graphics Test")
    init_canvas(grid=True)
