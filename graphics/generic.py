"""
Generic Drawing Functions

"""

from vpython import *

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
