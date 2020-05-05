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

    ~Other if available later~
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
    return


def draw_grid():
    return


def draw_label():
    return


def draw_reference_frame_axes():
    return
