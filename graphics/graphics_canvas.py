from vpython import scene, color, arrow, compound
from graphics.common_functions import *
from graphics.graphics_grid import GraphicsGrid


def init_canvas(height=500, width=1000, title='', caption='', grid=True):
    """
    Set up the scene with initial conditions.
        - White background
        - Width, height
        - Title, caption
        - Axes drawn (if applicable)

    :param height: Height of the canvas on screen (Pixels), defaults to 500.
    :type height: `int`, optional
    :param width: Width of the canvas on screen (Pixels), defaults to 1000.
    :type width: `int`, optional
    :param title: Title of the plot. Gets displayed above canvas, defaults to ''.
    :type title: `str`, optional
    :param caption: Caption (subtitle) of the plot. Gets displayed below the canvas, defaults to ''.
    :type caption: `str`, optional
    :param grid: Whether a grid should be displayed in the plot, defaults to `True`.
    :type grid: `bool`, optional
    :return: The graphics grid object for use outside canvas creation
    :rtype: class:`GraphicsGrid`
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

    convert_grid_to_z_up()

    graphics_grid = GraphicsGrid()
    if not grid:
        graphics_grid.set_visibility(False)

    return graphics_grid


def convert_grid_to_z_up():
    """
    Rotate the camera so that +z is up
    (Default vpython scene is +y up)
    """
    # Place camera at center to aid rotations
    scene.camera.pos = vector(0, 0, 0)
    # Rotate about y then x axis
    # (Camera defaults looking in -z direction -> (0, 0, -1))
    scene.camera.rotate(radians(90), axis=y_axis_vector)
    scene.camera.rotate(radians(90), axis=x_axis_vector)
    # Place the camera in the + axes
    scene.camera.pos = vector(10, 10, 10)
    scene.camera.axis = -scene.camera.pos
    return


def draw_reference_frame_axes(se3_pose):
    """
    Draw x, y, z axes from the given point.
    Each axis is represented in the objects reference frame.

    :param se3_pose: SE3 pose representation of the reference frame
    :type se3_pose: class:`spatialmath.pose3d.SE3`
    :return: Compound object of the 3 axis arrows.
    :rtype: class:`vpython.compound`
    """

    origin = get_pose_pos(se3_pose)
    x_axis = get_pose_x_vec(se3_pose)
    y_axis = get_pose_y_vec(se3_pose)

    # Create Basic Frame
    # Draw X Axis
    x_arrow = arrow(pos=origin, axis=x_axis_vector, length=0.25, color=color.red)

    # Draw Y Axis
    y_arrow = arrow(pos=origin, axis=y_axis_vector, length=0.25, color=color.green)

    # Draw Z Axis
    z_arrow = arrow(pos=origin, axis=z_axis_vector, length=0.25, color=color.blue)

    # Combine all to manipulate together
    # Set origin to where axis converge (instead of the middle of the resulting object bounding box)
    frame_ref = compound([x_arrow, y_arrow, z_arrow], origin=origin)

    # Set frame axes
    frame_ref.axis = x_axis
    frame_ref.up = y_axis

    return frame_ref
