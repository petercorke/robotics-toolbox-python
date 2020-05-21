from graphics.graphics_grid import *


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
    :param grid: Whether a grid should be displayed in the plot, defaults to `True`.
    :type grid: bool, optional
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

    graphics_grid = GraphicsGrid()
    graphics_grid.draw_grid()
    if not grid:
        graphics_grid.set_visibility(False)

    convert_grid_to_z_up()

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
    scene.camera.rotate(radians(90), axis=vector(0, 1, 0))
    scene.camera.rotate(radians(90), axis=vector(1, 0, 0))
    # Place the camera in the + axes
    scene.camera.pos = vector(10, 10, 10)
    scene.camera.axis = -scene.camera.pos
    return


def draw_reference_frame_axes(origin, x_axis_vector, x_axis_rotation):
    """
    Draw x, y, z axes from the given point.
    Each axis is represented in the objects reference frame.

    :param origin: 3D vector representing the point to draw the reference from at.
    :type origin: class:`vpython.vector`
    :param x_axis_vector: 3D vector representing the direction of the positive x axis.
    :type x_axis_vector: class:`vpython.vector`
    :param x_axis_rotation: Angle in radians to rotate the frame around the x-axis.
    :type x_axis_rotation: float
    :return: Compound object of the 3 axis arrows.
    :rtype: class:`vpython.compound`
    """

    # Create Basic Frame
    # Draw X Axis
    x_arrow = arrow(pos=origin, axis=vector(1, 0, 0), length=0.25, color=color.red)

    # Draw Y Axis
    y_arrow = arrow(pos=origin, axis=vector(0, 1, 0), length=0.25, color=color.green)

    # Draw Z Axis
    z_arrow = arrow(pos=origin, axis=vector(0, 0, 1), length=0.25, color=color.blue)

    # Combine all to manipulate together
    # Set origin to where axis converge (instead of the middle of the resulting object bounding box)
    frame_ref = compound([x_arrow, y_arrow, z_arrow], origin=origin)

    # Rotate frame around x, y, z axes as required
    # Set x-axis along required vector, and rotate around the x-axis to corresponding angle to align last two axes
    # NB: Set XY axis first, as vpython is +y up bias, objects rotate respective to this bias when setting axis
    frame_ref.axis = vector(x_axis_vector.x, x_axis_vector.y, 0)
    frame_ref.axis = x_axis_vector
    frame_ref.rotate(angle=x_axis_rotation)

    return frame_ref
