from vpython import scene, color, arrow, compound, keysdown, rate, norm, sqrt, cos
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

    # Disable default controls
    scene.userpan = False
    scene.userzoom = True  # Keep zoom controls (scrollwheel)
    scene.userspin = False

    if title != '':
        scene.title = title

    if caption != '':
        scene.caption = caption

    convert_grid_to_z_up()
    # Any time a key or mouse is held down, run the callback function
    rate(30)  # 30Hz
    scene.bind('keydown mousedown', handle_keyboard_mouse_inputs)

    graphics_grid = GraphicsGrid()
    if not grid:
        graphics_grid.set_visibility(False)

    return graphics_grid


def convert_grid_to_z_up():
    """
    Rotate the camera so that +z is up
    (Default vpython scene is +y up)
    """

    '''
    There is an interaction between up and forward, the direction that the camera is pointing. By default, the camera
    points in the -z direction vector(0,0,-1). In this case, you can make the x or y axes (or anything between) be the
    up vector, but you cannot make the z axis be the up vector, because this is the axis about which the camera rotates
    when you set the up attribute. If you want the z axis to point up, first set forward to something other than the -z
    axis, for example vector(1,0,0). https://www.glowscript.org/docs/VPythonDocs/canvas.html
    '''
    # First set the x-axis forward
    scene.forward = x_axis_vector
    scene.up = z_axis_vector

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


def handle_keyboard_mouse_inputs():
    """
    Pans amount dependent on distance between camera and focus point.
    Closer = smaller pan amount

    A = move left (pan)
    D = move right (pan)
    W = move forward (pan)
    S = move backward (pan)

    <- = rotate left along camera axes (rotate)
    -> = rotate right along camera axes (rotate)
    ^ = rotate up along camera axes (rotate)
    V = rotate down along camera axes (rotate)

    Q = rotate clockwise around +Z (rotate)
    E = rotate counterclockwise around +Z (rotate)

    LMB = rotate
    RMB = pan

    """
    # Constants
    pan_amount = 0.05  # units
    rot_amount = 1.0  # deg

    # Current settings
    cam_distance = scene.camera.axis.mag
    cam_pos = vector(scene.camera.pos)
    cam_focus = vector(scene.center)

    # Weird manipulation to get correct vector directions. (scene.camera.up always defaults to world up)
    cam_axis = (vector(scene.camera.axis))  # X
    cam_side_axis = scene.camera.up.cross(cam_axis)  # Y
    cam_up = cam_axis.cross(cam_side_axis)  # Z

    # Get a list of keys
    keys = keysdown()

    ####################################################################################################################
    # PANNING

    # Check if the keys are pressed, update vectors as required
    if 'w' in keys:
        cam_pos = cam_pos + cam_axis * pan_amount
        # cam_focus = cam_focus + cam_axis * pan_amount
    if 's' in keys:
        cam_pos = cam_pos - cam_axis * pan_amount
        # cam_focus = cam_focus - cam_axis * pan_amount
    if 'a' in keys:
        cam_pos = cam_pos + cam_side_axis * pan_amount
        # cam_focus = cam_focus + cam_side_axis * pan_amount
    if 'd' in keys:
        cam_pos = cam_pos - cam_side_axis * pan_amount
        # cam_focus = cam_focus - cam_side_axis * pan_amount

    # Update camera
    scene.camera.pos = cam_pos
    # scene.camera.axis = cam_axis
    # scene.center = cam_focus

    ####################################################################################################################
    # Camera Roll

    # # Get an SE3 of the camera
    # cam_arr = array([
    #     [norm(cam_axis).x, norm(cam_side_axis).x, norm(cam_up).x, cam_pos.x],
    #     [norm(cam_axis).y, norm(cam_side_axis).y, norm(cam_up).y, cam_pos.y],
    #     [norm(cam_axis).z, norm(cam_side_axis).z, norm(cam_up).z, cam_pos.z],
    #     [               0,                     0,              0,         1]
    # ])
    # cam_se3 = SE3(cam_arr)
    #
    # # If only one rotation key is pressed
    # if 'q' in keys and 'e' not in keys:
    #     # Rotate SE3
    #     new_se3 = cam_se3 * SE3().Rz(-rot_amount, 'deg')
    #     # Obtain new parameters
    #     cam_axis = get_pose_x_vec(new_se3)
    #     cam_axis.mag = cam_distance
    #     cam_pos = cam_focus - cam_axis
    #     # Apply changes
    #     scene.camera.pos = cam_pos
    #     scene.camera.axis = cam_axis
    #
    # # If only one rotation key is pressed
    # if 'e' in keys and 'q' not in keys:
    #     # Rotate SE3
    #     new_se3 = cam_se3 * SE3().Rz(rot_amount, 'deg')
    #     # Obtain new parameters
    #     cam_axis = get_pose_x_vec(new_se3)
    #     cam_axis.mag = cam_distance
    #     cam_pos = cam_focus - cam_axis
    #     # Apply changes
    #     scene.camera.pos = cam_pos
    #     scene.camera.axis = cam_axis

    ####################################################################################################################
    # CAMERA ROTATION

    d = cam_distance
    move_dist = sqrt(d ** 2 + d ** 2 - 2 * d * d * cos(radians(rot_amount)))

    if 'left' in keys and 'right' not in keys:
        cam_pos = cam_pos + norm(cam_side_axis)*move_dist
        cam_axis = -(cam_pos - cam_focus)
    if 'right' in keys and 'left' not in keys:
        cam_pos = cam_pos - norm(cam_side_axis)*move_dist
        cam_axis = -(cam_pos - cam_focus)
    if 'up' in keys and 'down' not in keys:
        cam_pos = cam_pos + norm(cam_up)*move_dist
        cam_axis = -(cam_pos - cam_focus)
    if 'down' in keys and 'up' not in keys:
        cam_pos = cam_pos - norm(cam_up)*move_dist
        cam_axis = -(cam_pos - cam_focus)

    scene.camera.pos = cam_pos
    scene.camera.axis = cam_axis


