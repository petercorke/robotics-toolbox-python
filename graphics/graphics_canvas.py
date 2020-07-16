from vpython import scene, color, arrow, compound, keysdown, rate, norm, sqrt, cos, button, menu, checkbox, slider
from graphics.common_functions import *
from graphics.graphics_grid import GraphicsGrid


class GraphicsCanvas:
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
    """
    def __init__(self, height=500, width=1000, title='', caption='', grid=True):
        # Apply the settings
        scene.background = color.white
        scene.width = width
        scene.height = height
        scene.autoscale = False

        # Disable default controls
        scene.userpan = False  # Remove shift+mouse panning (not very good controls)
        scene.userzoom = True  # Keep zoom controls (scrollwheel)
        scene.userspin = True  # Keep ctrl+mouse enabled to rotate (keyboard rotation more tedious)

        # Apply HTML title/caption
        if title != '':
            scene.title = title

        self.__default_caption = caption
        if caption != '':
            scene.caption = caption

        # Create the UI
        self.__ui_controls = self.__setup_ui_controls([])
        # Indices to easily identify entities
        self.__idx_btn_reset = 0  # Camera Reset Button
        self.__idx_menu_robots = 1  # Menu box
        self.__idx_chkbox_ref = 2  # Reference Visibility Checkbox
        self.__idx_chkbox_rob = 3  # Robot Visibility Checkbox
        self.__idx_sld_opc = 4  # Opacity Slider

        # 2D array of all UI options data for each robot onscreen
        # checkbox ref, checkbox robot, and slider opacity
        self.__selected_robot = 0

        # List of robots currently in the scene
        self.__robots = []

        # Rotate the camera
        convert_grid_to_z_up()

        # Any time a key or mouse is held down, run the callback function
        rate(30)  # 30Hz
        scene.bind('keydown', handle_keyboard_inputs)

        # Create the grid, and display if wanted
        self.__graphics_grid = GraphicsGrid()
        if not grid:
            self.__graphics_grid.set_visibility(False)

    def clear_scene(self):
        self.__graphics_grid.clear_scene()

    def grid_visibility(self, is_visible):
        self.__graphics_grid.set_visibility(is_visible)

    def add_robot(self, robot):
        """
        This function is called when a new robot is created. It adds it to the menu.

        :param robot: A graphical robot to add to the scene
        :type robot: class:`GraphicalRobot`
        """
        # ALTHOUGH THE DOCUMENTATION SAYS THAT MENU CHOICES CAN BE UPDATED,
        # THE PACKAGE DOES NOT ALLOW IT.
        # THUS THIS 'HACK' MUST BE DONE TO REFRESH THE UI WITH AN UPDATED LIST

        # Save the list of robot names
        new_list = []
        for name in self.__ui_controls[self.__idx_menu_robots].choices:
            new_list.append(name)
        # Add the new one
        new_list.append(robot.name)

        # Remove all UI elements
        for item in self.__ui_controls:
            item.delete()
        # Restore the caption
        scene.caption = self.__default_caption
        # Create the updated caption.
        self.__ui_controls = self.__setup_ui_controls(new_list)

        # Add robot to list
        self.__robots.append(robot)

    def __setup_ui_controls(self, list_of_names):
        """
        The initial configuration of the user interface

        :param list_of_names: A list of names of the robots in the screen
        :type list_of_names: `list`
        """
        # Button to reset camera
        scene.append_to_caption('\n')
        btn_reset = button(bind=self.__reset_camera, text="Reset Camera")
        scene.append_to_caption('\t')

        # Drop down for robots / joints in frame
        menu_robots = menu(bind=self.__menu_item_chosen, choices=list_of_names)
        scene.append_to_caption('\n')

        # Checkbox for reference frame visibilities
        chkbox_ref = checkbox(bind=self.__reference_frame_checkbox, text="Show Reference Frames", checked=True)
        scene.append_to_caption('\t')

        # Checkbox for robot visibility
        chkbox_rob = checkbox(bind=self.__robot_visibility_checkbox, text="Show Robot", checked=True)
        scene.append_to_caption('\n')

        # Slider for robot opacity
        scene.append_to_caption('Opacity:')
        sld_opc = slider(bind=self.__opacity_slider, value=1)
        scene.append_to_caption('\n')

        # Control manual
        controls_str = '<br><b>Controls</b><br>' \
                       '<b>PAN</b><br>' \
                       'W , S | <i>forward / backward</i><br>' \
                       'A , D | <i>left / right</i><br>' \
                       'SPACE , SHIFT | <i>up / down</i><br>' \
                       '<b>ROTATE</b><br>' \
                       'CTRL + LMB | <i>free spin</i><br>' \
                       'ARROWS KEYS | <i>rotate direction</i><br>' \
                       'Q , E | <i>roll left / right</i><br>' \
                       '<b>ZOOM</b></br>' \
                       'MOUSEWHEEL | <i>zoom in / out</i><br>' \
                       '<script type="text/javascript">var arrow_keys_handler = function(e) {switch(e.keyCode){ case 37: case 39: case 38:  case 40: case 32: e.preventDefault(); break; default: break;}};window.addEventListener("keydown", arrow_keys_handler, false);</script>'
        # Disable the arrow keys from scrolling in the browser
        # https://stackoverflow.com/questions/8916620/disable-arrow-key-scrolling-in-users-browser
        scene.append_to_caption(controls_str)

        return [btn_reset, menu_robots, chkbox_ref, chkbox_rob, sld_opc]

    def __reset_camera(self):
        """
        Reset the camera to a default position and orientation
        """
        # Reset Camera
        scene.up = z_axis_vector
        scene.camera.pos = vector(10, 10, 10)
        scene.camera.axis = -scene.camera.pos

        # Update grid
        self.__graphics_grid.update_grid()

    def __menu_item_chosen(self, m):
        """
        When a menu item is chosen, update the relevant checkboxes/options
        """
        # Get selected item
        self.__selected_robot = m.index

        # Load settings for that robot and update UI
        self.__ui_controls[self.__idx_chkbox_ref].checked = \
            self.__robots[self.__selected_robot].ref_shown

        self.__ui_controls[self.__idx_chkbox_rob].checked = \
            self.__robots[self.__selected_robot].rob_shown

        self.__ui_controls[self.__idx_sld_opc].value = \
            self.__robots[self.__selected_robot].opacity

    def __reference_frame_checkbox(self, c):
        """
        When a checkbox is changed for the reference frame option, update the graphics
        """
        self.__robots[self.__selected_robot].set_reference_visibility(c.checked)

    def __robot_visibility_checkbox(self, c):
        """
        When a checkbox is changed for the robot visibility, update the graphics
        """
        self.__robots[self.__selected_robot].set_robot_visibility(c.checked)

    def __opacity_slider(self, s):
        """
        Update the opacity slider depending on the slider value
        """
        self.__robots[self.__selected_robot].set_transparency(s.value)


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


def handle_keyboard_inputs():
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

    Q = roll left (rotate)
    E = roll right (rotate)

    ctrl + LMB = rotate (Default Vpython)
    """
    # Constants
    pan_amount = 0.02  # units
    rot_amount = 1.0  # deg

    # Current settings
    cam_distance = scene.camera.axis.mag
    cam_pos = vector(scene.camera.pos)
    cam_focus = vector(scene.center)

    # Weird manipulation to get correct vector directions. (scene.camera.up always defaults to world up)
    cam_axis = (vector(scene.camera.axis))  # X
    cam_side_axis = scene.camera.up.cross(cam_axis)  # Y
    cam_up = cam_axis.cross(cam_side_axis)  # Z

    cam_up.mag = cam_axis.mag

    # Get a list of keys
    keys = keysdown()

    # Userspin uses ctrl, so skip this check to avoid changing camera pose while ctrl is held
    if 'ctrl' in keys:
        return

    ####################################################################################################################
    # PANNING
    # Check if the keys are pressed, update vectors as required
    # Changing camera position updates the scene center to follow same changes
    if 'w' in keys:
        cam_pos = cam_pos + cam_axis * pan_amount
    if 's' in keys:
        cam_pos = cam_pos - cam_axis * pan_amount
    if 'a' in keys:
        cam_pos = cam_pos + cam_side_axis * pan_amount
    if 'd' in keys:
        cam_pos = cam_pos - cam_side_axis * pan_amount
    if ' ' in keys:
        cam_pos = cam_pos + cam_up * pan_amount
    if 'shift' in keys:
        cam_pos = cam_pos - cam_up * pan_amount

    # Update camera position before rotation (to keep pan and rotate separate)
    scene.camera.pos = cam_pos

    ####################################################################################################################
    # Camera Roll
    # If only one rotation key is pressed
    if 'q' in keys and 'e' not in keys:
        # Rotate camera up
        cam_up = cam_up.rotate(angle=-radians(rot_amount), axis=cam_axis)
        # Set magnitude as it went to inf
        cam_up.mag = cam_axis.mag
        # Set
        scene.up = cam_up

    # If only one rotation key is pressed
    if 'e' in keys and 'q' not in keys:
        # Rotate camera up
        cam_up = cam_up.rotate(angle=radians(rot_amount), axis=cam_axis)
        # Set magnitude as it went to inf
        cam_up.mag = cam_axis.mag
        # Set
        scene.up = cam_up

    ####################################################################################################################
    # CAMERA ROTATION
    d = cam_distance
    move_dist = sqrt(d ** 2 + d ** 2 - 2 * d * d * cos(radians(rot_amount)))  # SAS Cosine

    # If only left not right key
    if 'left' in keys and 'right' not in keys:
        # Calculate distance to translate
        cam_pos = cam_pos + norm(cam_side_axis)*move_dist
        # Calculate new camera axis
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

    # Update camera position and axis
    scene.camera.pos = cam_pos
    scene.camera.axis = cam_axis
