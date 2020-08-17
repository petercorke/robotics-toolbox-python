from vpython import canvas, color, arrow, compound, keysdown, rate, norm, sqrt, cos, button, menu, checkbox, slider
from graphics.common_functions import *
from graphics.graphics_grid import GraphicsGrid, create_line, create_segmented_line, create_marker


class GraphicsCanvas3D:
    """
    Set up the scene with initial conditions.
        - White background
        - Width, height
        - Title, caption
        - Axes drawn (if applicable)

    :param height: Height of the canvas on screen (Pixels), defaults to 360.
    :type height: `int`, optional
    :param width: Width of the canvas on screen (Pixels), defaults to 640.
    :type width: `int`, optional
    :param title: Title of the plot. Gets displayed above canvas, defaults to ''.
    :type title: `str`, optional
    :param caption: Caption (subtitle) of the plot. Gets displayed below the canvas, defaults to ''.
    :type caption: `str`, optional
    :param grid: Whether a grid should be displayed in the plot, defaults to `True`.
    :type grid: `bool`, optional
    """

    def __init__(self, height=360, width=640, title='', caption='', grid=True):

        # Create a new independent scene
        self.scene = canvas()

        # Apply the settings
        self.scene.background = color.white
        self.scene.width = width
        self.scene.height = height
        self.scene.autoscale = False

        # Disable default controls
        self.scene.userpan = False  # Remove shift+mouse panning (not very good controls)
        self.scene.userzoom = True  # Keep zoom controls (scrollwheel)
        self.scene.userspin = True  # Keep ctrl+mouse enabled to rotate (keyboard rotation more tedious)

        # Apply HTML title/caption
        if title != '':
            self.scene.title = title

        self.__default_caption = caption
        if caption != '':
            self.scene.caption = caption

        # List of robots currently in the scene
        self.__robots = []
        self.__selected_robot = 0
        # Checkbox states
        self.__grid_visibility = grid
        self.__camera_lock = False
        self.__grid_relative = True

        # Create the UI
        self.__ui_controls = self.__setup_ui_controls([])
        # Indices to easily identify entities
        self.__idx_btn_reset = 0  # Camera Reset Button
        self.__idx_menu_robots = 1  # Menu box
        self.__idx_chkbox_ref = 2  # Reference Visibility Checkbox
        self.__idx_chkbox_rob = 3  # Robot Visibility Checkbox
        self.__idx_chkbox_grid = 4  # Grid Visibility Checkbox
        self.__idx_chkbox_cam = 5  # Camera Lock Checkbox
        self.__idx_chkbox_rel = 6  # Grid Relative Checkbox
        self.__idx_sld_opc = 7  # Opacity Slider
        self.__idx_btn_del = 8  # Delete button
        self.__idx_btn_clr = 9  # Clear button

        # List of joint sliders
        self.__teachpanel = []

        # Rotate the camera
        convert_grid_to_z_up(self.scene)

        # Any time a key or mouse is held down, run the callback function
        rate(30)  # 30Hz
        self.scene.bind('keydown', self.__handle_keyboard_inputs)

        # Create the grid, and display if wanted
        self.__graphics_grid = GraphicsGrid(self.scene)
        if not self.__grid_visibility:
            self.__graphics_grid.set_visibility(False)

    #######################################
    #  Canvas Management
    #######################################
    def clear_scene(self):
        """
        This function will clear the screen of all objects
        """
        # self.__graphics_grid.clear_scene()

        # Set all robots variables as invisible
        for robot in self.__robots:
            robot.set_reference_visibility(False)
            robot.set_robot_visibility(False)

        self.scene.waitfor("draw_complete")

        new_list = []
        for name in self.__ui_controls[self.__idx_menu_robots].choices:
            new_list.append(name)

        self.__selected_robot = 0
        self.__reload_caption(new_list)

    def grid_visibility(self, is_visible):
        """
        Update the grid visibility in the scene

        :param is_visible: Whether the grid should be visible or not
        :type is_visible: `bool`
        """
        self.__graphics_grid.set_visibility(is_visible)

    def add_robot(self, robot):
        """
        This function is called when a new robot is created. It adds it to the drop down menu.

        :param robot: A graphical robot to add to the scene
        :type robot: class:`graphics.graphics_robot.GraphicalRobot`
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

        self.__reload_caption(new_list)

        # Add robot to list
        self.__robots.append(robot)
        # Set it as selected
        self.__ui_controls[self.__idx_menu_robots].index = len(self.__robots) - 1
        self.__selected_robot = len(self.__robots) - 1

        for joint in robot.joints:
            self.__add_joint_slider(joint.qlim, joint.theta)

    #######################################
    #  UI Management
    #######################################
    def __del_robot(self):
        """
        Remove a robot from the scene and the UI controls
        """
        if len(self.__robots) == 0:
            # Alert the user and return
            self.scene.append_to_caption('<script type="text/javascript">alert("No robot to delete");</script>')
            return

        # Clear the robot visuals
        self.__robots[self.__selected_robot].set_reference_visibility(False)
        self.__robots[self.__selected_robot].set_robot_visibility(False)

        # Remove from UI
        new_list = []
        for name in self.__ui_controls[self.__idx_menu_robots].choices:
            new_list.append(name)
        # Add the new one
        del new_list[self.__selected_robot]
        del self.__robots[self.__selected_robot]

        # Select the top item
        if len(self.__ui_controls[self.__idx_menu_robots].choices) > 0:
            self.__ui_controls[self.__idx_menu_robots].index = 0
            self.__selected_robot = len(self.__robots) - 1

        # Update UI
        self.__reload_caption(new_list)

    def __handle_keyboard_inputs(self):
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

        space = move up (pan)
        shift = move down (pan)

        ctrl + LMB = rotate (Default Vpython)
        """
        # If camera lock, just skip the function
        if self.__camera_lock:
            return

        # Constants
        pan_amount = 0.02  # units
        rot_amount = 1.0  # deg

        # Current settings
        cam_distance = self.scene.camera.axis.mag
        cam_pos = vector(self.scene.camera.pos)
        cam_focus = vector(self.scene.center)

        # Weird manipulation to get correct vector directions. (scene.camera.up always defaults to world up)
        cam_axis = (vector(self.scene.camera.axis))  # X
        cam_side_axis = self.scene.camera.up.cross(cam_axis)  # Y
        cam_up = cam_axis.cross(cam_side_axis)  # Z

        cam_up.mag = cam_axis.mag

        # Get a list of keys
        keys = keysdown()

        # Userspin uses ctrl, so skip this check to avoid changing camera pose while ctrl is held
        if 'ctrl' in keys:
            return

        ################################################################################################################
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
        self.scene.camera.pos = cam_pos

        ################################################################################################################
        # Camera Roll
        # If only one rotation key is pressed
        if 'q' in keys and 'e' not in keys:
            # Rotate camera up
            cam_up = cam_up.rotate(angle=-radians(rot_amount), axis=cam_axis)
            # Set magnitude as it went to inf
            cam_up.mag = cam_axis.mag
            # Set
            self.scene.up = cam_up

        # If only one rotation key is pressed
        if 'e' in keys and 'q' not in keys:
            # Rotate camera up
            cam_up = cam_up.rotate(angle=radians(rot_amount), axis=cam_axis)
            # Set magnitude as it went to inf
            cam_up.mag = cam_axis.mag
            # Set
            self.scene.up = cam_up

        ################################################################################################################
        # CAMERA ROTATION
        d = cam_distance
        move_dist = sqrt(d ** 2 + d ** 2 - 2 * d * d * cos(radians(rot_amount)))  # SAS Cosine

        # If only left not right key
        if 'left' in keys and 'right' not in keys:
            # Calculate distance to translate
            cam_pos = cam_pos + norm(cam_side_axis) * move_dist
            # Calculate new camera axis
            cam_axis = -(cam_pos - cam_focus)
        if 'right' in keys and 'left' not in keys:
            cam_pos = cam_pos - norm(cam_side_axis) * move_dist
            cam_axis = -(cam_pos - cam_focus)
        if 'up' in keys and 'down' not in keys:
            cam_pos = cam_pos + norm(cam_up) * move_dist
            cam_axis = -(cam_pos - cam_focus)
        if 'down' in keys and 'up' not in keys:
            cam_pos = cam_pos - norm(cam_up) * move_dist
            cam_axis = -(cam_pos - cam_focus)

        # Update camera position and axis
        self.scene.camera.pos = cam_pos
        self.scene.camera.axis = cam_axis

    def __reload_caption(self, new_list):
        """
        Reload the UI with the new list of robot names
        """
        # Remove all UI elements
        for item in self.__ui_controls:
            item.delete()
        # Restore the caption
        self.scene.caption = self.__default_caption
        # Create the updated caption.
        self.__ui_controls = self.__setup_ui_controls(new_list)

    def __setup_ui_controls(self, list_of_names):
        """
        The initial configuration of the user interface

        :param list_of_names: A list of names of the robots in the screen
        :type list_of_names: `list`
        """
        # Button to reset camera
        self.scene.append_to_caption('\n')
        btn_reset = button(bind=self.__reset_camera, text="Reset Camera")
        self.scene.append_to_caption('\t')

        chkbox_cam = checkbox(bind=self.__camera_lock_checkbox, text="Camera Lock", checked=self.__camera_lock)
        # Prevent the space bar from toggling the active checkbox/button/etc (default browser behaviour)
        self.scene.append_to_caption('''
            <script type="text/javascript">
                $(document).keyup(function(event) {
                    if(event.which === 32) {
                        event.preventDefault();
                    }
                });
            </script>''')
        # https://stackoverflow.com/questions/22280139/prevent-space-button-from-triggering-any-other-button-click-in-jquery
        self.scene.append_to_caption('\t')

        chkbox_rel = checkbox(bind=self.__grid_relative_checkbox, text="Grid Relative", checked=self.__grid_relative)
        self.scene.append_to_caption('\n')

        # Drop down for robots / joints in frame
        menu_robots = menu(bind=self.__menu_item_chosen, choices=list_of_names)
        self.scene.append_to_caption('\t')

        # Button to delete the selected robot
        btn_del = button(bind=self.__del_robot, text="Delete Robot")
        self.scene.append_to_caption('\t')

        # Button to clear the robots in screen
        btn_clr = button(bind=self.clear_scene, text="Clear Scene")
        self.scene.append_to_caption('\n')

        # Checkbox for grid visibility
        chkbox_grid = checkbox(bind=self.__grid_visibility_checkbox, text="Grid Visibility",
                               checked=self.__grid_visibility)
        self.scene.append_to_caption('\t')

        # Checkbox for reference frame visibilities
        if len(self.__robots) == 0:
            chkbox_ref = checkbox(bind=self.__reference_frame_checkbox, text="Show Reference Frames", checked=True)
        else:
            chk = self.__robots[self.__selected_robot].ref_shown
            chkbox_ref = checkbox(bind=self.__reference_frame_checkbox, text="Show Reference Frames", checked=chk)
        self.scene.append_to_caption('\t')

        # Checkbox for robot visibility
        if len(self.__robots) == 0:
            chkbox_rob = checkbox(bind=self.__robot_visibility_checkbox, text="Show Robot", checked=True)
        else:
            chk = self.__robots[self.__selected_robot].rob_shown
            chkbox_rob = checkbox(bind=self.__robot_visibility_checkbox, text="Show Robot", checked=chk)
        self.scene.append_to_caption('\n')

        # Slider for robot opacity
        self.scene.append_to_caption('Opacity:')
        if len(self.__robots) == 0:
            sld_opc = slider(bind=self.__opacity_slider, value=1)
        else:
            opc = self.__robots[self.__selected_robot].opacity
            sld_opc = slider(bind=self.__opacity_slider, value=opc)
        self.scene.append_to_caption('\n')

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
        self.scene.append_to_caption(controls_str)

        return [btn_reset, menu_robots, chkbox_ref, chkbox_rob, chkbox_grid, chkbox_cam, chkbox_rel, sld_opc, btn_del,
                btn_clr]

    def __add_joint_slider(self, qlim, theta):
        """

        """
        s = slider(bind=self.__joint_slider, min=qlim[0], max=qlim[1], value=theta)
        self.scene.append_to_caption('\n')
        self.__teachpanel.append(s)

    #######################################
    # UI CALLBACKS
    #######################################
    def __reset_camera(self):
        """
        Reset the camera to a default position and orientation
        """
        # Reset Camera
        self.scene.up = z_axis_vector
        self.scene.camera.pos = vector(10, 10, 10)
        self.scene.camera.axis = -self.scene.camera.pos

        # Update grid
        self.__graphics_grid.update_grid()

    def __menu_item_chosen(self, m):
        """
        When a menu item is chosen, update the relevant checkboxes/options

        :param m: The menu object that has been used to select an item.
        :type: class:`menu`
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

        :param c: The checkbox that has been toggled
        :type c: class:`checkbox`
        """
        if len(self.__robots) > 0:
            self.__robots[self.__selected_robot].set_reference_visibility(c.checked)

    def __robot_visibility_checkbox(self, c):
        """
        When a checkbox is changed for the robot visibility, update the graphics

        :param c: The checkbox that has been toggled
        :type c: class:`checkbox`
        """
        if len(self.__robots) > 0:
            self.__robots[self.__selected_robot].set_robot_visibility(c.checked)

    def __grid_visibility_checkbox(self, c):
        """
        When a checkbox is changed for the grid visibility, update the graphics

        :param c: The checkbox that has been toggled
        :type c: class:`checkbox`
        """
        self.grid_visibility(c.checked)
        self.__grid_visibility = c.checked

    def __camera_lock_checkbox(self, c):
        """
        When a checkbox is changed for the camera lock, update the camera

        :param c: The checkbox that has been toggled
        :type c: class:`checkbox`
        """
        # Update parameters
        # True = locked
        self.__camera_lock = c.checked
        # True = enabled
        self.scene.userspin = not c.checked
        self.scene.userzoom = not c.checked

    def __grid_relative_checkbox(self, c):
        """
        When a checkbox is changed for the grid lock, update the grid

        :param c: The checkbox that has been toggled
        :type c: class:`checkbox`
        """
        self.__graphics_grid.set_relative(c.checked)
        self.__grid_relative = c.checked

    def __opacity_slider(self, s):
        """
        Update the opacity slider depending on the slider value

        :param s: The slider object that has been modified
        :type s: class:`slider`
        """
        if len(self.__robots) > 0:
            self.__robots[self.__selected_robot].set_transparency(s.value)

    def __joint_slider(self, s):
        """

        """
        # Get all angles
        angles = []
        for joint_slider in self.__teachpanel:
            angles.append(joint_slider.value)

        # Run fkine
        poses = self.__robots[self.__selected_robot].fkine(angles)
        poses = poses[1:len(poses)]  # Ignore the first item

        # Update joints
        self.__robots[self.__selected_robot].set_joint_poses(poses)



class GraphicsCanvas2D:
    """
        Set up the scene with initial conditions.
            - White background
            - Width, height
            - Title, caption
            - Axes drawn (if applicable)

        :param height: Height of the canvas on screen (Pixels), defaults to 360.
        :type height: `int`, optional
        :param width: Width of the canvas on screen (Pixels), defaults to 640.
        :type width: `int`, optional
        :param title: Title of the plot. Gets displayed above canvas, defaults to ''.
        :type title: `str`, optional
        :param caption: Caption (subtitle) of the plot. Gets displayed below the canvas, defaults to ''.
        :type caption: `str`, optional
        :param grid: Whether a grid should be displayed in the plot, defaults to `True`.
        :type grid: `bool`, optional
        """

    def __init__(self, height=360, width=640, title='', caption='', grid=True):

        # Private lists
        self.__line_styles = [
            '',  # None
            '-',  # Solid (default)
            '--',  # Dashes
            ':',  # Dotted
            '-.',  # Dash-dot
        ]
        self.__marker_styles = [
            '+',  # Plus
            'o',  # Circle
            '*',  # Star
            '.',  # Dot
            'x',  # Cross
            's',  # Square
            'd',  # Diamond
            '^',  # Up triangle
            'v',  # Down triangle
            '<',  # Left triangle
            '>',  # Right triangle
            'p',  # Pentagon
            'h',  # Hexagon
        ]
        self.__colour_styles = [
            'r',  # Red
            'g',  # Green
            'b',  # Blue
            'y',  # Yellow
            'c',  # Cyan
            'm',  # Magenta
            'k',  # Black (default)
            'w',  # White
        ]
        self.__colour_dictionary = {
            'r': color.red.value,
            'g': color.green.value,
            'b': color.blue.value,
            'c': color.cyan.value,
            'y': color.yellow.value,
            'm': color.magenta.value,
            'k': color.black.value,
            'w': color.white.value
        }

        # Create a new independent scene
        self.scene = canvas()

        # Apply the settings
        self.scene.background = color.white
        self.scene.width = width
        self.scene.height = height
        self.scene.autoscale = False

        # Disable default controls
        self.scene.userpan = True  # Keep shift+mouse panning (key overwritten)
        self.scene.userzoom = True  # Keep zoom controls (scrollwheel)
        self.scene.userspin = False  # Remove ctrl+mouse enabled to rotate

        self.__grid_visibility = grid
        self.__camera_lock = False

        # Apply HTML title/caption
        if title != '':
            self.scene.title = title

        self.__default_caption = caption
        if caption != '':
            self.scene.caption = caption

        # Rotate the camera
        # convert_grid_to_z_up(self.scene)

        # Any time a key or mouse is held down, run the callback function
        rate(30)  # 30Hz
        self.scene.bind('keydown', self.__handle_keyboard_inputs)

        # Create the grid, and display if wanted
        self.__graphics_grid = GraphicsGrid(self.scene)
        # Toggle grid to 2D
        self.__graphics_grid.toggle_2d_3d()
        # Lock the grid
        # self.__graphics_grid.set_relative(False)
        # Turn off grid if applicable
        if not self.__grid_visibility:
            self.__graphics_grid.set_visibility(False)

        # Reset the camera to known spot
        self.__reset_camera()
        self.__graphics_grid.update_grid()

    #######################################
    #  Canvas Management
    #######################################
    # TODO
    def clear_scene(self):
        pass

    def grid_visibility(self, is_visible):
        """
        Update the grid visibility in the scene

        :param is_visible: Whether the grid should be visible or not
        :type is_visible: `bool`
        """
        self.__graphics_grid.set_visibility(is_visible)

    #######################################
    #  UI Management
    #######################################
    # TODO
    def __setup_ui_controls(self):
        pass

    def __handle_keyboard_inputs(self):
        """
        Pans amount dependent on distance between camera and focus point.
        Closer = smaller pan amount

        A = move left (pan)
        D = move right (pan)
        W = move up (pan)
        S = move down (pan)

        <- = rotate left along camera axes (rotate)
        -> = rotate right along camera axes (rotate)
        ^ = rotate up along camera axes (rotate)
        V = rotate down along camera axes (rotate)

        Q = roll left (rotate)
        E = roll right (rotate)

        ctrl + LMB = rotate (Default Vpython)
        """
        # If camera lock, just skip the function
        if self.__camera_lock:
            return

        # Constants
        pan_amount = 0.02  # units
        rot_amount = 1.0  # deg

        # Current settings
        cam_distance = self.scene.camera.axis.mag
        cam_pos = vector(self.scene.camera.pos)
        cam_focus = vector(self.scene.center)

        # Weird manipulation to get correct vector directions. (scene.camera.up always defaults to world up)
        cam_axis = (vector(self.scene.camera.axis))  # X
        cam_side_axis = self.scene.camera.up.cross(cam_axis)  # Y
        cam_up = cam_axis.cross(cam_side_axis)  # Z

        cam_up.mag = cam_axis.mag

        # Get a list of keys
        keys = keysdown()

        # Userpan uses ctrl, so skip this check to avoid changing camera pose while shift is held
        if 'shift' in keys:
            return

        ################################################################################################################
        # PANNING
        # Check if the keys are pressed, update vectors as required
        # Changing camera position updates the scene center to follow same changes
        if 'w' in keys:
            cam_pos = cam_pos + cam_up * pan_amount
        if 's' in keys:
            cam_pos = cam_pos - cam_up * pan_amount
        if 'a' in keys:
            cam_pos = cam_pos + cam_side_axis * pan_amount
        if 'd' in keys:
            cam_pos = cam_pos - cam_side_axis * pan_amount

        # Update camera position before rotation (to keep pan and rotate separate)
        self.scene.camera.pos = cam_pos

        ################################################################################################################
        # Camera Roll
        # If only one rotation key is pressed
        if 'q' in keys and 'e' not in keys:
            # Rotate camera up
            cam_up = cam_up.rotate(angle=-radians(rot_amount), axis=cam_axis)
            # Set magnitude as it went to inf
            cam_up.mag = cam_axis.mag
            # Set
            self.scene.up = cam_up

        # If only one rotation key is pressed
        if 'e' in keys and 'q' not in keys:
            # Rotate camera up
            cam_up = cam_up.rotate(angle=radians(rot_amount), axis=cam_axis)
            # Set magnitude as it went to inf
            cam_up.mag = cam_axis.mag
            # Set
            self.scene.up = cam_up

        ################################################################################################################
        # CAMERA ROTATION
        d = cam_distance
        move_dist = sqrt(d ** 2 + d ** 2 - 2 * d * d * cos(radians(rot_amount)))  # SAS Cosine

        # If only left not right key
        if 'left' in keys and 'right' not in keys:
            # Calculate distance to translate
            cam_pos = cam_pos + norm(cam_side_axis) * move_dist
            # Calculate new camera axis
            cam_axis = -(cam_pos - cam_focus)
        if 'right' in keys and 'left' not in keys:
            cam_pos = cam_pos - norm(cam_side_axis) * move_dist
            cam_axis = -(cam_pos - cam_focus)
        if 'up' in keys and 'down' not in keys:
            cam_pos = cam_pos + norm(cam_up) * move_dist
            cam_axis = -(cam_pos - cam_focus)
        if 'down' in keys and 'up' not in keys:
            cam_pos = cam_pos - norm(cam_up) * move_dist
            cam_axis = -(cam_pos - cam_focus)

        # Update camera position and axis
        self.scene.camera.pos = cam_pos
        self.scene.camera.axis = cam_axis

    def __reset_camera(self):
        """
        Reset the camera to a known position
        """
        # Reset Camera
        self.scene.camera.pos = vector(5, 5, 12)  # Hover above (5, 5, 0)
        # Ever so slightly off focus, to ensure grid is rendered in the right region
        # (if directly at, draws numbers wrong spots)
        self.scene.camera.axis = vector(-0.001, -0.001, -12)  # Focus on (5, 5, 0)
        self.scene.up = y_axis_vector

    #######################################
    #  Drawing Functions
    #######################################
    def __draw_path(self, x_path, y_path, opt_line, opt_marker, opt_colour, thickness=0.05):
        """
        Draw a line from point to point in the 2D path

        :param x_path: The x path to draw on the canvas
        :type x_path: `list`
        :param y_path: The y path to draw on the canvas
        :type y_path: `list`
        :param opt_line: The line option argument
        :type opt_line: `str`
        :param opt_marker: The marker option argument
        :type opt_marker: `str`
        :param opt_colour: The colour option argument
        :type opt_colour: `str`
        :param thickness: Thickness of the line
        :type thickness: `float`
        :raises ValueError: Invalid line type given
        """
        # Get colour
        colour = self.__get_colour_from_string(opt_colour)

        # For every point in the list, draw a line to the next one (excluding last point)
        for point in range(0, len(x_path)):
            # Get point 1
            x1 = x_path[point]
            y1 = y_path[point]
            p1 = vector(x1, y1, 0)

            # If at end / only coordinate - draw a marker
            if point == len(x_path) - 1:
                create_marker(self.scene, x1, y1, opt_marker, colour)
                return

            # Get point 2
            x2 = x_path[point + 1]
            y2 = y_path[point + 1]
            p2 = vector(x2, y2, 0)

            if opt_line == '':
                # Only one marker to avoid double-ups
                create_marker(self.scene, x1, y1, opt_marker, colour)
            elif opt_line == '-':
                create_line(p1, p2, self.scene, colour=colour, thickness=thickness)
                # Only one marker to avoid double-ups
                create_marker(self.scene, x1, y1, opt_marker, colour)
            elif opt_line == '--':
                create_segmented_line(p1, p2, self.scene, 0.3, colour=colour, thickness=thickness)
                # Only one marker to avoid double-ups
                create_marker(self.scene, x1, y1, opt_marker, colour)
            elif opt_line == ':':
                create_segmented_line(p1, p2, self.scene, 0.05, colour=colour, thickness=thickness)
                # Only one marker to avoid double-ups
                create_marker(self.scene, x1, y1, opt_marker, colour)
            elif opt_line == '-.':
                raise NotImplementedError("Other line types not implemented")
            else:
                raise ValueError("Invalid line type given")

    def plot(self, x_coords, y_coords=None, options=''):
        """
        Same usage as MATLAB's plot.

        If given one list of coordinates, plots against index
        If given two lists of coordinates, plots both (1st = x, 2nd = y)

        Options string is identical to MATLAB's input string
        If you do not specify a marker type, plot uses no marker.
        If you do not specify a line style, plot uses a solid line.

            b     blue          .     point              -     solid
            g     green         o     circle             :     dotted
            r     red           x     x-mark             -.    dashdot
            c     cyan          +     plus               --    dashed
            m     magenta       *     star             (none)  no line
            y     yellow        s     square
            k     black         d     diamond
            w     white         v     triangle (down)
                                ^     triangle (up)
                                <     triangle (left)
                                >     triangle (right)
                                p     pentagram
                                h     hexagram

        :param x_coords: The first plane of coordinates to plot
        :type x_coords: `list`
        :param y_coords: The second plane of coordinates to plot with.
        :type y_coords: `list`, `str`, optional
        :param options: A string of options to plot with
        :type options: `str`, optional
        :raises ValueError: Number of X and Y coordinates must be equal
        """
        # TODO
        #  add options for line width, marker size

        # If y-vector is str, then only x vector given
        if isinstance(y_coords, str):
            options = y_coords
            y_coords = None

        one_set_data = False
        # Set y-vector to default if None
        if y_coords is None:
            one_set_data = True
            y_coords = [*range(0, len(x_coords))]

        # Verify x, y coords have same length
        if len(x_coords) != len(y_coords):
            raise ValueError("Number of X coordinates does not equal number of Y coordinates.")

        # Verify options given (and save settings to be applied)
        verified_options = self.__verify_plot_options(options)

        if one_set_data:
            # Draw plot for one list of data
            self.__draw_path(
                y_coords,  # Y is default x-coords in one data set
                x_coords,  # User input
                verified_options[0],  # Line
                verified_options[1],  # Marker
                verified_options[2],  # Colour
            )
        else:
            # Draw plot for two lists of data
            self.__draw_path(
                x_coords,  # User input
                y_coords,  # User input
                verified_options[0],  # Line
                verified_options[1],  # Marker
                verified_options[2],  # Colour
            )

    def __verify_plot_options(self, options_str):
        """
        Verify that the given options are usable.

        :param options_str: The given options from the plot command to verify user input
        :type options_str: `str`
        :raises ValueError: Unknown character entered
        :raises ValueError: Too many line segments used
        :raises ValueError: Too many marker segments used
        :raises ValueError: Too many colour segments used
        :returns: List of options to plot with
        :rtype: `list`
        """
        default_line = '-'
        default_marker = ''
        default_colour = 'k'

        # Split str into chars list
        options_split = list(options_str)

        # If 0, set defaults and return early
        if len(options_split) == 0:
            return [default_line, default_marker, default_colour]

        # If line_style given, join the first two options if applicable (some types have 2 characters)
        for char in range(0, len(options_split)-1):
            # If char is '-' (only leading character in double length option)
            if options_split[char] == '-' and len(options_split) > 1:
                # If one of the leading characters is valid
                if options_split[char+1] == '-' or options_split[char+1] == '.':
                    # Join the two into the first
                    options_split[char] = options_split[char] + options_split[char+1]
                    # Shuffle down the rest
                    for idx in range(char+2, len(options_split)):
                        options_split[idx - 1] = options_split[idx]
                    # Remove duplicate extra
                    options_split.pop()

        # If any unknown, throw error
        for option in options_split:
            if option not in self.__line_styles and \
                    option not in self.__marker_styles and \
                    option not in self.__colour_styles:
                error_string = "Unknown character entered: '{0}'"
                raise ValueError(error_string.format(option))

        ##############################
        # Verify Line Style
        ##############################
        line_style_count = 0  # Count of options used
        line_style_index = 0  # Index position of index used (only used when count == 1)
        for option in options_split:
            if option in self.__line_styles:
                line_style_count = line_style_count + 1
                line_style_index = self.__line_styles.index(option)

        # If more than one, throw error
        if line_style_count > 1:
            raise ValueError("Too many line style arguments given. Only one allowed")
        # If none, set as solid
        elif line_style_count == 0 or not any(item in options_split for item in self.__line_styles):
            output_line = default_line
        # If one, set as given
        else:
            output_line = self.__line_styles[line_style_index]
        ##############################

        ##############################
        # Verify Marker Style
        ##############################
        marker_style_count = 0  # Count of options used
        marker_style_index = 0  # Index position of index used (only used when count == 1)
        for option in options_split:
            if option in self.__marker_styles:
                marker_style_count = marker_style_count + 1
                marker_style_index = self.__marker_styles.index(option)

        # If more than one, throw error
        if marker_style_count > 1:
            raise ValueError("Too many marker style arguments given. Only one allowed")
        # If none, set as no-marker
        elif marker_style_count == 0 or not any(item in options_split for item in self.__marker_styles):
            output_marker = default_marker
        # If one, set as given
        else:
            output_marker = self.__marker_styles[marker_style_index]
            # If marker set and no line given, turn line to no-line
            if line_style_count == 0 or not any(item in options_split for item in self.__line_styles):
                output_line = ''
        ##############################

        ##############################
        # Verify Colour Style
        ##############################
        colour_style_count = 0  # Count of options used
        colour_style_index = 0  # Index position of index used (only used when count == 1)
        for option in options_split:
            if option in self.__colour_styles:
                colour_style_count = colour_style_count + 1
                colour_style_index = self.__colour_styles.index(option)

        # If more than one, throw error
        if colour_style_count > 1:
            raise ValueError("Too many colour style arguments given. Only one allowed")
        # If none, set as black
        elif colour_style_count == 0 or not any(item in options_split for item in self.__colour_styles):
            output_colour = default_colour
        # If one, set as given
        else:
            output_colour = self.__colour_styles[colour_style_index]
        ##############################

        return [output_line, output_marker, output_colour]

    def __get_colour_from_string(self, colour_string):
        """
        Using the colour plot string input, return an rgb array of the colour selected
        :param colour_string: The colour string option
        :type colour_string: `str`
        :returns: List of RGB values for the representative colour
        :rtype: `list`
        """
        # Return the RGB list (black if not in dictionary)
        return self.__colour_dictionary.get(colour_string, color.black.value)

    # MAY NOT BE REQUIRED
    # def add_object(self, obj):
    #     """
    #     Add an object into the scene.
    #
    #     :param obj: The object to be added
    #     :type obj: class:`graphics.object2d`
    #     """
    #     pass


def convert_grid_to_z_up(scene):
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


def draw_reference_frame_axes(se3_pose, scene):
    """
    Draw x, y, z axes from the given point.
    Each axis is represented in the objects reference frame.


    :param se3_pose: SE3 pose representation of the reference frame
    :type se3_pose: class:`spatialmath.pose3d.SE3`
    :param scene: Which scene to put the graphics in
    :type scene: class:`vpython.canvas`
    :return: Compound object of the 3 axis arrows.
    :rtype: class:`vpython.compound`
    """

    origin = get_pose_pos(se3_pose)
    x_axis = get_pose_x_vec(se3_pose)
    y_axis = get_pose_y_vec(se3_pose)

    # Create Basic Frame
    # Draw X Axis
    x_arrow = arrow(canvas=scene, pos=origin, axis=x_axis_vector, length=0.25, color=color.red)

    # Draw Y Axis
    y_arrow = arrow(canvas=scene, pos=origin, axis=y_axis_vector, length=0.25, color=color.green)

    # Draw Z Axis
    z_arrow = arrow(canvas=scene, pos=origin, axis=z_axis_vector, length=0.25, color=color.blue)

    # Combine all to manipulate together
    # Set origin to where axis converge (instead of the middle of the resulting object bounding box)
    frame_ref = compound([x_arrow, y_arrow, z_arrow], origin=origin, canvas=scene)

    # Set frame axes
    frame_ref.axis = x_axis
    frame_ref.up = y_axis

    return frame_ref
