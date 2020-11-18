#!/usr/bin/env python
"""
@author Micah Huth
"""

from vpython import canvas, color, arrow, compound, keysdown, rate, norm, \
    sqrt, cos, button, menu, checkbox, slider, wtext, degrees, vector, \
    radians
from roboticstoolbox.backends.VPython.common_functions import \
    get_pose_x_vec, get_pose_y_vec, get_pose_pos, \
    x_axis_vector, y_axis_vector, z_axis_vector
from roboticstoolbox.backends.VPython.grid import GraphicsGrid, create_line, \
    create_segmented_line, create_marker
from enum import Enum
from collections.abc import MutableMapping


class UImode(Enum):  # pragma nocover
    CANVASCONTROL = 1
    TEACHPANEL = 2


class UIMMap(MutableMapping):    # pragma nocover
    def __init__(self, *args, **kwargs):
        self.__dict__.update(*args, **kwargs)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)


class GraphicsCanvas3D:  # pragma nocover
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
    :param title: Title of the plot. Gets displayed above canvas, defaults
        to ''.
    :type title: `str`, optional
    :param caption: Caption (subtitle) of the plot. Gets displayed below the
        canvas, defaults to ''.
    :type caption: `str`, optional
    :param grid: Whether a grid should be displayed in the plot, defaults
        to `True`.
    :type grid: `bool`, optional
    :param g_col: The RGB grid colour
    :type g_col: `list`, optional
    """

    def __init__(self, height=500, width=888, title='', caption='', grid=True, g_col=None):

        # Create a new independent scene
        self.scene = canvas()

        # Apply the settings
        self.scene.background = color.white
        self.scene.width = width
        self.scene.height = height
        self.scene.autoscale = False

        # Disable default controls
        # Remove shift+mouse panning (not very good controls)
        self.scene.userpan = False
        # Keep zoom controls (scrollwheel)
        self.scene.userzoom = True
        # Keep ctrl+mouse enabled to rotate (keyboard rotation more tedious)
        self.scene.userspin = True

        # Apply HTML title/caption
        if title != '':
            self.scene.title = title

        self.scene.append_to_title(
            '<style>#glowscript h3{margin-bottom: -10px;border-bottom: 2px solid #15578a;padding-bottom: 5px;}#glowscript button{background: #8db9db !important;color: #fff !important;border: 2px solid #5F9ED0;border-radius: 8px;}#glowscript > div:nth-of-type(4n){display:inline-block;margin:20px;vertical-align:top;}#glowscript > div:nth-of-type(4n-3){background:#d9d9d9;padding:20px;border-radius:10px;margin-bottom:10px;}</style>' \
            '<script type="text/javascript">var arrow_keys_handler = function(e) {switch(e.keyCode){ case 37: case 39: case 38:  case 40: case 32: e.preventDefault(); break; default: break;}};window.addEventListener("keydown", arrow_keys_handler, false);</script>' \
            '<script type="text/javascript">$(document).keyup(function(event){if (event.which === 32){event.preventDefault();}});</script>'
            # noqa
        )
        # Disable the arrow keys from scrolling in the browser
        # https://stackoverflow.com/questions/8916620/disable-arrow-key-scrolling-in-users-browser

        # Prevent the space bar from toggling the active checkbox/button/etc
        # (default browser behaviour)
        # https://stackoverflow.com/questions/22280139/prevent-space-button-from-triggering-any-other-button-click-in-jquery

        self.__default_caption = caption
        if caption != '':
            self.scene.caption = caption

        # List of robots currently in the scene
        self.__robots = []
        self.__selected_robot = 0
        # List of joint sliders per robot
        self.__teachpanel = []  # 3D, robot -> joint -> options
        self.__teachpanel_sliders = []
        self.__idx_qlim_min, self.__idx_qlim_max, self.__idx_theta, \
            self.__idx_text = 0, 1, 2, 3
        # Checkbox states
        self.__grid_visibility = grid
        self.__camera_lock = False
        self.__grid_relative = True
        # Screen Shot tally
        self._ss_tally = 0

        # Create the UI
        self.__ui_mode = UImode.CANVASCONTROL
        self.__toggle_button_text_dict = {
            UImode.CANVASCONTROL: "Canvas Controls",
            UImode.TEACHPANEL: "Robot Controls"
        }
        self.__ui_controls = UIMMap()
        self.__add_mode_button()
        self.__setup_ui_controls([])

        # Rotate the camera
        convert_grid_to_z_up(self.scene)
        self.scene.waitfor("draw_complete")  # Ensure camera updates before grid is created

        # Any time a key or mouse is held down, run the callback function
        rate(30)  # 30Hz
        self.scene.bind('keydown', self.__handle_keyboard_inputs)

        # Create the grid, and display if wanted
        self.__graphics_grid = GraphicsGrid(self.scene, colour=g_col)
        if not self.__grid_visibility:
            self.__graphics_grid.set_visibility(False)

    #######################################
    #  Canvas Management
    #######################################
    def clear_scene(self):
        """
        This function will clear the screen of all objects
        """
        # Set all robots variables as invisible
        for robot in self.__robots:
            robot.set_reference_visibility(False)
            robot.set_robot_visibility(False)

        self.scene.waitfor("draw_complete")

        new_list = []
        for name in self.__ui_controls.get('menu_robots').choices:
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
        This function is called when a new robot is created. It adds it to
        the drop down menu.

        :param robot: A graphical robot to add to the scene
        :type robot: class:`graphics.graphics_robot.GraphicalRobot`
        """
        # ALTHOUGH THE DOCUMENTATION SAYS THAT MENU CHOICES CAN BE UPDATED,
        # THE PACKAGE DOES NOT ALLOW IT.
        # THUS THIS 'HACK' MUST BE DONE TO REFRESH THE UI WITH AN UPDATED LIST

        # Save the list of robot names
        new_list = []
        for name in self.__ui_controls.get('menu_robots').choices:
            new_list.append(name)
        # Add the new one
        new_list.append(robot.name)

        # Add robot to list
        self.__robots.append(robot)
        self.__selected_robot = len(self.__robots) - 1

        num_options = 4
        #  Add spot for current robot settings
        self.__teachpanel.append([[0] * num_options] * robot.num_joints)

        # Add robot joint sliders
        i = 0
        for joint in robot.joints:
            if joint.qlim[0] == joint.qlim[1]:
                self.__teachpanel[self.__selected_robot][i] = [
                    joint.qlim[0], joint.qlim[1],
                    joint.theta, None]
            else:
                string = "{:.2f} rad ({:.2f} deg)".format(
                    joint.theta, degrees(joint.theta))
                self.__teachpanel[self.__selected_robot][i] = [
                    joint.qlim[0], joint.qlim[1],
                    joint.theta, wtext(text=string)]
            i += 1

        # Refresh the caption
        self.__reload_caption(new_list)

        # Set it as selected
        self.__ui_controls.get('menu_robots').index = \
            len(self.__robots) - 1

        # Place camera based on robots effective radius * 1.25
        if robot.robot is not None:
            radius = sum([abs(link.a) + abs(link.d) for link in robot.robot.links]) * 1.25
            self.scene.camera.pos = vector(radius, radius, radius) + get_pose_pos(robot.joints[1].get_pose())
            self.scene.camera.axis = vector(-radius, -radius, -radius)

    def delete_robot(self, robot):
        """
        This function is called when a new robot is to be deleted
        from the scene.

        :param robot: A graphical robot to add to the scene
        :type robot: class:`graphics.graphics_robot.GraphicalRobot`
        """
        if len(self.__robots) == 0 or robot not in self.__robots:
            return

        robot_index = self.__robots.index(robot)

        # Clear the robot visuals
        self.__robots[robot_index].set_reference_visibility(False)
        self.__robots[robot_index].set_robot_visibility(False)

        # Remove from UI
        new_list = []
        for name in self.__ui_controls.get('menu_robots').choices:
            new_list.append(name)

        del new_list[robot_index]
        del self.__robots[robot_index]
        del self.__teachpanel[robot_index]

        self.__selected_robot = 0
        # Update UI
        self.__reload_caption(new_list)
        # Select the top item
        if len(self.__ui_controls.get('menu_robots').choices) > 0:
            self.__ui_controls.get('menu_robots').index = 0

    def is_robot_in_canvas(self, robot):
        """
        Checks whether the given robot is in the canvas

        :param robot: A graphical robot to add to the scene
        :type robot: class:`graphics.graphics_robot.GraphicalRobot`
        """
        return robot in self.__robots

    def take_screenshot(self, filename):
        """
        Take a screenshot and save it
        """
        self.scene.capture(filename)

    #######################################
    #  UI Management
    #######################################
    def __add_mode_button(self):
        """
        Adds a button to the UI that toggles the UI mode
        """
        btn_text = self.__toggle_button_text_dict.get(
            self.__ui_mode, "Unknown Mode Set")
        btn_text = "<span style='font-size:20px;'>" + btn_text + "</span>"

        btn_toggle = button(bind=self.__toggle_mode, text=btn_text)
        self.__ui_controls.btn_toggle = btn_toggle
        self.scene.append_to_caption('\n')

    def __del_robot(self):
        """
        Remove a robot from the scene and the UI controls
        """
        if len(self.__robots) == 0:
            # Alert the user and return
            self.scene.append_to_caption(
                '<script type="text/javascript">alert'
                '("No robot to delete");</script>')
            return

        # Clear the robot visuals
        self.__robots[self.__selected_robot].set_reference_visibility(False)
        self.__robots[self.__selected_robot].set_robot_visibility(False)

        # Remove from UI
        new_list = []
        for name in self.__ui_controls.get('menu_robots').choices:
            new_list.append(name)

        del new_list[self.__selected_robot]
        del self.__robots[self.__selected_robot]
        del self.__teachpanel[self.__selected_robot]

        self.__selected_robot = 0
        # Update UI
        self.__reload_caption(new_list)
        # Select the top item
        if len(self.__ui_controls.get('menu_robots').choices) > 0:
            self.__ui_controls.get('menu_robots').index = 0

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

        # Weird manipulation to get correct vector directions.
        # (scene.camera.up always defaults to world up)
        cam_axis = (vector(self.scene.camera.axis))  # X
        cam_side_axis = self.scene.camera.up.cross(cam_axis)  # Y
        cam_up = cam_axis.cross(cam_side_axis)  # Z

        cam_up.mag = cam_axis.mag

        # Get a list of keys
        keys = keysdown()

        # Userspin uses ctrl, so skip this check to avoid changing camera pose
        # while ctrl is held
        if 'ctrl' in keys:
            return

        ######################################################################
        # PANNING
        # Check if the keys are pressed, update vectors as required
        # Changing camera position updates the scene center to
        # follow same changes
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

        # Update camera position before rotation
        # (to keep pan and rotate separate)
        self.scene.camera.pos = cam_pos

        ######################################################################
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

        ######################################################################
        # CAMERA ROTATION
        d = cam_distance
        move_dist = sqrt(d ** 2 + d ** 2 - 2 * d * d * cos(
            radians(rot_amount)))  # SAS Cosine

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

        :param new_list: The new list to apply to the menu
        :type new_list: `list`
        """
        # Remove all UI elements
        for item in self.__ui_controls:
            if self.__ui_controls.get(item) is None:
                continue
            self.__ui_controls.get(item).delete()
        for item in self.__teachpanel_sliders:
            item.delete()
        self.__teachpanel_sliders = []
        # Restore the caption
        self.scene.caption = self.__default_caption
        # Create the updated caption.
        self.__load_mode_ui(new_list)

    def __load_mode_ui(self, new_list):
        """
        Load the UI menu depending on the current mode

        :param new_list: The new list to apply to the menu
        :type new_list: `list`
        """
        self.__add_mode_button()
        if self.__ui_mode == UImode.CANVASCONTROL:
            self.__setup_ui_controls(new_list)
        elif self.__ui_mode == UImode.TEACHPANEL:
            self.__setup_joint_sliders()
        else:
            self.scene.append_to_caption("UNKNOWN MODE ENTERED\n")

    def __setup_ui_controls(self, list_of_names):
        """
        The initial configuration of the user interface

        :param list_of_names: A list of names of the robots in the screen
        :type list_of_names: `list`
        """

        ##################################################
        self.scene.append_to_caption('<h3>Scene Settings</h3>\n')
        # Button to reset camera
        reset_button = button(
            bind=self.__reset_camera, text="Reset Camera")
        self.__ui_controls.btn_reset = reset_button
        self.scene.append_to_caption('\t')

        screenshot_button = button(bind=self.__screenshot, text="Take Screenshot")
        self.__ui_controls.btn_ss = screenshot_button
        self.scene.append_to_caption('\n')

        camera_lock_checkbox = checkbox(bind=self.__camera_lock_checkbox, text="Camera Lock",
                                        checked=self.__camera_lock)
        self.__ui_controls.chkbox_cam = camera_lock_checkbox
        self.scene.append_to_caption('\t')

        grid_relative_checkbox = checkbox(
            bind=self.__grid_relative_checkbox,
            text="Grid Relative", checked=self.__grid_relative)
        self.__ui_controls.chkbox_rel = grid_relative_checkbox

        self.scene.append_to_caption('\t')
        # Checkbox for grid visibility
        checkbox_grid_visibility = checkbox(
            bind=self.__grid_visibility_checkbox, text="Grid Visibility",
            checked=self.__grid_visibility)
        self.__ui_controls.chkbox_grid = checkbox_grid_visibility
        self.scene.append_to_caption('\n')

        ##################################################
        self.scene.append_to_caption('<h3>Robot</h3>\n')
        # Drop down for robots / joints in frame
        menu_robots_list = menu(bind=self.__menu_item_chosen, choices=list_of_names)
        if not len(list_of_names) == 0:
            menu_robots_list.index = self.__selected_robot
        self.__ui_controls.menu_robots = menu_robots_list
        self.scene.append_to_caption('\t')

        # Button to delete the selected robot
        delete_button = button(bind=self.__del_robot, text="Delete Robot")
        self.__ui_controls.btn_del = delete_button
        self.scene.append_to_caption('\t')

        # Button to clear the robots in screen
        clear_button = button(bind=self.clear_scene, text="Clear Scene")
        self.__ui_controls.btn_clr = clear_button
        self.scene.append_to_caption('\n')

        ##################################################
        self.scene.append_to_caption('<h3>Characteristics</h3>\n')
        # Checkbox for reference frame visibilities
        if len(self.__robots) == 0:
            reference_checkbox = checkbox(
                bind=self.__reference_frame_checkbox,
                text="Show Reference Frames", checked=True)
        else:
            chk = self.__robots[self.__selected_robot].ref_shown
            reference_checkbox = checkbox(
                bind=self.__reference_frame_checkbox,
                text="Show Reference Frames", checked=chk)
        self.__ui_controls.chkbox_ref = reference_checkbox
        self.scene.append_to_caption('\t')

        # Checkbox for robot visibility
        if len(self.__robots) == 0:
            robot_vis_checkbox = checkbox(
                bind=self.__robot_visibility_checkbox,
                text="Show Robot", checked=True)
        else:
            chk = self.__robots[self.__selected_robot].rob_shown
            robot_vis_checkbox = checkbox(
                bind=self.__robot_visibility_checkbox,
                text="Show Robot", checked=chk)
        self.__ui_controls.chkbox_rob = robot_vis_checkbox
        self.scene.append_to_caption('\n')

        # Slider for robot opacity
        self.scene.append_to_caption('Robot Opacity:')
        if len(self.__robots) == 0:
            opacity_slider = slider(bind=self.__opacity_slider, value=1)
        else:
            opc = self.__robots[self.__selected_robot].opacity
            opacity_slider = slider(bind=self.__opacity_slider, value=opc)
        self.__ui_controls.sld_opc = opacity_slider
        self.scene.append_to_caption('\n')

        ##################################################
        # Control manual
        controls_str = '<h3>Controls</h3><br>' \
                       '<b>PAN</b><br>' \
                       'W , S | <i>forward / backward</i><br>' \
                       'A , D | <i>left / right</i><br>' \
                       'SPACE , SHIFT | <i>up / down</i><br>' \
                       '<b>ROTATE</b><br>' \
                       'CTRL + LMB | <i>free spin</i><br>' \
                       'ARROWS KEYS | <i>rotate direction</i><br>' \
                       'Q , E | <i>roll left / right</i><br>' \
                       '<b>ZOOM</b></br>' \
                       'MOUSEWHEEL | <i>zoom in / out</i>'

        self.scene.append_to_caption(controls_str)

    def __setup_joint_sliders(self):
        """
        Display the Teachpanel mode of the UI
        """
        self.scene.append_to_caption('\n')
        if len(self.__teachpanel) == 0:
            self.scene.append_to_caption("No robots available\n")
            return
        i = 0
        for joint in self.__teachpanel[self.__selected_robot]:
            if joint[self.__idx_qlim_min] == joint[self.__idx_qlim_max]:
                # If a slider with (effectively) no values, skip it
                i += 1
                continue
            # Add a title
            self.scene.append_to_caption('Joint {0}:\t'.format(i))
            # Add the slider, with the correct joint variables
            s = slider(
                bind=self.__joint_slider,
                min=joint[self.__idx_qlim_min],
                max=joint[self.__idx_qlim_max],
                value=joint[self.__idx_theta],
                id=i
            )
            self.__teachpanel_sliders.append(s)
            string = "{:.2f} rad ({:.2f} deg)".format(
                joint[self.__idx_theta], degrees(joint[self.__idx_theta]))
            joint[self.__idx_text] = wtext(text=string)
            self.scene.append_to_caption('\n\n')
            i += 1

    #######################################
    # UI CALLBACKS
    #######################################
    def __toggle_mode(self):
        """
        Callback for when the toggle mode button is pressed
        """
        # Update mode
        # Update mode, default canvas controls
        self.__ui_mode = {
            UImode.CANVASCONTROL: UImode.TEACHPANEL,
            UImode.TEACHPANEL: UImode.CANVASCONTROL
        }.get(self.__ui_mode, UImode.CANVASCONTROL)

        # Update UI
        # get list of robots
        new_list = []
        for name in self.__ui_controls.get('menu_robots').choices:
            new_list.append(name)

        self.__reload_caption(new_list)

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

        # Update the checkboxes/sliders for the selected robot
        self.__ui_controls.get('chkbox_ref').checked = \
            self.__robots[self.__selected_robot].ref_shown

        self.__ui_controls.get('chkbox_rob').checked = \
            self.__robots[self.__selected_robot].rob_shown

        self.__ui_controls.get('sld_opc').value = \
            self.__robots[self.__selected_robot].opacity

    def __reference_frame_checkbox(self, c):
        """
        When a checkbox is changed for the reference frame option, update the
        graphics

        :param c: The checkbox that has been toggled
        :type c: class:`checkbox`
        """
        if len(self.__robots) > 0:
            self.__robots[self.__selected_robot].set_reference_visibility(
                c.checked)

    def __robot_visibility_checkbox(self, c):
        """
        When a checkbox is changed for the robot visibility, update the
        graphics

        :param c: The checkbox that has been toggled
        :type c: class:`checkbox`
        """
        if len(self.__robots) > 0:
            self.__robots[self.__selected_robot].set_robot_visibility(
                c.checked)

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
        The callback for when a joint slider has changed value

        :param s: The slider object that has been modified
        :type s: class:`slider`
        """
        # Save the value
        self.__teachpanel[self.__selected_robot][s.id][self.__idx_theta] = \
            s.value

        # Get all angles for the robot
        angles = []
        for idx in range(len(self.__teachpanel_sliders)):
            angles.append(self.__teachpanel_sliders[idx].value)

        # Run fkine
        poses = self.__robots[self.__selected_robot].fkine(angles)

        # Update joints
        self.__robots[self.__selected_robot].set_joint_poses(poses)

        for joint in self.__teachpanel[self.__selected_robot]:
            if joint[self.__idx_text] is None:
                continue
            string = "{:.2f} rad ({:.2f} deg)".format(
                joint[self.__idx_theta], degrees(joint[self.__idx_theta]))
            joint[self.__idx_text].text = string

    def __screenshot(self, b):
        """
        Take a screencap
        """
        filename = "vp_ss_{:04d}.png".format(self._ss_tally)
        self.take_screenshot(filename)
        self._ss_tally += 1


class GraphicsCanvas2D:  # pragma nocover
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
    :param g_col: The RGB grid colour
    :type g_col: `list`, optional
    """

    def __init__(self, height=500, width=888, title='', caption='', grid=True, g_col=None):

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
        self.__grid_relative = True

        # Apply HTML title/caption
        if title != '':
            self.scene.title = title

        self.__default_caption = caption
        if caption != '':
            self.scene.caption = caption

        self.__ui_controls = UIMMap()
        self.__reload_caption()

        # Any time a key or mouse is held down, run the callback function
        rate(30)  # 30Hz
        self.scene.bind('keydown', self.__handle_keyboard_inputs)

        # Create the grid, and display if wanted
        self.__graphics_grid = GraphicsGrid(self.scene, colour=g_col)
        # Toggle grid to 2D
        # self.__graphics_grid.toggle_2d_3d()
        # Lock the grid
        # self.__graphics_grid.set_relative(False)
        # Turn off grid if applicable
        if not self.__grid_visibility:
            self.__graphics_grid.set_visibility(False)

        # Reset the camera to known spot
        self.__reset_camera()
        self.__graphics_grid.update_grid()
        self.__graphics_grid.toggle_2d_3d()

    #######################################
    #  Canvas Management
    #######################################
    def clear_scene(self):
        """
        Clear the scene of all objects, keeping the grid visible if set on
        """
        # Save grid visibility
        restore = self.__grid_visibility

        # Set invis
        if restore:
            self.__graphics_grid.set_visibility(False)

        # Set all objects invis
        for obj in self.scene.objects:
            obj.visible = False

        # Restore grid (if needed)
        if restore:
            self.__graphics_grid.set_visibility(True)

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

        shift + LMB = pan (Default Vpython)
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

        # Weird manipulation to get correct vector directions.
        # (scene.camera.up always defaults to world up)
        cam_axis = (vector(self.scene.camera.axis))  # X
        cam_side_axis = self.scene.camera.up.cross(cam_axis)  # Y
        cam_up = cam_axis.cross(cam_side_axis)  # Z

        cam_up.mag = cam_axis.mag

        # Get a list of keys
        keys = keysdown()

        # Userpan uses ctrl, so skip this check to avoid changing camera pose
        # while shift is held
        if 'shift' in keys:
            return

        #####################################################################
        # PANNING
        # Check if the keys are pressed, update vectors as required
        # Changing camera position updates the scene center to follow
        # same changes
        if 'w' in keys:
            cam_pos = cam_pos + cam_up * pan_amount
        if 's' in keys:
            cam_pos = cam_pos - cam_up * pan_amount
        if 'a' in keys:
            cam_pos = cam_pos + cam_side_axis * pan_amount
        if 'd' in keys:
            cam_pos = cam_pos - cam_side_axis * pan_amount

        # Update camera position before rotation
        # (to keep pan and rotate separate)
        self.scene.camera.pos = cam_pos

        #####################################################################
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

        ######################################################################
        # CAMERA ROTATION
        d = cam_distance
        move_dist = sqrt(d ** 2 + d ** 2 - 2 * d * d * cos(
            radians(rot_amount)))  # SAS Cosine

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
        # Ever so slightly off focus, to ensure grid is rendered in the right
        # region
        # (if directly at, draws numbers wrong spots)
        # Focus on (5, 5, 0)
        self.scene.camera.axis = vector(-0.001, -0.001, -12)
        self.scene.up = y_axis_vector

    def __reload_caption(self):
        """
        Reload the UI with the new list of robot names
        """
        # Remove all UI elements
        for item in self.__ui_controls:
            if self.__ui_controls.get(item) is None:
                continue
            self.__ui_controls.get(item).delete()
        # Restore the caption
        self.scene.caption = self.__default_caption
        # Create the updated caption.
        self.__setup_ui_controls()

    def __setup_ui_controls(self):
        """
        The initial configuration of the user interface
        """
        self.scene.append_to_caption('\n')

        # Button to reset camera
        btn_reset = button(
            bind=self.__reset_camera, text="Reset Camera")
        self.__ui_controls.btn_reset = btn_reset
        self.scene.append_to_caption('\t')

        chkbox_cam = checkbox(
            bind=self.__camera_lock_checkbox,
            text="Camera Lock", checked=self.__camera_lock)
        self.__ui_controls.chkbox_cam = chkbox_cam
        self.scene.append_to_caption('\t')

        chkbox_rel = checkbox(
            bind=self.__grid_relative_checkbox,
            text="Grid Relative", checked=self.__grid_relative)
        self.__ui_controls.chkbox_rel = chkbox_rel
        self.scene.append_to_caption('\n\n')

        # Button to clear the screen
        btn_clr = button(bind=self.clear_scene, text="Clear Scene")
        self.__ui_controls.btn_clear = btn_clr
        self.scene.append_to_caption('\n\n')

        # Checkbox for grid visibility
        chkbox_grid = checkbox(
            bind=self.__grid_visibility_checkbox, text="Grid Visibility",
            checked=self.__grid_visibility)
        self.__ui_controls.chkbox_grid = chkbox_grid
        self.scene.append_to_caption('\t')

        # Prevent the space bar from toggling the active checkbox/button/etc
        # (default browser behaviour)
        self.scene.append_to_caption('''
                       <script type="text/javascript">
                           $(document).keyup(function(event) {
                               if(event.which === 32) {
                                   event.preventDefault();
                               }
                           });
                       </script>''')
        # https://stackoverflow.com/questions/22280139/prevent-space-button-from-triggering-any-other-button-click-in-jquery

        # Control manual
        controls_str = '<br><b>Controls</b><br>' \
                       '<b>PAN</b><br>' \
                       'SHFT + LMB | <i>free pan</i><br>' \
                       'W , S | <i>up / down</i><br>' \
                       'A , D | <i>left / right</i><br>' \
                       '<b>ROTATE</b><br>' \
                       'ARROWS KEYS | <i>rotate direction</i><br>' \
                       'Q , E | <i>roll left / right</i><br>' \
                       '<b>ZOOM</b></br>' \
                       'MOUSEWHEEL | <i>zoom in / out</i><br>' \
                       '<script type="text/javascript">var arrow_keys_handler = function(e) {switch(e.keyCode){ case 37: case 39: case 38:  case 40: case 32: e.preventDefault(); break; default: break;}};window.addEventListener("keydown", arrow_keys_handler, false);</script>'  # noqa
        # Disable the arrow keys from scrolling in the browser
        # https://stackoverflow.com/questions/8916620/disable-arrow-key-scrolling-in-users-browser
        self.scene.append_to_caption(controls_str)

    #######################################
    # UI CALLBACKS
    #######################################
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

    def __grid_visibility_checkbox(self, c):
        """
        When a checkbox is changed for the grid visibility, update the graphics

        :param c: The checkbox that has been toggled
        :type c: class:`checkbox`
        """
        self.grid_visibility(c.checked)
        self.__grid_visibility = c.checked

    #######################################
    #  Drawing Functions
    #######################################
    def __draw_path(
            self, x_path, y_path, opt_line, opt_marker,
            opt_colour, thickness=0.05):
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

        # For every point in the list, draw a line to the next one
        # (excluding last point)
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
                create_line(
                    p1, p2, self.scene, colour=colour, thickness=thickness)
                # Only one marker to avoid double-ups
                create_marker(self.scene, x1, y1, opt_marker, colour)
            elif opt_line == '--':
                create_segmented_line(
                    p1, p2, self.scene, 0.3, colour=colour,
                    thickness=thickness)
                # Only one marker to avoid double-ups
                create_marker(self.scene, x1, y1, opt_marker, colour)
            elif opt_line == ':':
                create_segmented_line(
                    p1, p2, self.scene, 0.05, colour=colour,
                    thickness=thickness)
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
            raise ValueError(
                "Number of X coordinates does not equal "
                "number of Y coordinates.")

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

        :param options_str: The given options from the plot command to verify
        user input

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

        # If line_style given, join the first two options if applicable
        # (some types have 2 characters)
        for char in range(0, len(options_split) - 1):
            # If char is '-' (only leading character in double length option)
            if options_split[char] == '-' and len(options_split) > 1:
                # If one of the leading characters is valid
                if options_split[char + 1] == '-' or \
                        options_split[char + 1] == '.':
                    # Join the two into the first
                    options_split[char] = options_split[char] \
                                          + options_split[char + 1]
                    # Shuffle down the rest
                    for idx in range(char + 2, len(options_split)):
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
        # Index position of index used (only used when count == 1)
        line_style_index = 0
        for option in options_split:
            if option in self.__line_styles:
                line_style_count = line_style_count + 1
                line_style_index = self.__line_styles.index(option)

        # If more than one, throw error
        if line_style_count > 1:
            raise ValueError(
                "Too many line style arguments given. Only one allowed")
        # If none, set as solid
        elif line_style_count == 0 or not any(
                item in options_split for item in self.__line_styles):
            output_line = default_line
        # If one, set as given
        else:
            output_line = self.__line_styles[line_style_index]
        ##############################

        ##############################
        # Verify Marker Style
        ##############################
        marker_style_count = 0  # Count of options used
        # Index position of index used (only used when count == 1)
        marker_style_index = 0
        for option in options_split:
            if option in self.__marker_styles:
                marker_style_count = marker_style_count + 1
                marker_style_index = self.__marker_styles.index(option)

        # If more than one, throw error
        if marker_style_count > 1:
            raise ValueError(
                "Too many marker style arguments given. Only one allowed")
        # If none, set as no-marker
        elif marker_style_count == 0 or not any(
                item in options_split for item in self.__marker_styles):
            output_marker = default_marker
        # If one, set as given
        else:
            output_marker = self.__marker_styles[marker_style_index]
            # If marker set and no line given, turn line to no-line
            if line_style_count == 0 or not any(
                    item in options_split for item in self.__line_styles):
                output_line = ''
        ##############################

        ##############################
        # Verify Colour Style
        ##############################
        colour_style_count = 0  # Count of options used
        # Index position of index used (only used when count == 1)
        colour_style_index = 0
        for option in options_split:
            if option in self.__colour_styles:
                colour_style_count = colour_style_count + 1
                colour_style_index = self.__colour_styles.index(option)

        # If more than one, throw error
        if colour_style_count > 1:
            raise ValueError(
                "Too many colour style arguments given. Only one allowed")
        # If none, set as black
        elif colour_style_count == 0 or not any(
                item in options_split for item in self.__colour_styles):
            output_colour = default_colour
        # If one, set as given
        else:
            output_colour = self.__colour_styles[colour_style_index]
        ##############################

        return [output_line, output_marker, output_colour]

    def __get_colour_from_string(self, colour_string):
        """
        Using the colour plot string input, return an rgb array of the colour
        selected

        :param colour_string: The colour string option
        :type colour_string: `str`
        :returns: List of RGB values for the representative colour
        :rtype: `list`
        """
        # Return the RGB list (black if not in dictionary)
        return self.__colour_dictionary.get(colour_string, color.black.value)


def convert_grid_to_z_up(scene):  # pragma nocover
    """
    Rotate the camera so that +z is up
    (Default vpython scene is +y up)
    """

    '''
    There is an interaction between up and forward, the direction that the
    camera is pointing. By default, the camera points in the -z direction
    vector(0,0,-1). In this case, you can make the x or y axes (or anything
    between) be the up vector, but you cannot make the z axis be the up
    vector, because this is the axis about which the camera rotates when
    you set the up attribute. If you want the z axis to point up, first set
    forward to something other than the -z axis, for example vector(1,0,0).
    https://www.glowscript.org/docs/VPythonDocs/canvas.html
    '''
    # First set the x-axis forward
    scene.forward = x_axis_vector
    scene.up = z_axis_vector

    # Place the camera in the + axes
    scene.camera.pos = vector(10, 10, 10)
    scene.camera.axis = -scene.camera.pos
    return


def draw_reference_frame_axes(se3_pose, scene):  # pragma nocover
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
    x_arrow = arrow(
        canvas=scene, pos=origin, axis=x_axis_vector,
        length=0.25, color=color.red)

    # Draw Y Axis
    y_arrow = arrow(
        canvas=scene, pos=origin, axis=y_axis_vector,
        length=0.25, color=color.green)

    # Draw Z Axis
    z_arrow = arrow(
        canvas=scene, pos=origin, axis=z_axis_vector,
        length=0.25, color=color.blue)

    # Combine all to manipulate together
    # Set origin to where axis converge (instead of the middle of
    # the resulting object bounding box)
    frame_ref = compound(
        [x_arrow, y_arrow, z_arrow], origin=origin, canvas=scene)

    # Set frame axes
    frame_ref.axis = x_axis
    frame_ref.up = y_axis

    return frame_ref
