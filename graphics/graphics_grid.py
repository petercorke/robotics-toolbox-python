from graphics.graphics_text import *


class GraphicsGrid:
    """
    This class holds the current grid displayed in the canvas
    """
    def __init__(self):
        # Save the current camera settings
        self.camera_pos = scene.camera.pos
        self.camera_axes = scene.camera.axis
        # Initialise a grid object
        self.grid_object = None

    def draw_grid(self):
        """
        Display grids along the x, y, z axes.

        """

        num_squares = 10  # Length of the grid in each direction (in units)
        relative_cam = True  # Whether the grid follows the camera rotation and movement

        the_grid = self.create_grid(relative_cam, num_squares)
        the_numbers = create_grid_numbers(relative_cam, num_squares)

        self.grid_object = [the_grid, the_numbers]

    def create_grid(self, bool_camera_relative, num_squares):
        """
        Draw a grid along each 3D plane, that is closest to the camera.

        :param bool_camera_relative: Whether to draw the axes at the camera focus point or at (0, 0, 0).
        :type bool_camera_relative: bool
        :param num_squares: How many unit squares to draw along the axis.
        :type num_squares: int
        :return: Vpython compound object of the three drawn axes.
        :rtype: class:`vpython.compound`
        """

        # Initial conditions
        xz_lines = []
        xy_lines = []
        yz_lines = []
        camera_axes = self.camera_axes
        # Locate centre of axes
        if bool_camera_relative:
            x_origin, y_origin, z_origin = round(scene.center.x), round(scene.center.y), round(scene.center.z)
        else:
            x_origin, y_origin, z_origin = 0, 0, 0

        #   CAMERA AXES |  DISPLAYED GRID | XZ PLANE | XY PLANE | YZ PLANE
        #      x,y,z    |      x,y,z      |   x,z    |    x,y   |    y,z
        #  -------------+-----------------+----------+----------+----------
        #      -,-,-    |      +,+,+      |   +,+    |    +,+   |    +,+
        #      -,-,+    |      +,+,-      |   +,-    |    +,+   |    +,-
        #      -,+,-    |      +,-,+      |   +,+    |    +,-   |    -,+
        #      -,+,+    |      +,-,-      |   +,-    |    +,-   |    -,-
        #      +,-,-    |      -,+,+      |   -,+    |    -,+   |    +,+
        #      +,-,+    |      -,+,-      |   -,-    |    -,+   |    +,-
        #      +,+,-    |      -,-,+      |   -,+    |    -,-   |    -,+
        #      +,+,+    |      -,-,-      |   -,-    |    -,-   |    -,-
        # min = -num_squares or 0, around the default position
        # max = +num_squares or 0, around the default position
        # e.g. at the origin, for negative axes: -10 -> 0, positive axes: 0 -> 10
        min_x_coord = x_origin + int(-(num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2))
        max_x_coord = x_origin + int((num_squares / 2) + (sign(camera_axes.x) * -1) * (num_squares / 2))

        min_y_coord = y_origin + int(-(num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2))
        max_y_coord = y_origin + int((num_squares / 2) + (sign(camera_axes.y) * -1) * (num_squares / 2))

        min_z_coord = z_origin + int(-(num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2))
        max_z_coord = z_origin + int((num_squares / 2) + (sign(camera_axes.z) * -1) * (num_squares / 2))

        # XZ plane
        for x_point in range(min_x_coord, max_x_coord + 1):
            # Draw a line across for each x coord, along the same y-axis, from min to max z coord
            xz_lines.append(create_line(
                vector(x_point, y_origin, min_z_coord),
                vector(x_point, y_origin, max_z_coord)
            ))
        for z_point in range(min_z_coord, max_z_coord + 1):
            # Draw a line across each z coord, along the same y-axis, from min to max z coord
            xz_lines.append(create_line(
                vector(min_x_coord, y_origin, z_point),
                vector(max_x_coord, y_origin, z_point)
            ))

        # XY plane
        for x_point in range(min_x_coord, max_x_coord + 1):
            # Draw a line across each x coord, along the same z-axis, from min to max y coord
            xy_lines.append(create_line(
                vector(x_point, min_y_coord, z_origin),
                vector(x_point, max_y_coord, z_origin)
            ))
        for y_point in range(min_y_coord, max_y_coord + 1):
            # Draw a line across each y coord, along the same z-axis, from min to max x coord
            xy_lines.append(create_line(
                vector(min_x_coord, y_point, z_origin),
                vector(max_x_coord, y_point, z_origin)
            ))

        # YZ plane
        for y_point in range(min_y_coord, max_y_coord + 1):
            # Draw a line across each y coord, along the same x-axis, from min to max z coord
            yz_lines.append(create_line(
                vector(x_origin, y_point, min_z_coord),
                vector(x_origin, y_point, max_z_coord)
            ))
        for z_point in range(min_z_coord, max_z_coord + 1):
            # Draw a line across each z coord, along the same x-axis, from min to max y coord
            yz_lines.append(create_line(
                vector(x_origin, min_y_coord, z_point),
                vector(x_origin, max_y_coord, z_point)
            ))

        # Compound the lines together into respective objects
        xz_plane = compound(xz_lines)
        xy_plane = compound(xy_lines)
        yz_plane = compound(yz_lines)

        # Combine all into one object
        grid = compound([xy_plane, xz_plane, yz_plane])

        return grid

    def update_grid(self):
        """
        Update the grid axes and numbers if the camera position/rotation has changed.

        """

        # Obtain the new camera settings
        new_camera_pos = scene.camera.pos
        new_camera_axes = scene.camera.axis

        old_camera_pos = self.camera_pos
        old_camera_axes = self.camera_axes

        # If camera is different to previous: update
        if (not new_camera_axes.equals(old_camera_axes)) or (not new_camera_pos.equals(old_camera_pos)):
            # Update old positions
            self.camera_pos = new_camera_pos
            self.camera_axes = new_camera_axes

            # Delete old grid
            self.set_visibility(False)
            self.grid_object = None

            # Save new grid
            self.draw_grid()
        # Else save current grid
        else:
            # Already current
            pass

    def set_visibility(self, is_visible):
        """
        Set the visibility of the grid

        :param is_visible: Boolean of whether to display the grid
        :type is_visible: bool
        """
        # If no grid, turn the objects invisible
        self.grid_object[0].visible = is_visible
        for number in self.grid_object[1]:
            number.visible = is_visible


def create_line(pos1, pos2):
    """
    Create a line from position 1 to position 2.

    :param pos1: 3D position of one end of the line.
    :type pos1: class:`vpython.vector`
    :param pos2: 3D position of the other end of the line.
    :type pos2: class:`vpython.vector`
    """

    # Length of the line using the magnitude
    line_len = mag(pos2-pos1)

    # Position of the line is the midpoint (centre) between the ends
    position = (pos1 + pos2) / 2

    # Axis direction of the line (to align the box (line) to intersect the two points)
    axis_dir = pos2 - pos1

    # Return a box of thin width and height to resemble a line
    thickness = 0.01
    return box(pos=position, axis=axis_dir, length=line_len, width=thickness, height=thickness, color=color.black)
