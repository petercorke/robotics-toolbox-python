#!/usr/bin/env python
"""
@author Micah Huth
"""

from vpython import shapes, radians, extrusion, vector
from roboticstoolbox.backends.VPython.stl import import_object_from_numpy_stl


class Object2D:   # pragma nocover
    """
    This object will allow the ability to update it's pose in the scene.
    For example, it could resemble a car or drone that moves around in
    2D space.

    :param se2: The SE2 object representing position and orientation
    :type se2: class:`spatialmath.se2`
    :param scene: The scene in which to add the link
    :type scene: class:`vpython.canvas`
    :param shape: The shape of the object
    :type shape: `str`
    :param colour: The colour of the shape
    :type colour: `list`
    :raises ValueError: The shape must be in the list of possible shapes
    """
    def __init__(self, se2, scene, shape, colour):
        # Save inputs
        self.se2 = se2
        self.scene = scene
        self.shape = shape
        self.size = 0
        if colour[0] > 1.0 or colour[1] > 1.0 or colour[2] > 1.0 or \
                colour[0] < 0.0 or colour[1] < 0.0 or colour[2] < 0.0:
            raise ValueError("RGB values must be normalised between 0 and 1")
        self.colourVec = vector(colour[0], colour[1], colour[2])
        self.graphic = None

    def __create_object(self):
        """
        To be overridden by child classes
        """
        pass

    def update_pose(self, new_se2):
        """
        Update the pose of the object

        :param new_se2: The SE2 representation of the pose
        :type new_se2: class:`spatialmath.se2`
        """
        self.se2 = new_se2
        x = self.se2.t[0]
        y = self.se2.t[1]
        t = self.se2.theta
        self.graphic.pos = vector(x, y, 0)
        self.graphic.axis = vector(0, 1, 0).rotate(t)

    def update_colour(self, colour):
        """
        Update the colour of the object

        :param colour: The RGB colour
        :type colour: `list`
        """
        if colour[0] > 1.0 or colour[1] > 1.0 or colour[2] > 1.0 or \
                colour[0] < 0.0 or colour[1] < 0.0 or colour[2] < 0.0:
            raise ValueError("RGB values must be normalised between 0 and 1")
        self.graphic.color = vector(colour[0], colour[1], colour[2])
        self.colourVec = vector(colour[0], colour[1], colour[2])

    def update_visibility(self, is_visible):
        """
        Update the visibility of the object

        :param is_visible: Whether to draw or not
        :type is_visible: `bool`
        """
        self.graphic.visible = is_visible

    def update_size(self, multiply):
        """
        Update the size of the object by a multiple of the original size.

        :param multiply: A number to multiply the original size by
        :type multiply: `float`, `int`
        """
        self.graphic.size = self.size * multiply


class STL2D(Object2D):   # pragma nocover
    """
    This object is for 2D objects that contain an STL.

    :param se2: The SE2 object representing position and orientation
    :type se2: class:`spatialmath.se2`
    :param scene: The scene in which to add the link
    :type scene: class:`vpython.canvas`
    :param stl_path: The file path to the STL to apply
    :type stl_path: `str`
    :param colour: The colour of the shape
    :type colour: `list`
    """
    def __init__(self, se2, scene, stl_path, colour):
        super().__init__(se2, scene, stl_path, colour)
        self.graphic = self.__create_object()
        self.size = self.graphic.size
        self.graphic.color = self.colourVec

    def __create_object(self):
        """
        Return a compound of the loaded STL

        :return: A compound object of the triangles in the STL
        :rtype: class:`vpython.compound`
        """
        return import_object_from_numpy_stl(self.shape, self.scene)


class Marker2D(Object2D):   # pragma nocover
    """
    This class will place a marker in the given location based on the given
    marker inputs

    :param se2: The SE2 object representing position and orientation
    :type se2: class:`spatialmath.se2`
    :param scene: The scene in which to add the link
    :type scene: class:`vpython.canvas`
    :param shape: The shape of the object
    :type shape: `str`
    :param colour: The colour of the shape
    :type colour: `list`
    :raises ValueError: The shape must be in the list of possible shapes
    """
    def __init__(self, se2, scene, shape, colour):
        super().__init__(se2, scene, shape, colour)

        self.__marker_size = 0.2

        marker_styles = [
            '',  # None
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

        if shape not in marker_styles:
            raise ValueError(
                "The shape must be in the list of possible shapes")

        # Create the object
        self.graphic = self.__create_object()

    def __create_object(self):
        """
                Create the physical graphical object

                :returns: The graphical entity
                :rtype: class:`vpython.baseobj`
                """
        if self.shape == '':
            # 2D coords of the circle boundary
            shape_path = shapes.circle(radius=self.__marker_size / 2)
        elif self.shape == '+':
            # 2D coords of the cross boundary
            shape_path = shapes.cross(
                width=self.__marker_size, thickness=self.__marker_size / 5)
        elif self.shape == 'o':
            # 2D coords of the circle boundary
            shape_path = shapes.circle(radius=self.__marker_size / 2)
        elif self.shape == '*':
            # 2D coords of the star boundary
            shape_path = shapes.star(radius=self.__marker_size / 2, n=6)
        elif self.shape == '.':
            # 2D coords of the square boundary
            shape_path = shapes.rectangle(
                width=self.__marker_size, height=self.__marker_size)
        elif self.shape == 'x':
            # 2D coords of the cross boundary
            shape_path = shapes.cross(
                width=self.__marker_size, thickness=self.__marker_size / 5,
                rotate=radians(45))
        elif self.shape == 's':
            # 2D coords of the square boundary
            shape_path = shapes.rectangle(
                width=self.__marker_size, height=self.__marker_size,
                thickness=self.__marker_size)
        elif self.shape == 'd':
            # 2D coords of the diamond boundary
            shape_path = shapes.rectangle(
                width=self.__marker_size, height=self.__marker_size,
                thickness=self.__marker_size, rotate=radians(45))
        elif self.shape == '^':
            # 2D coords of the triangle boundary
            shape_path = shapes.triangle(
                length=self.__marker_size)
        elif self.shape == 'v':
            # 2D coords of the triangle boundary
            shape_path = shapes.triangle(
                length=self.__marker_size, rotate=radians(180))
        elif self.shape == '<':
            # 2D coords of the triangle boundary
            shape_path = shapes.triangle(
                length=self.__marker_size, rotate=radians(90))
        elif self.shape == '>':
            # 2D coords of the triangle boundary
            shape_path = shapes.triangle(
                length=self.__marker_size, rotate=radians(-90))
        elif self.shape == 'p':
            # 2D coords of the pentagon boundary
            shape_path = shapes.pentagon(
                length=self.__marker_size)
        elif self.shape == 'h':
            # 2D coords of the hexagon boundary
            shape_path = shapes.hexagon(
                length=self.__marker_size)
        # CURRENTLY UNUSED
        # elif self.__shape == 'o':
        #     # 2D coords of the octagon boundary
        #     shape_path = shapes.octagon(length=1)
        # elif self.__shape == 'r':
        #     # 2D coords of the ring boundary (with thickness = 10%)
        #     shape_path = shapes.circle(radius=0.5, thickness=0.1)
        else:
            raise ValueError("Invalid shape given")

        # Create the shape
        x = self.se2.t[0]
        y = self.se2.t[1]
        obj = extrusion(scene=self.scene,
                        path=[vector(x, y, 0.001), vector(x, y, -0.001)],
                        shape=shape_path,
                        color=self.colourVec,
                        shininess=0)
        self.size = obj.size
        if self.shape == '':
            obj.visible = False
        return obj
