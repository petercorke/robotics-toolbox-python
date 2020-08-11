from vpython import shapes, radians, extrusion, vector

class Object2D:
    """
    This object will allow the ability to update it's pose in the scene.
    For example, it could resemble a car or drone that moves around in 2D space.

    :param se2: The SE2 object representing position and orientation
    :type se2: class:`spatialmath.se2`
    :param g_canvas: The canvas in which to add the link
    :type g_canvas: class:`graphics.graphics_canvas.graphicscanvas3d`
    :param shape: The shape of the object
    :type shape: `str`
    :param colour: The colour of the shape
    :type colour: `list`
    :raises ValueError: The shape must be in the list of possible shapes
    """

    # TODO
    #  Have options to give stl file, shape, or arrow
    #  Have option to set size, colour, etc
    #  Have funcs that update pose, texture/colours, visibility

    def __init__(self, se2, g_canvas, shape, colour):
        # Save inputs
        self.__se2 = se2
        self.__scene = g_canvas.scene
        self.__shape = shape
        if colour[0] > 1.0 or colour[1] > 1.0 or colour[2] > 1.0 or \
                colour[0] < 0.0 or colour[1] < 0.0 or colour[2] < 0.0:
            raise ValueError("RGB values must be normalised between 0 and 1")
        self.__colour = vector(colour[0], colour[1], colour[2])

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
            raise ValueError("The shape must be in the list of possible shapes")

        # Create the object
        self.__graphic = self.__create_object()

    def __create_object(self):
        """
        Create the physical graphical object

        :returns: The graphical entity
        :rtype: class:`vpython.baseobj`
        """
        if self.__shape == '':
            # 2D coords of the circle boundary
            shape_path = shapes.circle(radius=self.__marker_size/2)
        elif self.__shape == '+':
            # 2D coords of the cross boundary
            shape_path = shapes.cross(width=self.__marker_size, thickness=self.__marker_size/5)
        elif self.__shape == 'o':
            # 2D coords of the circle boundary
            shape_path = shapes.circle(radius=self.__marker_size/2)
        elif self.__shape == '*':
            # 2D coords of the star boundary
            shape_path = shapes.star(radius=self.__marker_size/2, n=6)
        elif self.__shape == '.':
            # 2D coords of the square boundary
            shape_path = shapes.rectangle(width=self.__marker_size, height=self.__marker_size)
        elif self.__shape == 'x':
            # 2D coords of the cross boundary
            shape_path = shapes.cross(width=self.__marker_size, thickness=self.__marker_size/5, rotate=radians(45))
        elif self.__shape == 's':
            # 2D coords of the square boundary
            shape_path = shapes.rectangle(width=self.__marker_size, height=self.__marker_size,
                                          thickness=self.__marker_size/10)
        elif self.__shape == 'd':
            # 2D coords of the diamond boundary
            shape_path = shapes.rectangle(width=self.__marker_size, height=self.__marker_size,
                                          thickness=self.__marker_size/10, rotate=radians(45))
        elif self.__shape == '^':
            # 2D coords of the triangle boundary
            shape_path = shapes.triangle(length=self.__marker_size)
        elif self.__shape == 'v':
            # 2D coords of the triangle boundary
            shape_path = shapes.triangle(length=self.__marker_size, rotate=radians(180))
        elif self.__shape == '<':
            # 2D coords of the triangle boundary
            shape_path = shapes.triangle(length=self.__marker_size, rotate=radians(90))
        elif self.__shape == '>':
            # 2D coords of the triangle boundary
            shape_path = shapes.triangle(length=self.__marker_size, rotate=radians(-90))
        elif self.__shape == 'p':
            # 2D coords of the pentagon boundary
            shape_path = shapes.pentagon(length=self.__marker_size)
        elif self.__shape == 'h':
            # 2D coords of the hexagon boundary
            shape_path = shapes.hexagon(length=self.__marker_size)
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
        x = self.__se2.t[0]
        y = self.__se2.t[1]
        obj = extrusion(scene=self.__scene,
                        path=[vector(x, y, 0.001), vector(x, y, -0.001)],
                        shape=shape_path,
                        color=self.__colour,
                        shininess=0)
        if self.__shape == '':
            obj.visible = False
        return obj
