from vpython import shapes, arrow, extrusion, vector


class Object2D:
    """
    This object will allow the ability to update it's pose in the scene.
    For example, it could resemble a car or drone that moves around in 2D space.

    :param se2: The SE2 object representing position and orientation
    :type se2: class:`spatialmath.se2`
    :param scene: The scene in which to draw the object
    :type scene: class:`vpython.canvas`
    :param shape: The shape of the object
    :type shape: `str`
    :raises ValueError: The shape must be in the list of possible shapes
    """

    # TODO
    #  Have options to give stl file, shape, or arrow (default)
    #  Have option to set size, colour, etc
    #  Have funcs that update pose, texture/colours, visibility

    def __init__(self, se2, scene, shape):
        # Save inputs
        self.__se2 = se2
        self.__scene = scene
        self.__shape = shape

        possible_shapes = [
            'c',  # circle
            's',  # square
            't',  # triangle
            'p',  # pentagon
            'h',  # hexagon
            'o',  # octagon
            'a',  # arrow
            'r',  # ring
            '*',  # star
            '+',  # cross
        ]

        if shape not in possible_shapes:
            raise ValueError("The shape must be in the list of possible shapes")

        # Create the object
        self.__graphic = self.__create_object()

    def __create_object(self):
        """
        Create the physical graphical object

        :returns: The graphical entity
        :rtype: class:`vpython.baseobj`
        """
        # Circle
        if self.__shape == 'c':
            # 2D coords of the circle boundary
            shape_path = shapes.circle(radius=0.5)
        elif self.__shape == 's':
            # 2D coords of the square boundary
            shape_path = shapes.rectangle(width=1, height=1)
        elif self.__shape == 't':
            # 2D coords of the triangle boundary
            shape_path = shapes.triangle(length=1)
        elif self.__shape == 'p':
            # 2D coords of the pentagon boundary
            shape_path =shapes.pentagon(length=1)
        elif self.__shape == 'h':
            # 2D coords of the hexagon boundary
            shape_path = shapes.hexagon(length=1)
        elif self.__shape == 'o':
            # 2D coords of the octagon boundary
            shape_path = shapes.octagon(length=1)
        elif self.__shape == 'a':
            # Create an arrow object and return it
            obj = arrow(pos=vector(1, 1, 0), axis=vector(1, 1, 0), shaftwidth=0.01)
            return obj
        elif self.__shape == 'r':
            # 2D coords of the circle boundary (with thickness = 0.1*r)
            shape_path = shapes.circle(radius=0.5, thickness=0.1)
        elif self.__shape == '*':
            # 2D coords of the star boundary
            shape_path = shapes.star(radius=0.5, n=6)
        elif self.__shape == '+':
            # 2D coords of the cross boundary
            shape_path = shapes.cross(width=5, thickness=1)
        else:
            raise ValueError("Invalid shape given")

        # Create the shape
        obj = extrusion(path=[vector(1, 1, 0.001), vector(1, 1, -0.001)], shape=shape_path, shininess=0)
        return obj
