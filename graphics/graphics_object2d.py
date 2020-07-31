from vpython import shapes, paths, extrusion, vector


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
            circ = shapes.circle(pos=[0, 0], radius=0.5)
            # Display the shape
            obj = extrusion(path=[vector(1, 1, 0.001), vector(1, 1, -0.001)], shape=circ, shininess=0)
        elif self.__shape == 's':
            raise NotImplementedError()
        elif self.__shape == 't':
            raise NotImplementedError()
        elif self.__shape == 'p':
            raise NotImplementedError()
        elif self.__shape == 'o':
            raise NotImplementedError()
        elif self.__shape == 'h':
            raise NotImplementedError()
        elif self.__shape == 'a':
            raise NotImplementedError()
        elif self.__shape == 'r':
            raise NotImplementedError()
        elif self.__shape == '*':
            raise NotImplementedError()
        elif self.__shape == '+':
            raise NotImplementedError()
        else:
            raise ValueError("Invalid shape given")

        return obj
