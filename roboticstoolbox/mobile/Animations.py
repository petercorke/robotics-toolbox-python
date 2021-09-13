"""
Python Vehicle
@Author: Kristian Gibson
@Author: Peter Corke
"""
from abc import ABC, abstractmethod
import warnings
from math import pi, sin, cos, tan, atan2
import numpy as np
from scipy import integrate, linalg, interpolate
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches, colors
import matplotlib.transforms as mtransforms

from spatialmath import SE2, base
from roboticstoolbox import rtb_load_data

"""
Class to support animation of a vehicle on Matplotlib plot

There are three concrete subclasses:

- ``VehicleMarker`` animates a Matplotlib marker
- ``VehiclePolygon`` animates a polygon, including predefined shapes
- ``VehicleIcon`` animates an image



These can be used in two different ways, firstly::

    a.add()
    a.update(q)

adds an instance of the animation shape to the plot and subsequent calls
to ``update`` will animate it.

Secondly::

    a.plot(q)

adds an instance of the animation shape to the plot with the specified
configuration.  It cannot be moved, but the method does return a reference
to the Matplotlib object added to the plot.

Any of these can be passed to a Vehicle subclass object to make an animation
during simulation::

        a = VehiclePolygon('car', color='r')
        veh = Bicycle()
        veh.run(animation=a)

    """

class VehicleAnimationBase(ABC):
    """
    Class to support animation of a vehicle on Matplotlib plot

    There are three concrete subclasses:

    - ``VehicleMarker`` animates a Matplotlib marker
    - ``VehiclePolygon`` animates a polygon, including predefined shapes
    - ``VehicleIcon`` animates an image

    These can be used in two different ways, firstly::

        a.add()
        a.update(q)
    
    adds an instance of the animation shape to the plot and subsequent calls
    to ``update`` will animate it.

    Secondly::

        a.plot(q)

    adds an instance of the animation shape to the plot with the specified
    configuration.  It cannot be moved, but the method does return a reference
    to the Matplotlib object added to the plot.

    Any of these can be passed to a Vehicle subclass object to make an animation
    during simulation::

            a = VehiclePolygon('car', color='r')
            veh = Bicycle()
            veh.run(animation=a)

    """
    
    def __init__(self):
        self._object = None
        self._ax = None

    def add(self, ax=None, **kwargs):
        """
        Add vehicle animation to the current plot

        :param ax: Axis to add to, defaults to current axis
        :type ax: Axes, optional

        A reference to the animation object is kept, and it will be deleted
        from the plot when the ``VehicleAnimation`` object is garbage collected.
        """
        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

        self._add(**kwargs)

    def update(self, q):
        """
        Update the vehicle animation

        :param q: vehicle configuration
        :type q: ndarray(2) or ndarray(3)

        The graphical depiction of the vehicle position or pose is updated.

        For ``AnimationMarker`` only position can be depicted so the third element
        of ``q``, if given, is ignored.
        """
        self._update(q)

    def plot(self, q, **kwargs):
        """
        Add vehicle to the current plot

        :param q: vehicle configuration
        :type q: ndarray(2) or ndarray(3)
        :return: reference to Matplotlib object

        A reference to the animation object is kept, and it will be deleted
        from the plot when the ``VehicleAnimation`` object is garbage collected.
        """
        self.add(**kwargs)
        self.update(q)
        return self._object

    def __del__(self):

        if self._object is not None:
            self._object.remove()

# ========================================================================= #

class VehicleMarker(VehicleAnimationBase):

    def __init__(self, **kwargs):
        """
        Create graphical animation of vehicle as a Matplotlib marker

        :param kwargs: additional arguments passed to matplotlib ``plot``.
        :return: animation object
        :rtype: VehicleAnimation

        Creates an object that can be passed to a ``Vehicle`` subclass to depict
        the moving robot as a simple matplotlib marker during simulation.

        For example, a blue filled square, is::

            a = VehicleMarker(marker='s', markerfacecolor='b')
            veh = Bicycle(driver=RandomPath(10), animation=a)
            veh.run()

        .. note:: A marker can only indicate vehicle position, not orientation.

        :seealso: :func:`~Vehicle`
        """
        super().__init__()
        if len(kwargs) == 0:
            kwargs = {
                'marker': 'o',
                'markerfacecolor': 'r',
                'markeredgecolor': 'w',
                'markersize': 12
            }
        self._args = kwargs
        

    def _update(self, x):
        self._object.set_xdata(x[0])
        self._object.set_ydata(x[1])

    def _add(self, x=None, **kwargs):
        if x is None:
            x = (0, 0)
        self._object = plt.plot(x[0], x[1], **self._args)[0]

# ========================================================================= #

class VehiclePolygon(VehicleAnimationBase):

    def __init__(self, shape='car', scale=1, **kwargs):
        """
        Create graphical animation of vehicle as a polygon

        :param shape: polygon shape as vertices or a predefined shape, defaults to 'car'
        :type shape: ndarray(2,n) or str
        :param scale: Length of the vehicle on the plot, defaults to 1
        :type scale: float
        :param kwargs: additional arguments passed to matplotlib patch such as
            ``color`` (face+edge), ``alpha``, ``facecolor``, ``edgecolor``,
            ``linewidth`` etc.
        :raises ValueError: unknown shape name
        :raises TypeError: bad shape argument
        :return: animation object
        :rtype: VehicleAnimation

        Creates an object that can be passed to a ``Vehicle`` subclass to 
        depict the moving robot as a polygon during simulation.

        For example, a red filled car-shaped polygon is::

            a = VehiclePolygon('car', color='r')
            veh = Bicycle()
            veh.run(animation=a)

        ``shape`` can be:

            * ``"car"``  a rectangle with chamfered front corners
            * ``"triangle"`` a triangle
            * an Nx2 NumPy array of vertices, does not have to be closed.

        The polygon is scaled to an image with a length of ``scale`` in the units of
        the plot.

        :seealso: :func:`~Vehicle`
        """
        super().__init__()
        if isinstance(shape, str):

            h = 0.3
            t = 0.8  # start of head taper
            c = 0.5  # centre x coordinate
            w = 1    # width in x direction

        if isinstance(shape, str):
            if shape == 'car':
                self._coords = np.array([
                    [-c,     h],
                    [t - c,  h],
                    [w - c,  0],
                    [t - c, -h],
                    [-c,    -h],
                ]).T
            elif shape == 'triangle':
                self._coords = np.array([
                    [-c,  h],
                    [ w,  0],
                    [-c, -h],
                ]).T
            else:
                raise ValueError('unknown vehicle shape name')
            
        elif isinstance(shape, np.ndarray) and shape.shape[1] == 2:
            self._coords = shape
        else:
            raise TypeError('unknown shape argument')
        self._coords *= scale
        self._args = kwargs

    def _add(self, **kwargs):
        # color is fillcolor + edgecolor
        # facecolor if None is default
        self._ax = plt.gca()
        self._object = patches.Polygon(self._coords.T, **{**self._args, **kwargs})
        self._ax.add_patch(self._object)

    def _update(self, x):
        
        if self._object is not None:
            # if animation is initialized
            xy = SE2(x) * self._coords
            self._object.set_xy(xy.T)

# ========================================================================= #

class VehicleIcon(VehicleAnimationBase):

    def __init__(self, filename, origin=None, scale=1, rotation=0):
        """
        Create graphical animation of vehicle as an icon

        :param filename: Standard icon name or a path to an image
        :type filename: str
        :param origin: Origin of the vehicle coordinate frame, defaults to centre
        :type origin: 2-tuple
        :param scale: Length of the vehicle on the plot, defaults to 1
        :type scale: float
        :param rotation: Vehicle icon heading in degrees, defaults to 0
        :type rotation: float
        :raises ValueError: Icon file not found
        :return: animation object
        :rtype: VehicleAnimation

        Creates an object that can be passed to a ``Vehicle`` subclass to 
        depict the moving robot as a polygon during simulation.

        For example, the image of a red car is::

            a = VehicleIcon('redcar', scale=2)
            veh = Bicycle()
            veh.run(animation=a)

        ``filename`` can be:

            * ``"greycar"`` a grey and white car (top view)
            * ``"redcar"`` a red car (top view)
            * ``"piano"`` a piano (top view)
            * path to an image file, including extension

        .. image:: ../../rtb-data/rtbdata/data/greycar.png
          :width: 200px
          :align: center

        .. image:: ../../rtb-data/rtbdata/data/redcar.png
          :width: 300px
          :align: center

        .. image:: ../../rtb-data/rtbdata/data/piano.png
          :width: 300px
          :align: center


        The car is scaled to an image with a length of ``scale`` in the units of
        the plot.

        The vehicle rotates about its ``origin`` which is expressed in terms of
        normalized coordinates in the range 0 to 1.  By default it is in the
        middle of the icon image, (0.2, 0.5) moves it toward the back of the
        vehicle, (0.8, 0.5) moves it toward the front of the vehicle.

        .. note:: 
        
            - The standard icons are kept in ``roboticstoolbox/data``
            - By default the image is assumed to contain a car parallel to
              the x-axis and facing right.  If the vehicle is facing upward
              set ``rotation`` to 90.

        :seealso: :func:`~Vehicle`
        """
        super().__init__()
        if '.' not in filename:
            try:
                # try the default folder first
                image = rtb_load_data(Path("data") / Path(filename + ".png"), plt.imread)
            except FileNotFoundError:
                raise ValueError(f"{filename} is not a provided icon")
        else:
            try:
                image = plt.imread(filename)
            except FileNotFoundError:
                raise ValueError(f"icon file {filename} not found")
        
        self._rotation = rotation
        self._image = image

        # figure size of bounding box the image will fill in data coordinates
        if origin is None:
            origin = [0.5, 0.5]
        self._origin = origin

        if image.shape[0] >= image.shape[1]:
            # width >= height
            self._width = scale
            self._height =scale * image.shape[1] / image.shape[0]
        else:
            # width < height
            self._height = scale
            self._width = scale * image.shape[0] / image.shape[1]


    def _add(self, ax=None, **kwargs):

        def imshow_affine(ax, z, *args, **kwargs):
            im = ax.imshow(z, *args, **kwargs)
            x1, x2, y1, y2 = im.get_extent()
            # im._image_skew_coordinate = (x2, y1)
            return im

        self._ax = plt.gca()
        extent = [  -self._origin[0] * self._height, 
                    (1 - self._origin[0]) * self._height, 
                    -self._origin[1] * self._width, 
                    (1 - self._origin[1]) * self._width
                ]
        self._ax = plt.gca()

        args = {}
        if 'color' in kwargs and self._image.ndim == 2:
            color = kwargs['color']
            del kwargs['color']
            rgb = colors.to_rgb(color)
            cmapdata = {'red': [(0.0, 0.0, 0.0), (1.0, rgb[0], 0.0)],
                        'green': [(0.0, 0.0, 0.0), (1.0, rgb[1], 0.0)],
                        'blue': [(0.0, 0.0, 0.0), (1.0, rgb[2], 0.0)]
                        }
            cmap = colors.LinearSegmentedColormap('linear', segmentdata=cmapdata, N=256)
            args = {'cmap': cmap}
        elif self._image.ndim == 2:
            args = {'cmap': 'gray'}
        if 'zorder' not in kwargs:
            args['zorder'] = 3
            
        self._object = imshow_affine(self._ax, self._image,
                            interpolation='none',
                            extent=extent, clip_on=True,
                            **{**kwargs, **args})

    def _update(self, x):

        # center_x = self._width // 2
        # center_y = self._height // 2
        center_x = 0
        center_y = 0

        T = (mtransforms.Affine2D()
                    .rotate_deg_around(center_x, center_y, np.degrees(x[2]) - self._rotation)
                    .translate(x[0], x[1])
                    + self._ax.transData)
        self._object.set_transform(T)

if __name__ == "__main__":
    from math import pi

    from roboticstoolbox import Bicycle, RandomPath

    V = np.diag(np.r_[0.02, 0.5 * pi / 180] ** 2)

    v = VehiclePolygon()
    # v = VehicleIcon('greycar2', scale=2, rotation=90)

    veh = Bicycle(covar=V, animation=v, control=RandomPath(10), verbose=False)
    print(veh)

    odo = veh.step(1, 0.3)
    print(odo)

    print(veh.x)

    print(veh.f([0, 0, 0], odo))

    def control(v, t, x):
        goal = (6,6)
        goal_heading = atan2(goal[1]-x[1], goal[0]-x[0])
        d_heading = base.angdiff(goal_heading, x[2])
        v.stopif(base.norm(x[0:2] - goal) < 0.1)

        return (1, d_heading)

    veh.control=RandomPath(10)
    p = veh.run(1000, plot=True)
    # plt.show()
    print(p)

    veh.plot_xyt_t()
    # veh.plot(p)

    # t, x = veh.path(5, u=control)
    # print(t)

    # fig, ax = plt.subplots()

    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)


    # v = VehicleAnimation.Polygon(shape='triangle', maxdim=0.1, color='r')
    # v = VehicleAnimation.Icon('car3.png', maxdim=2, centre=[0.3, 0.5])
    # v = VehicleAnimation.Icon('/Users/corkep/Dropbox/code/robotics-toolbox-python/roboticstoolbox/data/car1.png', maxdim=2, centre=[0.3, 0.5])
    # v = VehicleAnimation.icon('car3.png', maxdim=2, centre=[0.3, 0.5])
    # v = VehicleAnimation.marker()
    # v.start()
    # plt.grid(True)
    # # plt.axis('equal')

    # for theta in np.linspace(0, 2 * np.pi, 100):
    #     v.update([0, 0, theta])
    #     plt.pause(0.1)