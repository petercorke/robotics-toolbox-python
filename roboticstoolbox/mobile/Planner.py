"""
Python Navigation Abstract Class
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing

Not ready for use yet.
"""
from scipy import integrate
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *
from spatialmath import SE2, SE3
from matplotlib import cm
from abc import ABC, abstractmethod
import matplotlib as mpl
import copy
from roboticstoolbox.mobile.OccGrid import OccGrid
from colored import fg, attr

class Planner:

    def __init__(self, occgrid=None, goal=None, inflate=0,
                 reset=False, verbose=False, seed=None,
                 transform=SE2()):
        self._occgrid = None
        self._start = None
        self._verbose = verbose
        self._seed = seed
        self._spin_count = None
        self._private_random = np.random.default_rng(seed=seed)
        self._seed_0 = None
        self._inflate = inflate
        self._reset = reset
        self._transform = transform
        self._goal = None

        if occgrid is not None:
            if not isinstance(occgrid, OccGrid):
                occgrid = OccGrid(occgrid)
            self._occgrid0 = occgrid  # original occgrid for reference

            if inflate > 0:
                self._occgrid = occgrid.copy()
                self._occgrid.inflate(inflate)
            else:
                self._occgrid = occgrid

        if goal is not None:
            self._goal = np.transpose(goal)


        self._spin_count = 0

    def __str__(self):
        s = f"Navigation class: {self.__class__.__name__}"
        if self._occgrid0 is not None:
            s += f"\n  Occupancy grid: {self._occgrid0.shape}"
        if self._goal is not None:
            s += f"\n  Goal: {self.goal}"
        return s

    @property
    def occgrid(self):
        """
        Occupancy grid

        :return: occupancy grid used for planning
        :rtype: OccGrid instance

        This returns the grid that was optionally inflated at constructor time.
        """
        return self._occgrid

    @property
    def goal(self):
        """
        Goal point or configuration for planning

        :return: goal point or configuration
        :rtype: ndarray(2) or ndarray(3)
        """
        return self._goal

    @goal.setter
    def goal(self, goal):
        """
        Set goal point or configuration for planning

        :param goal: Set goal point or configuration
        :type goal: array_like(2) or array_like(3)
        :raises ValueError: if goal point is occupied
        """
        if self.isoccupied(goal):
            raise ValueError("Goal location inside obstacle")
        self._goal = base.getvector(goal)

    @property
    def start(self):
        """
        Start point or configuration for planning

        :return: start point or configuration
        :rtype: ndarray(2) or ndarray(3)
        """
        return self._start

    @start.setter
    def start(self, start):
        """
        Set start point or configuration for planning

        :param start: Set start point or configuration
        :type start: array_like(2) or array_like(3)
        :raises ValueError: if start point is occupied
        """
        if self.isoccupied(start):
            raise ValueError("Start location inside obstacle")
        self._start = base.getvector(start)

    def select_goal(self):
        self.plot()
        print("Select goal location")
        goal = round(plt.ginput(1))

        self.goal = goal

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        self._verbose = v

    @property
    def seed(self):
        return self._seed

    @property
    def spin_count(self):
        return self._spin_count

    @property
    def random(self):
        return self._private_random

    @property
    def seed_0(self):
        return self._seed_0

    @property
    def reset(self):
        return self._reset

    # Define abstract classes to be implemented later
    @abstractmethod
    def plan(self):
        pass

    @abstractmethod
    def next(self):
        pass

    def query(self, start, goal=None, animate=False, movie=None):
        """
        ssss

        :param start: [description]
        :type start: [type]
        :param goal: [description], defaults to None
        :type goal: [type], optional
        :param animate: [description], defaults to False
        :type animate: bool, optional
        :return: [description]
        :rtype: [type]

                    %Navigation.query Find a path from start to goal using plan
            %
            % N.query(START, OPTIONS) animates the robot moving from START (2x1) to the goal (which is a 
            % property of the object) using a previously computed plan.
            %
            % X = N.query(START, OPTIONS) returns the path (Mx2) from START to the goal (which is a property of 
            % the object).
            %
            % The method performs the following steps:
            %  - Initialize navigation, invoke method N.navigate_init()
            %  - Visualize the environment, invoke method N.plot()
            %  - Iterate on the next() method of the subclass until the goal is
            %    achieved.
            %
            % Options::
            % 'animate'    Show the computed path as a series of green dots.
            %
            % Notes::
            %  - If START given as [] then the user is prompted to click a point on the map.
            %
            %
            % See also Navigation.navigate_init, Navigation.plot, Navigation.goal.
        """
        # make sure start and goal are set and valid
        self.check_points(start, goal)

        if movie is not None:
            animate = True

        if animate:
            self.plot()

        movie = MovieWriter(movie)

        robot = self._start
        path = [robot]

        while True:
            if animate:
                plt.plot(robot[0], robot[1], 'y.', 12)
                plt.pause(0.05)

            # get next point on the path
            robot = self.next(robot)

            # are we are done?
            if robot is None:
                path.append(self._goal)
                return np.array(path).astype(int)

            path.append(robot)

    def plot(self, path=None, 
            path_marker=None, path_marker_reverse=None,
            start_marker=None, goal_marker=None, 
            direction=None, background=True,
            ax=None, block=False, **kwargs):
        """
        Plot vehicle path

        :param path: path, defaults to None
        :type p: ndarray(N,2) or ndarray(N,3)
        :param path_marker: style for marking points on path
        :type path_marker: dict, optional
        :param path_marker: style for marking points on path when reversing
        :type path_marker: dict, optional
        :param start_marker: style for marking start point
        :type start_marker: dict, optional
        :param goal_marker: style for marking goal point
        :type goal_marker: dict, optional
        :param direction: direction of travel, defaults to None
        :type direction: ndarray(N) with positive value for forward, and
            negative value for reverse, optional
        :param background: plot occupancy grid if present, default True
        :type background: bool, optional
        :param ax: axes to plot into
        :type ax: matplotlib axes
        :param block: block after displaying the plot
        :type block: bool, optional

        The path has one row per time step and has 2 or 3 columns for motion in
        the :math:`x-y` plane or in :math:`x-y-\theta` configuration space
        respectively.  If 3 columns are given the plot will be a 3D plot.

        Markers are specified as dicts using matplotlib keywords, for example::

            planner.plot(path, path_marker=dict(marker='s', color='b'))

        Default values are provided for all markers, start point is a circle and
        goal point is a star.

        If the planner supports bi-directional motion then the ``direction``
        option gives the direction for every point on the path.
        """
        # create default markers
        if path_marker is None:
            path_marker = { 'marker': '.',
                            'markerfacecolor': 'b',
                            'markersize': 12,
                          }
            path_marker_reverse = { 'marker': '.',
                            'markerfacecolor': 'r',
                            'markersize': 10,
                          }
        if start_marker is None:
            start_marker = {'marker': 'o',
                            'markeredgecolor': 'w',
                            'markerfacecolor': 'y', 
                            'markersize': 10,
                           }
        if goal_marker is None:
            goal_marker = { 'marker': '*',
                            'markeredgecolor': 'w',
                            'markerfacecolor': 'y',
                            'markersize': 16,
                          }

        # plot occupancy grid background
        if background:
            self.plot_bg(**kwargs)
        
        if path is not None:
            if not isinstance(path, np.ndarray) or not (path.shape[1] in (2, 3)):
                raise ValueError('path must be an Nx2 or Nx3 matrix of points')
            ndims = path.shape[1]
        elif self.goal is not None:
            ndims = len(self.goal)

        ax = base.axes_logic(ax, ndims)
        # mark the path
        if path is not None:

            if ndims == 2:
                # 2D case
                if direction is not None:
                    direction = np.array(direction)
                    if direction.shape[0] != path.shape[0]:
                        raise ValueError('direction vector must have same length as path')
                    ax.plot(path[:, 0], path[:, 1], 'k')
                    ax.plot(path[direction > 0, 0], path[direction > 0, 1], color='none', **path_marker)
                    ax.plot(path[direction < 0, 0], path[direction < 0, 1], color='none', **path_marker_reverse)
                else:
                    ax.plot(path[:, 0], path[:, 1], **path_marker)
            elif path.shape[1] == 3:
                # 3D case
                if direction is not None:
                    direction = np.array(direction)
                    if direction.shape[0] != path.shape[0]:
                        raise ValueError('direction vector must have same length as path')
                    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'k')
                    ax.plot(path[direction > 0, 0], path[direction > 0, 1], path[direction > 0, 2], color='none', **path_marker)
                    ax.plot(path[direction < 0, 0], path[direction < 0, 1], path[direction < 0, 2], color='none', **path_marker_reverse)
                else:
                    ax.plot(path[:, 0], path[:, 1], path[:, 2], **path_marker)

        # mark start and goal if requested
        if ndims == 2:
            if self.start is not None:
                ax.plot(self.start[0], self.start[1], zorder=10, **start_marker)
            if self.goal is not None:
                ax.plot(self.goal[0], self.goal[1], zorder=10, **goal_marker)

        elif ndims == 3:
            if self.start is not None:
                ax.plot(self.start[0], self.start[1], self.start[2]+0.1, zorder=10, **start_marker)
            if self.goal is not None:
                plt.plot(self.goal[0], self.goal[1], self.goal[2]+0.1, zorder=10, **goal_marker)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if ndims == 2:
            ax.set_aspect('equal')
        else:
            ax.set_zlabel(r'$\theta$')

        plt.show(block=block)


    def plot_bg(self, distance=None, color_map='gray',
                inflated=True, path_marker=None, **unused_args):
        """
        [summary]

        :param p: [description], defaults to None
        :type p: [type], optional
        :param distance: [description], defaults to None
        :type distance: [type], optional
        :param color_map: [description], defaults to cm.get_cmap('bone')
        :type color_map: [type], optional
        :param beta: [description], defaults to 0.2
        :type beta: float, optional
        :param inflated: [description], defaults to False
        :type inflated: bool, optional

                %Navigation.plot  Visualization background
        %
        % N.plot_bg(OPTIONS) displays the occupancy grid with occupied cells shown as
        % red and an optional distance field.
        %
        % N.plot_bg(P,OPTIONS) as above but overlays the points along the path (2xM) matrix. 
        %
        % Options::
        %  'distance',D      Display a distance field D behind the obstacle map.  D is
        %                    a matrix of the same size as the occupancy grid.
        %  'colormap',@f     Specify a colormap for the distance field as a function handle, eg. @hsv
        %  'beta',B          Brighten the distance field by factor B.
        %  'inflated'        Show the inflated occupancy grid rather than original
        %  'pathmarker',M    Options to draw a path point
        %  'startmarker',M   Options to draw the start marker
        %  'goalmarker',M    Options to draw the goal marker
        %
        % Notes::
        % - The distance field at a point encodes its distance from the goal, small
        %   distance is dark, a large distance is bright.  Obstacles are encoded as
        %   red.
        % - Beta value -1<B<0 to darken, 0<B<+1 to lighten.
        %
        % See also Navigation.plot, Navigation.plot_fg, brighten.
        """
        if self._occgrid is None:
            return

        fig, ax = plt.subplots(nrows=1, ncols=1)

        if hasattr(self, 'distancemap'):
            # distance field with obstacles

            # find largest finite value
            v = distance.ravel()
            vmax = max(v[np.isfinite(v)])

            # create a copy of greyscale color map
            c_map = copy.copy(mpl.cm.get_cmap(color_map))
            c_map.set_bad(color=(1,0,0,1))  # nan and inf are red

            # change all inf to large value, so they are not 'bad' ie. red
            distance[np.isinf(distance)] = 2 * vmax
            c_map.set_over(color=(0,0,1))  # ex-infs are now blue

            # display image
            norm = mpl.colors.Normalize(vmin=0, vmax=vmax, clip=False)
            ax.imshow(distance, origin='lower',
                cmap=c_map,
                norm=norm,
                )
            ax.grid(True, alpha=0.1, color=(1,1,1))

            # add colorbar
            scalar_mappable_c_map = cm.ScalarMappable(cmap=c_map, norm=norm)
            plt.colorbar(scalar_mappable_c_map, label='Distance', shrink=0.7, aspect=20*0.7)

            ax.set_xlabel('X')
            ax.set_ylabel('y (cells)')

        else:
            # occupancy grid only

            # create color map for free space + obstacle:
            #   free space, color index = 1, white, alpha=0 to allow background and grid lines to show
            #   obstacle, color index = 2, red, alpha=1

            if self._inflate > 0 and inflated:
                # 0 background (white, transparent)
                # 1 inflated obstacle (pink)
                # 2 original obstacle (red)
                colors = [(1, 1, 1, 0), (1, 0.75, 0.8, 1), (1, 0, 0, 1)]
                image = self.occgrid.grid.astype(int) + self._occgrid0.grid.astype(int)
            else:
                # 0 background
                # 1 obstacle
                colors = [(1, 1, 1, 0), (1, 0, 0, 1)]
                image = self.occgrid.grid

            ax.set_facecolor((1, 1, 1)) # create white background
            c_map = mpl.colors.ListedColormap(colors)
            self.occgrid.plot(image, cmap=c_map, zorder=1)
            ax.grid(True, zorder=0)

        # lock axis limits to current value
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

        plt.draw()
        plt.show(block=False)


    def check_points(self, start=None, goal=None, dim=2):
        # if start is None:
        #     self.plot()
        #     disp("Select start location")
        #     start = round(plt.ginput(1))

        # if goal is None:
        #     self.plot()
        #     disp("Select goal location")
        #     goal = round(plt.ginput(1))
        # TODO

        if start is not None:
            self._start = base.getvector(start, dim, dtype=int)
            if self.occgrid.isoccupied(self._start[:2]):
                raise ValueError("Start location inside obstacle")

        if goal is not None:
            self._goal = base.getvector(goal, dim, dtype=int)
            if self.occgrid.isoccupied(self.goal[:2]):
                raise ValueError("Goal location inside obstacle")

    def isoccupied(self, p):
        """
        Test if coordinate is occupied

        :param p: world coordinate (x, y)
        :type p: array_like(2)
        :return: occupancy status of corresponding grid cell
        :rtype: bool

        The world coordinate is rounded to integer grid coordinates and the
        occupancy status of that cell is returned.

        """
        if self.occgrid is None:
            return False
        else:
            return self.occgrid.isoccupied(p)

    def goal_change(self):
        pass

    def navigate_init(self, start):
        pass

    def message(self, s):
        if self.verbose:
            print(fg('yellow'), "Planner:: " + s, attr(0))


    def spinner(self):
        spin_chars = "\|/"  # TODO: This might break?
        self._spin_count = self._spin_count + 1
        print(spin_chars[np.mod(self._spin_count, len(spin_chars))+1])

    @staticmethod
    def show_distance(d):
        d[np.isinf(d)] = None
        ax = plt.gca()
        c_map = plt.get_cmap("Greys")
        plt.clim(0, np.max(d[:]))
        plt.figimage(d)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    # There's no waitbar in matplotlib, so just going with placeholder functions
    @staticmethod
    def progress_init(title):
        h = "Waiting"
        return h

    @staticmethod
    def progress(h, x):
        pass

    @staticmethod
    def progress_delete(h):
        plt.clf()


class MovieWriter:

    def __init__(self, filename=None, interval=0.1, fig=None):
        """
        Save animation as a movie file

        :param filename: name of movie file, or tuple containing filename and
            frame interval
        :type filename: str or tuple(str, float)
        :param interval: frame interval, defaults to 0.1
        :type interval: float, optional
        :param fig: figure to record for the movie
        :type fig: figure reference 

        Example::

            movie = MovieWriter(filename)

            while ...
                movie.add()

            movie.done()

        To avoid extra user-logic, if ``MovieWriter`` is called with ``filename`` equal to None,
        then the writer will do nothing when the ``add`` and ``done`` methods are called.
        """
        # Set up formatting for the movie files
        if filename is None:
            self.writer = None
            return

        if isinstance(filename, (tuple, list)):
            filename, interval = filename

        if os.path.exists(filename):
            print("overwriting movie", filename)
        else:
            print("creating movie", filename)
        self.writer = animation.FFMpegWriter(
            fps=round(1 / interval), extra_args=["-vcodec", "libx264"]
        )
        if fig is None:
            fig = plt.gcf()
        self.writer.setup(fig, filename)
        self.filename = filename

    def add(self):
        """
        Add frame to the movie
        """
        if self.writer is not None:
            self.writer.grab_frame()

    def done(self):
        if self.writer is not None:
            self.writer.finish()
            self.writer = None

# # Sourced from: https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python/28995315#28995315
# def sub2ind(array_shape, rows, cols):
#     ind = rows*array_shape[1] + cols
#     ind[ind < 0] = -1
#     ind[ind >= array_shape[0]*array_shape[1]] = -1
#     return ind

# class Error(Exception):
#     """Base class for other exceptions"""
#     pass

