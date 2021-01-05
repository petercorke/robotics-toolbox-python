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
from spatialmath.pose2d import SE2
from scipy.ndimage import binary_dilation
from matplotlib import cm
from abc import ABCMeta, abstractmethod
import matplotlib as mpl
import copy

class Navigation:
    # __metaclass__ = ABCMeta

    def __init__(self, occ_grid=None, goal=None, inflate=0,
                 reset=False, verbose=None, seed=None,
                 transform=SE2()):
        self._occ_grid_nav = None
        self._start = None
        self._verbose = verbose
        self._seed = seed
        self._spin_count = None
        self._rand_stream = None
        self._seed_0 = None
        self._w2g = None
        self._inflate = inflate
        self._reset = reset
        self._transform = transform
        self._goal = None

        # if occ_grid is not None:
        #     # this is the inverse of the matlab code
        #     if type(occ_grid) is dict:
        #         self._occ_grid = occ_grid["map"]
        #         # self._w2g = self._T  # TODO
        #     else:
        #         self._occ_grid = occ_grid
        #         # self._w2g = SE2(0, 0, 0)
        self._occ_grid = occ_grid

        if inflate > 0:
            # Generate a circular structuring element
            r = inflate
            y, x = np.ogrid[-r: r+1, -r: r+1]
            SE = np.square(x) + np.square(y) <= np.square(r)
            SE = SE.astype(int)

            # do the inflation using SciPy
            self._occ_grid_nav = binary_dilation(self._occ_grid, SE)
        else:
            self._occ_grid_nav = self._occ_grid

        if goal is not None:
            self._goal = np.transpose(goal)

        # Simplification of matlab code
        self._privaterandom = np.random.default_rng(seed=seed)

        rs = np.random.RandomState()
        if seed is not None:
            rs = np.random.RandomState(seed)

        self._seed_0 = rs.get_state()
        self._rand_stream = rs
        self._w2g = transform
        self._spin_count = 0

    def __str__(self):
        s = f"Navigation class: {self.__class__.__name__}"
        if self._occ_grid is not None:
            s += f"\n  Occupancy grid: {self._occ_grid.shape}"
        if self._goal is not None:
            s += f"\n  Goal: {self.goal}"
        return s

    @property
    def occ_grid(self):
        return self._occ_grid

    @property
    def occ_grid_nav(self):
        return self._occ_grid_nav

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, goal):
        if self.is_occupied(goal):
            raise ValueError("Goal location inside obstacle")
        self._goal = base.getvector(goal, 2)

    def select_goal(self):
        self.plot()
        print("Select goal location")
        goal = round(plt.ginput(1))

        self.goal = goal

    @property
    def start(self):
        return self._start

    @property
    def verbose(self):
        return self._verbose

    @property
    def seed(self):
        return self._seed

    @property
    def spin_count(self):
        return self._spin_count

    @property
    def rand_stream(self):
        return self._rand_stream

    @property
    def seed_0(self):
        return self._seed_0

    @property
    def w2g(self):
        return self._w2g

    @property
    def inflate(self):
        return self._inflate

    @property
    def private(self):
        return self._private

    @property
    def reset(self):
        return self._reset

    @property
    def transform(self):
        return self._transform

    # Define abstract classes to be implemented later
    @abstractmethod
    def plan(self):
        pass

    @abstractmethod
    def next(self):
        pass

    def query(self, start, goal=None, animate=False):
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
        if goal is not None:
            self.check_query(start, goal)
        else:
            self.check_query(start, self.goal)

        if animate:
            self.plot()

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
                return np.array(path)

            path.append(robot)

    def plot(self, block=False, **kwargs):
        self.plot_bg(**kwargs)
        self.plot_fg(**kwargs)
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

        fig, ax = plt.subplots(nrows=1, ncols=1)

        if distance is None:
            # occupancy grid only

            # create color map for free space + obstacle:
            #   free space, color index = 1, white, alpha=0 to allow background and grid lines to show
            #   obstacle, color index = 2, red, alpha=1

            if self._inflate and inflated:
                # 0 background (white, transparent)
                # 1 inflated obstacle (pink)
                # 2 original obstacle (red)
                colors = [(1, 1, 1, 0), (1, 0.75, 0.8, 1), (1, 0, 0, 1)]
                image = (self._occ_grid + self._occ_grid_nav)
            else:
                # 0 background
                # 1 obstacle
                colors = [(1, 1, 1, 0), (1, 0, 0, 1)]
                image = self._occ_grid

            ax.set_facecolor((1, 1, 1)) # create white background
            c_map = mpl.colors.ListedColormap(colors)
            ax.imshow(image, cmap=c_map, origin='lower', zorder=1)

            ax.grid(True, zorder=0)

        else:
            # distance field with obstacles

            # find largest finite value
            v = distance.flatten()
            vmax = max(v[np.isfinite(v)])

            # create a copy of greyscale color map
            c_map = copy.copy(mpl.cm.get_cmap(color_map))
            c_map.set_bad(color=(1,0,0,1))  # nan and inf are red

            # change all inf to large value, so they are not 'bad' ie. red
            distance[np.isinf(distance)] = 2 * vmax
            c_map.set_over(color=(0,0,1))  # ex-infs are now blue

            # display image
            ax.imshow(distance, origin='lower',
                cmap=c_map,
                norm=mpl.colors.Normalize(vmin=0, vmax=vmax, clip=False)
                )
            ax.grid(True, alpha=0.1, color=(1,1,1))

            # add colorbar
            scalar_mappable_c_map = cm.ScalarMappable(cmap=c_map)
            plt.colorbar(scalar_mappable_c_map)

        ax.set_xlabel('x (cells)')
        ax.set_ylabel('y (cells)')

        # lock axis limits to current value
        ax.set_xlim(ax.get_xlim())
        ax.set_ylim(ax.get_ylim())

        plt.draw()
        plt.show(block=False)


    def plot_fg(self, path=None, path_marker=None, start_marker=None, goal_marker=None, **unused_args):
        """
        [summary]

        :param p: [description], defaults to None
        :type p: [type], optional
        :param path_marker: [description], defaults to None
        :type path_marker: [type], optional
        :param start_marker: [description], defaults to None
        :type start_marker: [type], optional
        :param goal_marker: [description], defaults to None
        :type goal_marker: [type], optional
        :param goal: [description], defaults to None
        :type goal: [type], optional

                %Navigation.plot_fg  Visualization foreground
        %
        % N.plot_fg(OPTIONS) displays the start and goal locations if specified.
        % By default the goal is a pentagram and start is a circle.
        %
        % N.plot_fg(P, OPTIONS) as above but overlays the points along the path (2xM) matrix.
        % 
        % Options::
        %  'pathmarker',M    Options to draw a path point
        %  'startmarker',M   Options to draw the start marker
        %  'goalmarker',M    Options to draw the goal marker
        %
        % Notes::
        % - In all cases M is a single string eg. 'r*' or a cell array of MATLAB LineSpec options.
        % - Typically used after a call to plot_bg().
        %
        % See also Navigation.plot_bg.
        """

        # create default markers
        if path_marker is None:
            path_marker = { 'marker': '.',
                            'markerfacecolor': 'y',
                            'markersize': 12,
                          }
        if start_marker is None:
            start_marker = {'marker': 'o',
                            'markeredgecolor': 'w',
                            'markerfacecolor': 'b', 
                            'markersize': 10,
                           }
        if goal_marker is None:
            goal_marker = { 'marker': '*',
                            'markeredgecolor': 'w',
                            'markerfacecolor': 'b',
                            'markersize': 16,
                          }

        # mark the path
        if path is not None:
            if not isinstance(path, np.ndarray) or not (path.shape[1] in (2,3)):
                raise ValueError('path must be an Nx2 or Nx3 matrix of points')
            if path.shape[1] == 2:
                plt.plot(path[:, 0], path[:, 1], **path_marker)
            elif path.shape[1] == 3:
                plt.plot3(path[:, 0], path[:, 1], path[:, 2], **path_marker)

        # mark start and goal if requested
        if self._goal is not None:
            if len(self._goal) == 2:
                plt.plot(self._goal[0], self._goal[1], **goal_marker)
            else:
                plt.plot3(self._goal[0], self._goal[1], self._goal[2]+0.1, **goal_marker)
        if self._start is not None:
            if len(self._start) == 2:
                plt.plot(self._start[0], self._start[1], **start_marker)
            else:
                plt.plot3(self._start[0], self._start[1], self._goal[2]+0.1, **start_marker)

        plt.draw()


    def check_query(self, start=None, goal=None):
        # if start is None:
        #     self.plot()
        #     disp("Select start location")
        #     start = round(plt.ginput(1))

        # if goal is None:
        #     self.plot()
        #     disp("Select goal location")
        #     goal = round(plt.ginput(1))
        # TODO

        self._start = base.getvector(start, 2, dtype=np.int)
        self._goal = base.getvector(goal, 2, dtype=np.int)

        if self.is_occupied(self._start):
               raise ValueError("Start location inside obstacle")
        if self.is_occupied(self.goal):
            raise ValueError("Goal location inside obstacle")

    def is_occupied(self, p):
        """
        [summary]

        :param p: [description]
        :type p: [type]
        :return: [description]
        :rtype: [type]

                    %Navigation.isoccupied Test if grid cell is occupied
            %
            % N.isoccupied(POS) is true if there is a valid grid map and the
            % coordinates given by the columns of POS (2xN) are occupied.
            %
            % N.isoccupied(X,Y) as above but the coordinates given separately.
            %
            % Notes:
            % -  X and Y are Cartesian rather than MATLAB row-column coordinates.
        """
        occ = None
        pis = None
        if self._occ_grid_nav is None:
            return False

        p = base.getvector(p, 2, 'list')
        try:
            return self._occ_grid_nav[p[1], p[0]] > 0
        except IndexError:
            return True  # points outside the map are considered occupied

        # if x is not None:
        #     if np.size(x) == 2:
        #         x = np.transpose(x)
        #     assert(np.shape(x, 0) == 2, "RTB:Navigation:isoccupied. P must have 2 rows")
        #     pos = x
        # else:
        #     assert(np.size(x) == np.size(y), "RTB:Navigation:isoccupied. X and Y must be same length")
        #     pos = np.array([(np.transpose(x)), np.transpose(y)])

        # pos = round(self._w2g * pos)
        # k = pos[0, :] > 0 & pos[0, :] <= np.shape(self._occ_grid) & pos[1,:] > 0 <= np.shape(self._occ_grid, 1)

        # i = sub2ind(np.shape(self._occ_grid), pos[1, k], pos[0, k])
        # occ = np.ones(1, np.size(pos, 1))  # TODO: this bit normally says 'logic' in matlab... should be fine
        # occ[k] = self._occ_grid_nav[i] > 0
        # return occ

    def goal_change(self):
        pass

    def navigate_init(self, start):
        pass

    def rand(self, **kwargs):
        return self._privaterandom.uniform(**kwargs)

    def randn(self, **kwargs):
        return self._privaterandom.normal(**kwargs)

    def randi(self, *pos, **kwargs):
        return self._privaterandom.integers(*pos, **kwargs)

    def verbosity(self, v):
        self._verbose = v

    def message(self, s, args=None):
        if self._verbose:
            if args is None:
                print("NavigationDebug:: " + s)
            else:
                print("NavigationDebug:: " + s.format(args))

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



# Sourced from: https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python/28995315#28995315
def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind

class Error(Exception):
    """Base class for other exceptions"""
    pass

