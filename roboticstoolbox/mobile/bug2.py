"""
Python Bug Planner
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing

Not ready for use yet.
"""
from numpy import disp
from scipy import integrate

from spatialmath.pose2d import SE2
from spatialmath import base
from spatialmath.base.animate import *
from scipy.ndimage import *
from matplotlib import cm
from roboticstoolbox.mobile.navigation import Navigation


class Bug2(Navigation):
    def __init__(self, occ_grid, **kwargs):

        """
        
                    %Bug2.Bug2 Construct a Bug2 navigation object 
            %
            % B = Bug2(MAP, OPTIONS) is a bug2 navigation object, and MAP is an occupancy grid,
            % a representation of a planar world as a matrix whose elements are 0 (free
            % space) or 1 (occupied).
            %
            % Options::
            % 'goal',G      Specify the goal point (1x2)
            % 'inflate',K   Inflate all obstacles by K cells.
            %
            % See also Navigation.Navigation.

                    %Navigation.Navigation Create a Navigation object
        %
        % N = Navigation(OCCGRID, OPTIONS) is a Navigation object that holds an
        % occupancy grid OCCGRID.  A number of options can be be passed.
        %
        % Options::
        % 'goal',G        Specify the goal point (2x1)
        % 'inflate',K     Inflate all obstacles by K cells.
        % 'private'       Use private random number stream.
        % 'reset'         Reset random number stream.
        % 'verbose'       Display debugging information
        % 'seed',S        Set the initial state of the random number stream.  S must
        %                 be a proper random number generator state such as saved in
        %                 the seed0 property of an earlier run.
        %
        % Notes::
        % - In the occupancy grid a value of zero means free space and non-zero means
        %   occupied (not driveable).
        % - Obstacle inflation is performed with a round structuring element (kcircle) 
        %   with radius given by the 'inflate' option.
        % - Inflation requires either MVTB or IPT installed.
        % - The 'private' option creates a private random number stream for the methods 
        %   rand, randn and randi.  If not given the global stream is used.
        %
        % See also randstream.

        """
        super().__init__(occ_grid=occ_grid, **kwargs)

        self._h = []
        self._j = 0
        self._step = 1
        self._m_line = None
        self._edge = None
        self._k = None


    @property
    def h(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self._h

    @property
    def j(self):
        return self._j

    @property
    def step(self):
        return self._step

    @property
    def m_line(self):
        return self._m_line

    @property
    def edge(self):
        return self._edge

    @property
    def k(self):
        return self._k

    def query(self, start=None, goal=None, animate=False, movie=None, current=False):
        """
        [summary]

        :param start: [description], defaults to None
        :type start: [type], optional
        :param goal: [description], defaults to None
        :type goal: [type], optional
        :param animate: [description], defaults to False
        :type animate: bool, optional
        :param movie: [description], defaults to None
        :type movie: [type], optional
        :param current: [description], defaults to False
        :type current: bool, optional
        :return: [description]
        :rtype: [type]

                    %Bug2.query  Find a path
            %
            % B.query(START, GOAL, OPTIONS) is the path (Nx2) from START (1x2) to GOAL
            % (1x2).  Row are the coordinates of successive points along the path.  If
            % either START or GOAL is [] the grid map is displayed and the user is
            % prompted to select a point by clicking on the plot.
            %
            % Options::
            %  'animate'   show a simulation of the robot moving along the path
            %  'movie',M   create a movie
            %  'current'   show the current position position as a black circle
            %
            % Notes::
            % - START and GOAL are given as X,Y coordinates in the grid map, not as
            %   MATLAB row and column coordinates.
            % - START and GOAL are tested to ensure they lie in free space.
            % - The Bug2 algorithm is completely reactive so there is no planning
            %   method.
            % - If the bug does a lot of back tracking it's hard to see the current
            %   position, use the 'current' option.
            % - For the movie option if M contains an extension a movie file with that
            %   extension is created.  Otherwise a folder will be created containing
            %   individual frames.
            %
            % See also Animate.
        """
        anim = None
        if movie is not None:
            anim = Animate(movie)
            animate = True

        # make sure start and goal are set and valid
        self.check_query(start, goal)

        # compute the m-line
        #  create homogeneous representation of the line
        #  line*[x y 1]' = 0
        self._m_line = hom_line(self._start[0], self._start[1],
                                   self._goal[0], self._goal[1])

        if animate:
            self.plot()
            self.plot_m_line()

        robot = self._start[:]
        self._step = 1
        path = [self._start]
        h = None

        # iterate using the next() method until we reach the goal
        while True:
            if animate:
                plt.plot(robot[0], robot[1], 'y.')
                plt.pause(0.1)
                if current:
                    h = self.plot(robot[0], robot[1])
                plt.draw()
                if movie is not None:
                    anim.plot(h)
                if current:
                    self.delete(h)

            # move to next point on path
            robot = self.next(robot)

            # have we been here before, ie. in a loop
            if any([all(robot == x) for x in path]):
                raise RuntimeError('trapped')

            # are we there yet?
            if robot is None:
                break
            else:
                path.append(robot)

        if movie is not None:
            anim.done()

        return np.r_[path]

    def plot_m_line(self, ls=None):
        if ls is None:
            ls = 'k--'

        x_min, x_max = plt.gca().get_xlim()
        y_min, y_max = plt.gca().get_ylim()
        if self._m_line[1] == 0:
            # handle the case that the line is vertical
            plt.plot([self._start[0], self._start[0]],
                     [y_min, y_max], 'k--')
        else:
            # regular line
            x = np.array([
                [x_min, 1],
                [x_max, 1]  ])
            y = -x @ np.r_[self._m_line[0], self._m_line[2]]
            y = y / self._m_line[1]
            plt.plot([x_min, x_max], y, ls, zorder=10)

    def next(self, position):

        l = None
        y = None

        if all(self._goal == position):
            return None  # we have arrived

        if self._step == 1:
            # Step 1.  Move along the M-line toward the goal
            self.message(f"{position}: moving along the M-line (step 1)")
            # motion on line toward goal
            d = self._goal - position
            if abs(d[0]) > abs(d[1]):
                # line slope less than 45 deg
                dx = 1 if d[0] >= 0 else -1  # np.sign(d[0])
                l = self._m_line
                y = -((position[0] + dx) * l[0] + l[2]) / l[1]
                dy = int(round(y - position[1]))
            else:
                # line slope greater than 45 deg
                dy = 1 if d[1] >= 0 else -1  # np.sign(d[1])
                l = self._m_line
                x = -((position[1] + dy) * l[1] + l[2]) / l[0]
                dx = int(round(x - position[0]))

            # detect if next step is an obstacle
            if self.is_occupied(position + np.r_[dx, dy]):
                self.message(f"  {position}: obstacle at {position + np.r_[dx, dy]}")
                self.h.append(position)
                self._step = 2  # transition to step 2
                self.message(f"  {position}: change to step 2")

                # get a list of all the points around the obstacle
                self._edge, _ = edgelist(self._occ_grid_nav == 0, position)
                self._k = 1
            else:
                n = position + np.array([dx, dy])

        if self._step == 2:
            # Step 2.  Move around the obstacle until we reach a point
            # on the M-line closer than when we started.

            self.message(f"{position}: moving around the obstacle (step 2)")
            if self._k <= len(self._edge):
                n = self._edge[self._k]  # next edge point
            else:
                # we are at the end of the list of edge points, we
                # are back where we started.  Step 2.c test.
                raise RuntimeError('robot is trapped')

            # are we on the M-line now ?
            if abs(np.inner(np.r_[position, 1], self._m_line)) <= 0.5:
                self.message(f"  {position}: crossed the M-line")

                # are we closer than when we encountered the obstacle?
                if base.norm(position - self._goal) < base.norm(self._h[-1] - self._goal):
                    # self._j += 1
                    self._step = 1  # transition to step 1
                    self.message(f"  {position}: change to step 1")
                    return n

            # no, keep going around
            self._k += 1

        return n
        
    def plan(self):
        Error('RTB:Bug2:badcall', 'This class has no plan method')

# Ported from Peter Corke's edgelist function found:
# https://github.com/petercorke/toolbox-common-matlab/blob/master/edgelist.m

#  these are directions of 8-neighbours in a clockwise direction
_dirs = np.array([
        [-1,  0],
        [-1,  1],
        [ 0,  1],
        [ 1,  1],
        [ 1,  0],
        [ 1, -1],
        [ 0, -1],
        [-1, -1],
    ])

def edgelist(im, p, direction=1):
    """
    Find edge of a region

    :param im: binary image
    :type im: ndarray(h,w,int)
    :param p: initial point
    :type p: array_like(2)
    :param direction: direction to traverse region, +1 clockwise [default], -1
        counter-clockwise
    :type direction: int, optional
    :raises ValueError: initial point is not on the edge
    :raises RuntimeError: not able to find path to the goal
    :return: edge list, direction vector list
    :rtype: tuple of lists

    ``edge, dirs = edgelist(im, seed)`` is the boundary/contour/edge of a region
    in the binary image ``im``.  ``seed=[X,Y]`` is the coordinate of a point on
    the edge of the region of interest, but belonging to the region.

    ``edge`` is a list of coordinates (2) of edge pixels of a region in theThe
    elements of the edgelist are NumPy ndarray(2).

    ``dirs`` is a list of integers representing the direction of the edge from
    the corresponding point in ``edge`` to the next point in ``edge``.  The
    integers in the range 0 to 7 represent directions: W SW S SE E NW N NW
    respectively.

    - Coordinates are given and returned assuming the matrix is an image, so the
      indices are always in the form (x,y) or (column,row).
    - ``im` is a binary image where 0 is assumed to be background, non-zero 
      is an object.
    - ``p`` must be a point on the edge of the region.
    - The initial point is always the first and last element of the returned edgelist.
    - 8-direction chain coding can give incorrect results when used with
      blobs founds using 4-way connectivty.

    :Reference:

    - METHODS TO ESTIMATE AREAS AND PERIMETERS OF BLOB-LIKE OBJECTS: A COMPARISON
      Luren Yang, Fritz Albregtsen, Tor Lgnnestad and Per Grgttum
      IAPR Workshop on Machine Vision Applications Dec. 13-15, 1994, Kawasaki

    """

    if direction > 0:
        neighbours = np.arange(start=0, stop=8, step=1)
    else:
        neighbours = np.arange(start=7, stop=-1, step=-1)

    try:
        pix0 = im[p[1], p[0]]  # color of pixel we start at
    except:
        raise ValueError('specified coordinate is not within image')

    p = base.getvector(p, 2, dtype=np.int)
    q = adjacent_point(im, p, pix0)

    # find an adjacent point outside the blob
    if q is None:
        raise ValueError('no neighbour outside the blob')

    d = None
    e = [p]  # initialize the edge list
    dir = [] # initialize the direction list
    p0 = None

    while True:

        # find which direction is Q
        dq = q - p
        for kq in range(0, 8):
            # get index of neighbour's direction in range [1,8]
            if np.all(dq == _dirs[kq]):
                break

        # now test for directions relative to Q
        for j in neighbours:
            # get index of neighbour's direction in range [1,8]
            k = (j + kq) % 8
            # if k > 7:
            #     k = k - 7
            

            # compute coordinate of the k'th neighbour
            nk = p + _dirs[k]

            try:
                if im[nk[1], nk[0]] == pix0:
                    # if this neighbour is in the blob it is the next edge pixel
                    p = nk
                    break
            except:
                raise ValueError("Something went wrong calculating edgelist")

            q = nk

        dir.append(k)
        # check if we are back where we started
        if p0 is None:
                p0 = p  # make a note of where we started
        else:
            if all(p == p0):
                break

        # keep going, add this point to the edgelist
        e.append(p)
    

    return e, dir

# Ported from Peter Corke's adjacent_point function found:
# https://github.com/petercorke/toolbox-common-matlab/blob/master/edgelist.m

def adjacent_point(im, seed, pix0):
    """Find adjacent point

    :param im: occupancy grid
    :type im: ndarray(m,n)
    :param seed: initial point
    :type seed: ndarray(2)
    :param pix0: value of occupancy grid at ``seed`` coordinate
    :type pix0: int
    :return: coordinate of a neighbour
    :rtype: ndarray(2) or None

    Is a neighbouring point of the coordinate ``seed`` that is not within the
    region containing the coordinate ``seed``, ie. it is a neighbour but
    outside.
    """
    p = None

    for d in _dirs:
        p = seed + d
        try:
            if im[p[1], p[0]] != pix0:
                return p
        except:
            pass 

    return None


# Implementation of Peter Corke's matlab homline function from:
# https://github.com/petercorke/spatialmath-matlab/blob/master/homline.m
def hom_line(x1, y1, x2, y2):
    line = np.cross(np.r_[x1, y1, 1], np.r_[x2, y2, 1])

    # normalize so that the result of x*l' is the pixel distance
    # from the line
    return line / np.linalg.norm(line[0:2])

# # Sourced from: https://stackoverflow.com/questions/28995146/matlab-ind2sub-equivalent-in-python/28995315#28995315
# def sub2ind(array_shape, rows, cols):
#     ind = rows*array_shape[1] + cols
#     ind[ind < 0] = -1
#     ind[ind >= array_shape[0]*array_shape[1]] = -1
#     return ind


# def ind2sub(array_shape, ind):
#     ind[ind < 0] = -1
#     ind[ind >= array_shape[0]*array_shape[1]] = -1
#     rows = (ind.astype('int') / array_shape[1])
#     cols = ind % array_shape[1]
#     return rows, cols

# def col_norm(x):
#     y = np.array([])
#     if x.ndim > 1:
#         x = np.column_stack(x)
#         for vector in x:
#             y = np.append(y, np.linalg.norm(vector))
#     else:
#         y = np.linalg.norm(x)
#     return y

if __name__ == "__main__":

    from roboticstoolbox import loadmat

    vars = loadmat("../data/map1.mat")
    map = vars['map']

    bug = Bug2(map)
    # bug.plan()
    path = bug.query([20, 10], [50, 35])