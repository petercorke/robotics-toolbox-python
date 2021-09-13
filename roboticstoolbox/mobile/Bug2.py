"""
Python Bug Planner
@Author: Kristian Gibson
@Author: Peter Corke
"""
from numpy import disp
from scipy import integrate

from spatialmath.pose2d import SE2
from spatialmath import base
from spatialmath.base.animate import *
from scipy.ndimage import *
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import animation
from roboticstoolbox.mobile.PlannerBase import PlannerBase

class Bug2(PlannerBase):
    """
    Construct a Bug2 reactive navigator

    :param occgrid: occupancy grid
    :type occgrid: :class:`OccGrid` instance or ndarray(N,M)
    :param kwargs: common arguments for :class:`Planner` superclass
    :return: Planner subclass implementing Bug2 algorithm
    :rtype: Bug2Planner instance

    Creates a planner object which be used to return a path from start
    to goal.

    .. note:: This is not strictly a planner since it is entirely reactive.
            Therefore, paths can be very inefficient.

    :reference: Path-Planning Strategies for a Point Mobile Automaton Moving
        Amidst Unknown Obstacles of Arbitrary Shape, Lumelsky and Stepanov,
        Algorithmica (1987)2, pp.403-430

    :author: Kristian Gibson and Peter Corke
    :seealso: :class:`Planner`
    """
    def __init__(self, **kwargs):

        super().__init__(ndims=2, **kwargs)

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
    def step(self):
        return self._step

    @property
    def m_line(self):
        return self._m_line

    # override query method of base class
    def run(self, start=None, goal=None, animate=False, pause=0.001, trail=True, movie=None, **kwargs):
        """
        Find a path using Bug2 reactive navigation algorithm
        
        :param start: starting position
        :type start: array_like(2)
        :param goal: goal position
        :type goal: array_like(2)
        :param animate: show animation of robot moving over the map, 
            defaults to False
        :type animate: bool, optional
        :param movie: save animation as a movie, defaults to None. Is either
            name of movie or a tuple (filename, frame interval)
        :type movie: str or tuple(str, float), optional
        :param trail: show the path followed by the robot, defaults to True
        :type current: bool, optional
        :return: path from ``start`` to ``goal``
        :rtype: ndarray(2,N)

        Compute the path from ``start`` to `goal` assuming the robot is capable
        of 8-way motion from its current cell to the next.
        
        .. note:: ``start`` and `goal` are given as (x,y) coordinates in the
              occupancy grid map, not as matrix row and column coordinates.

        :seealso: :meth:`Bug2.plot`
        """

        # make sure start and goal are set and valid
        # super().query(start=start, goal=goal, dtype=np.int, **kwargs)
        # make sure start and goal are set and valid
        self.start = self.validate_endpoint(start, dtype=int)
        self.goal = self.validate_endpoint(goal, dtype=int)

        # compute the m-line
        #  create homogeneous representation of the line
        #  line*[x y 1]' = 0
        self._m_line = hom_line(self.start, self.goal)

        if movie is not None:
            animate = True

        if animate:
            self.plot()
            self.plot_m_line()
            plt.pause(0.05)

        # movie = MovieWriter(movie)

        robot = self.start
        self._step = 1
        path = robot
        h = None

        trail_line, = plt.plot(0, 0, 'y.', label='robot path')
        trail_head, = plt.plot(0, 0, 'ko', zorder=10)

        # iterate using the next() method until we reach the goal
        while True:
            if animate:
                trail_head.set_data(robot[0], robot[1])
                if trail:
                    trail_line.set_data(path.T)

                if pause > 0:
                    plt.pause(pause)
                # plt.draw()
                # plt.show(block=False)
                # plt.gcf().canvas.flush_events()

                # movie.add()

            # move to next point on path
            robot = self.next(robot)

            # # have we been here before, ie. in a loop
            # if any([all(robot == x) for x in path]):
            #     raise RuntimeError('trapped')

            # are we there yet?
            if robot is None:
                break
            else:
                path = np.vstack((path, robot))

        # movie.done()

        return path

    def plot_m_line(self, ls=None):
        if ls is None:
            ls = 'k--'

        x_min, x_max = plt.gca().get_xlim()
        y_min, y_max = plt.gca().get_ylim()
        if self._m_line[1] == 0:
            # handle the case that the line is vertical
            plt.plot([self._start[0], self._start[0]],
                     [y_min, y_max], 'k--', label='m-line')
        else:
            # regular line
            x = np.array([
                [x_min, 1],
                [x_max, 1]  ])
            y = -x @ np.r_[self._m_line[0], self._m_line[2]]
            y = y / self._m_line[1]
            plt.plot([x_min, x_max], y, ls, zorder=10, label='m-line')

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
            if self.isoccupied(position + np.r_[dx, dy]):
                self.message(f"  {position}: obstacle at {position + np.r_[dx, dy]}")
                self.h.append(position)
                self._step = 2  # transition to step 2
                self.message(f"  {position}: change to step 2")

                # get a list of all the points around the obstacle
                self._edge, _ = edgelist(self.occgrid.grid == 0, position)
                self._k = 0
            else:
                n = position + np.array([dx, dy])

        if self._step == 2:
            # Step 2.  Move around the obstacle until we reach a point
            # on the M-line closer than when we started.

            self.message(f"{position}: moving around the obstacle (step 2)")
            if self._k < len(self._edge):
                n = self._edge[self._k]  # next edge point
            else:
                # we are at the end of the list of edge points, we
                # are back where we started.  Step 2.c test.
                plt.show(block=True)
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

    def query(self):
        raise NotImplementedError('This class has no query method')

    def plan(self):
        raise NotImplementedError('This class has no plan method')

# Ported from Peter Corke's edgelist function found:
# https://github.com/petercorke/toolbox-common-matlab/blob/master/edgelist.m

#  these are directions of 8-neighbours in a clockwise direction
# fmt: off
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
# fmt: on
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
    p = base.getvector(p, 2, dtype=np.int)
    
    if direction > 0:
        neighbours = np.arange(start=0, stop=8, step=1)
    else:
        neighbours = np.arange(start=7, stop=-1, step=-1)

    try:
        pix0 = im[p[1], p[0]]  # color of pixel we start at
    except:
        raise ValueError('specified coordinate is not within image')


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
def hom_line(p1, p2):
    line = np.cross(np.r_[p1[0], p1[1], 1], np.r_[p2[0], p2[1], 1])

    # normalize so that the result of x*l' is the pixel distance
    # from the line
    return line / np.linalg.norm(line[0:2])


if __name__ == "__main__":

    from roboticstoolbox import loadmat

    vars = loadmat("data/map1.mat")
    map = vars['map']

    bug = Bug2Planner(occgrid=map)
    # bug.plan()
    path = bug.query([20, 10], [50, 35], animate=True)
