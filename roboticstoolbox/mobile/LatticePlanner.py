from pgraph import DGraph, DVertex, Edge
import numpy as np
from spatialmath import SE2
import matplotlib.pyplot as plt
import itertools
from roboticstoolbox.mobile.PlannerBase import PlannerBase
from roboticstoolbox.mobile.OccGrid import BinaryOccupancyGrid
from collections import namedtuple

def make_arc(dir, radius=1, npoints=20):
    points = []

    if dir == 'S':
        points.append((0, 0))
        points.append((radius, 0))

    elif dir == 'L':
        for theta in np.linspace(0, np.pi/2, npoints):
            x = radius * np.sin(theta)
            y = radius * (1 - np.cos(theta))
            points.append((x, y))

    elif dir == 'R':
        for theta in np.linspace(0, np.pi/2, npoints):
            x = radius * np.sin(theta)
            y = radius * (-1 + np.cos(theta))
            points.append((x, y))
    
    return np.array(points).T

arcs = {}

class LatticeVertex(DVertex):
    def __init__(self, move=None, pose=None, name=None):
        super().__init__(name=name)
        self.move = move
        self.pose = pose
        self.coord = pose.xyt()

    def icoord(self):
        xyt = self.coord
        ix = int(round(xyt[0]))
        iy = int(round(xyt[1]))
        it = int(round(xyt[2]*2/np.pi))
        return f"({ix:d},{iy:d},{it:d}), {self.name}"

class LatticeEdge(Edge):
    def __init__(self, v1, v2, cost, pose, move):
        super().__init__(v1, v2, cost)
        self.pose = pose
        self.move = move
        self.arc = arcs[move]

    def plot(self, configspace=False, unwrap=False, **kwargs):
        T = self.pose
        xy = self.pose * self.arc
        if configspace:
            # 3D plot
            theta0 = self.pose.theta()
            if self.move == 'L':
                thetaf = theta0 + np.pi / 2
            elif self.move == 'R':
                thetaf = theta0 - np.pi / 2
            elif self.move == 'S':
                thetaf = theta0
            theta = np.linspace(theta0, thetaf, self.arc.shape[1])
            if unwrap:
                theta = np.unwrap(theta)
            plt.plot(xy[0, :], xy[1, :], theta, **kwargs)
        else:
            # 2D plot
            plt.plot(xy[0, :], xy[1, :], **kwargs)


class LatticePlanner(PlannerBase):
    """
    Lattice planner

    :param costs: cost for straight, left-turn, right-turn, defaults to :math:`(1, \pi/2, \pi/2)`
    :type costs: array_like(3), optional
    :param root: configuration of root node, defaults to (0,0,0)
    :type root: array_like(3), optional
    :param kwargs: arguments passed to ``Planner`` constructor

    ==================   ========================
    Feature              Capability
    ==================   ========================
    Plan                 Configuration space
    Obstacle avoidance   Yes
    Curvature            Discontinuous
    Motion               Forwards only
    ==================   ========================

    The lattice planner incrementally builds a graph from the root vertex, at
    each iteration adding three edges to the graph:

    - straight ahead 'S'
    - turn left 'L'
    - turn right 'R'

    If the configuration is already in the graph, the edge connects to that
    existing vertex.

    If an occupancy grid exists and the configuration is an obstacle, then
    the vertex is not added.

    ``costs`` changes the weighting for path costs at query time.
    """

    def __init__(self, costs=None, root=(0,0,0), **kwargs):

        global arcs

        super().__init__(ndims=3, **kwargs)

        self.poses = [SE2(1, 0, 0), SE2(1, 1, np.pi/2), SE2(1, -1, -np.pi/2)]
        self.moves = ['S', 'L', 'R']
        if costs is None:
            costs = [1, np.pi/2, np.pi/2]
        self.costs = costs
        self.root = root


        # create the set of possible moves
        for move in self.moves:
            arcs[move] = make_arc(move)

    def __str__(self):
        s = super().__str__() + f"\n  curvature={self.curvature}, stepsize={self.stepsize}"

    def icoord(self, xyt):
        ix = int(round(xyt[0]))
        iy = int(round(xyt[1]))
        it = int(round(xyt[2]*2/np.pi))
        return f"({ix:d},{iy:d},{it:d})"

    def plan(self, iterations=None, verbose=False):
        """
        Create a lattice plan

        :param iterations: number of iterations, defaults to None
        :type iterations: int, optional
        :param verbose: show frontier and added vertices/edges at each iteration, defaults to False
        :type verbose: bool, optional

        If an occupancy grid exists the if ``iterations`` is None the area of the
        grid will be completely filled.
        """
        if iterations is None and self.occgrid is None:
            raise ValueError('iterations must be finite if no occupancy grid is specified')

        self.graph = DGraph(metric='SE2')

        # add root vertex to the graph, place it in the frontier
        v0 = LatticeVertex(pose=SE2(self.root))
        self.graph.add_vertex(v0, name='0')
        frontier = [v0]

        iteration = 0
        while True:

            newfrontier = []
            for vertex in frontier:
                if verbose:
                    print('EXPAND:', vertex.icoord())

                for pose, move, cost in zip(self.poses, self.moves, self.costs):
                    newpose = vertex.pose * pose
                    xyt = newpose.xyt()
                    # theta is guaranteed to be in range [-pi, pi)

                    if verbose:
                        print('  MOVE', move, self.icoord(xyt))

                    if self.isoccupied(xyt[:2]):
                        if verbose:
                            print('    is occupied')
                        continue
                    vclose, d = self.graph.closest(xyt)

                    if d > 0.01:
                        # vertex does not already exists
                        vnew = LatticeVertex(move, newpose, name=vertex.name + move)

                        # ix = int(xyt[0])
                        # iy = int(xyt[1])
                        # it = int(round(xyt[2]*2/np.pi))
                        # vnew = LatticeVertex(move, newpose, name=f"{ix:d},{iy:d},{it:d}")
                        self.graph.add_vertex(vnew)
                        if verbose:
                            print('    add to graph as', vnew.name)

                        edge = LatticeEdge(vertex, vnew, cost=cost, pose=vertex.pose, move=move)

                        # connect it into the graph, add to frontier
                        vertex.connect(vnew, edge=edge)
                        newfrontier.append(vnew)
                    else:
                        # vertex already exists
                        # print('exists', vertex, move, vclose)

                        # connect it into the graph, don't add to frontier
                        if verbose:
                            print('    already in graph, connect to', vclose.icoord())
                        edge = LatticeEdge(vertex, vclose, cost=cost, pose=vertex.pose, move=move)
                        vertex.connect(vclose, edge=edge)

            frontier = newfrontier

            iteration += 1
            if iterations is None:
                if len(frontier) == 0:
                    print(f"finished after {iteration} iterations")
                    break
                print(f"iteration {iteration}, frontier length {len(frontier)}")
            elif iteration >= iterations:
                break
            
    def query(self, qs, qg):
        """
        Find a path through the lattice

        :param qs: initial configuration
        :type qs: array_like(3)
        :param qg: goal configuration
        :type qg: array_like(3)
        :return: path and status
        :rtype: ndarray(N,3), namedtuple

        The returned status value has elements:

        +-------------+-----------------------------------------------------+
        | Element     |  Description                                        |
        +-------------+-----------------------------------------------------+
        | ``cost``    | path cost                                           |
        +-------------+-----------------------------------------------------+
        |``segments`` | a list containing the type of each path segment as  |
        |             | a single letter code: either "L", "R" or "S" for    |
        |             | left turn, right turn or straight line respectively.|
        +-------------+-----------------------------------------------------+
        |``edges ``   | successive edges of the graph ``LatticeEdge`` type  |
        +-------------+-----------------------------------------------------+

        :seealso: :meth:`Planner.query`
        """

        vs, ds = self.graph.closest(qs)
        if ds > 0.001:
            raise ValueError('start configuration is not in the lattice')
        vg, dg = self.graph.closest(qg)
        if dg > 0.001:
            raise ValueError('goal configuration is not in the lattice')

        path, cost, _ = self.graph.path_Astar(vs, vg, verbose=False)

        status = namedtuple('LatticeStatus', ['cost', 'segments', 'edges'])

        segments = []
        edges = []
        for p, n in zip(path[:-1], path[1:]):
            e = p.edgeto(n)
            edges.append(e)
            segments.append(e.move)

        return np.array([p.coord for p in path]), status(cost, segments, edges)

    def plot(self, path=None, **kwargs):
        super().plot(**kwargs)
        
        if kwargs.get('configspace', False):

            # 3D plot
            for k, vertex in enumerate(self.graph):
                # for every node
                if k == 0:
                    plt.plot(vertex.coord[0], vertex.coord[1], vertex.coord[2], 'k>', markersize=10)
                else:
                    plt.plot(vertex.coord[0], vertex.coord[1], vertex.coord[2], 'bo')

            for edge in self.graph.edges():
                edge.plot(color='k', **kwargs)

            if path is not None:
                for p, n in zip(path[:-1], path[1:]):
                    # turn coordinaets back into vertices
                    vp, _ = self.graph.closest(p)
                    vn, _ = self.graph.closest(n)
                    e = vp.edgeto(vn)

                    #e.plot(color='b', linewidth=4)
                    
                    e.plot(color='k', linewidth=4)
                    e.plot(color='yellow', linewidth=3, dashes=(4,4))
        else:
            # 2D plot
            for k, vertex in enumerate(self.graph):
                # for every node
                if k == 0:
                    plt.plot(vertex.coord[0], vertex.coord[1], 'k>', markersize=10)
                else:
                    plt.plot(vertex.coord[0], vertex.coord[1], 'bo')

            for edge in self.graph.edges():
                edge.plot(color='k')

            if path is not None:
                for p, n in zip(path[:-1], path[1:]):
                    # turn coordinaets back into vertices
                    vp, _ = self.graph.closest(p)
                    vn, _ = self.graph.closest(n)
                    e = vp.edgeto(vn)

                    #e.plot(color='b', linewidth=4)
                    
                    e.plot(color='k', linewidth=4)
                    e.plot(color='yellow', linewidth=3, dashes=(4,4))



if __name__ == "__main__":

    og = BinaryOccupancyGrid(workspace=[-5, 5, -5, 5], value=False)
    og.set([3, 3, -2, 3], True)

    lattice = LatticePlanner(occgrid=og)

    lattice.plan(iterations=10)
    print(lattice.graph)

    qs = (0, 0, np.pi/2)
    qg = (1, 0, np.pi/2)

    # print('qs')
    # vs, d = lattice.graph.closest(qs)
    # print(vs, d, vs.coord)
    # print(vs.neighbours())
    # print()

    # print('[-1, 1, -np.pi]')
    # vs, d = lattice.graph.closest([-1, 1, -np.pi])
    # print(vs, d, vs.coord)
    # print(vs.neighbours())
    # print()

    # print('0L')
    # vs = lattice.graph['0L']
    # print(vs, vs.coord)
    # print(vs.neighbours())
    # print()

    path, status = lattice.query(qs, qg)

    print(path)
    print(status)


    lattice.plot(path=path)

    plt.show(block=True)

    # ax = plt.gca()
    # ax.set_aspect('equal')
    # ax.grid(True)
    # ax.set_xlim(-4, 4)
    # ax.set_ylim(-4, 4)
