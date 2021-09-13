#PoseGraph Pose graph 

import roboticstoolbox as rtb
import pgraph
from spatialmath import base, SE2
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import zipfile
import time
import math
from pathlib import Path
from progress.bar import FillingCirclesBar
class PGVertex(pgraph.UVertex):

    nvertices = 0

    def __init__(self, type, **kwargs):
        super().__init__(**kwargs)
        self.type = type
        self.index = PGVertex.nvertices
        PGVertex.nvertices += 1
class PGEdge(pgraph.Edge):
    def __init__(self, v1, v2, mean, info):
        super().__init__(v1, v2)
        self.mean = mean
        self.info = info

    def linear_factors(self):
        #computes the Taylor expansion of the error function of the k_th edge
        #vmeans: vertices positions
        #eids:   edge ids
        #emeans: edge means
        #  k:	 edge number
        #  e:	 e_k(x)
        #  A:	 d e_k(x) / d(x_i)
        #  B:	 d e_k(x) / d(x_j)
        v =  self.endpoints
        v_i =  v[0].coord
        v_j =  v[1].coord
        z_ij =  self.mean

        # compute the homoeneous transforms of the previous solutions
        zt_ij = base.trot2(z_ij[2], t=z_ij[0:2])
        vt_i = base.trot2(v_i[2], t=v_i[0:2])
        vt_j = base.trot2(v_j[2], t=v_j[0:2])

        # compute the displacement between x_i and x_j
        f_ij = base.trinv2(vt_i) @ vt_j

        # this below is too long to explain, to understand it derive it by hand
        theta_i = v_i[2]
        ti = v_i[0:2]
        tj = v_j[0:2]
        dt_ij = tj - ti

        si = np.sin(theta_i)
        ci = np.cos(theta_i)

        # fmt: off
        A =  np.array([
            [-ci, -si, [-si,  ci] @ dt_ij],
            [ si, -ci, [-ci, -si] @ dt_ij],
            [ 0,   0,  -1 ],
                ])
        B  = np.array([
            [  ci, si, 0],
            [ -si, ci, 0],
            [  0,  0,  1 ],
                ])
        # fmt: on
        ztinv = base.trinv2(zt_ij)
        T = ztinv @ f_ij
        e = base.tr2xyt(T)
        ztinv[0:2,2] =  0
        A = ztinv @ A
        B = ztinv @ B

        return e, A, B 

class PoseGraph:
    
    # properties
    #     graph
        
    #     ngrid
    #     center
    #     cellsize

    def __init__(self, filename, laser=False, verbose=False):
        # parse the file data
        # we assume g2o format
        #    VERTEX* vertex_id X Y THETA
        #    EDGE* startvertex_id endvertex_id X Y THETA IXX IXY IYY IXT IYT ITT
        # vertex numbers start at 0
        
        self.laser = laser
        
        self.graph = pgraph.UGraph(verbose=verbose)

        path = Path(rtb.rtb_path_to_datafile(filename))

        if path.suffix == '.zip':
            zf = zipfile.ZipFile(path, 'r')
            opener = zf.open
            path = path.stem
        else:
            opener = open

        with opener(path, 'r') as f:

            toroformat = False
            nlaser = 0
            
            # indices into ROBOTLASER1 record for the 3x3 info matrix in column major
            # order
            # IXX IXY IXT IYY IYT ITT
            #   6   7   8   9  10  11
            #   0   1   2   3   4   5
            g2o =  [0,  1,  2,  1,  3,  4,   2,  4,  5]
            # IXX IXY IYY ITT IXT IYT
            #   6   7   8   9  10  11
            #   0   1   2   3   4   5
            toro = [0,  1,  4,  1,  2,  5,  4,  5,  3]
            
            # we keep an array self. = vindex(gi) to map g2o vertex index to PGraph vertex index
            
            self.vindex = {}
            firstlaser = True
            for line in f:
                # for zip file, we get data as bytes not str
                if isinstance(line, bytes):
                    line = line.decode()
                
                # is it a comment?
                if line.startswith('#'):
                    continue
                
                tokens = line.split()
                
                # g2o format records
                if tokens[0] == 'VERTEX_SE2':
                    coord = np.array([float(x) for x in tokens[2:5]])
                    id = tokens[1]

                    v = PGVertex('vertex', coord=coord, name=id)
                    self.graph.add_vertex(v)

                elif tokens[0] ==  'VERTEX_XY':
                    coord = np.array([float(x) for x in tokens[2:4]])
                    id = tokens[1]

                    v = PGVertex('landmark', coord=coord, name=id)
                    self.graph.add_vertex(v)
                        
                elif tokens[0] ==  'EDGE_SE2':                    
                    v1 = tokens[1]
                    v2 = tokens[2]

                    # create the edge data as a structure
                    #  X  Y  T
                    #  3  4  5
                    mean = np.array([float(x) for x in tokens[3:6]])
                    
                    # IXX IXY IXT IYY IYT ITT
                    #   6   7   8   9  10  11
                    d = np.array([float(x) for x in tokens[6:12]])
                    info = d[g2o].reshape((3,3))

                    # create the edge
                    v1 = self.graph[v1]
                    v2 = self.graph[v2]
                    e = PGEdge(v1, v2, mean, info)
                    v1.connect(v2, edge=e)

                ## TORO format records
                # https://openslam-org.github.io/toro.html
                elif tokens[0] == 'VERTEX2':
                    toroformat = True

                    coord = np.array([float(x) for x in tokens[2:5]])
                    id = tokens[1]

                    v = PGVertex('vertex', coord=coord, name=id)
                    self.graph.add_vertex(v)
                        
                elif tokens[0] == 'EDGE2':
                    toroformat = True
                    
                    v1 = tokens[1]
                    v2 = tokens[2]

                    # create the edge data as a structure
                    #  X  Y  T
                    #  3  4  5
                    mean = np.array([float(x) for x in tokens[3:6]])
                    
                    # IXX IXY IYY ITT IXT IYT
                    #   6   7   8   9  10  11
                    d = np.array([float(x) for x in tokens[6:12]])
                    info = d[toro].reshape((3,3))

                    # create the edge
                    v1 = self.graph[v1]
                    v2 = self.graph[v2]
                    e = PGEdge(v1, v2, mean, info)
                    v1.connect(v2, edge=e)
                        
                elif tokens[0] == 'ROBOTLASER1':
                    if not laser:
                        continue
                    
                    # laser records are associated with the immediately 
                    # preceding VERTEX record
                    #
                    # not quite sure what all the fields are
                    # 1 ?
                    # 2 min scan angle
                    # 3 scan range
                    # 4 angular increment
                    # 5 maximum range possible
                    # ?
                    # 8 N = number of beams
                    # 9 to 9+N laser range data
                    # ?
                    # 9+N+12 timestamp (*nix timestamp)
                    # 9+N+13 laser type (str)


                    if firstlaser:
                        nbeams = int(tokens[8])
                        lasermeta = tokens[2:6]
                        firstlaser = False

                    # add attributes to the most recent vertex created
                    v.theta = np.arange(0, nbeams) * float(tokens[4]) + float(tokens[2])
                    v.range = np.array([float(x) for x in tokens[9:nbeams+9]])
                    v.time = float(tokens[21+nbeams])
                    self.vindex[v.index] = v
                    nlaser+= 1
                else:
                    raise RuntimeError(f"Unexpected line  {line} in {filename}")

            if toroformat:
                filetype = "TORO/LAGO"
            else:
                filetype = "g2o"
            print(f"loaded {filetype} format file: {self.graph.n} vertices, {self.graph.ne} edges")
            
            if nlaser > 0:
                lasermeta = [float(x) for x in lasermeta]
                
                self._angmin = lasermeta[0]
                self._angmax = sum(lasermeta[0:2])
                self._maxrange = lasermeta[3]

                fov = np.degrees([self._angmin, self._angmax])
                print(f"  {nlaser} laser scans: {nbeams} beams, fov {fov[0]:.1f}° to {fov[1]:.1f}°, max range {self._maxrange}")

    
    def scan(self, i):
        v = self.vindex[i]
        return v.range, v.theta
    
    def scanxy(self, i):
        
        range, theta = self.scan(i)
        range = np.where(range < self._maxrange, range, np.nan)
        x = range * np.cos(theta)
        y = range * np.sin(theta)


        return np.c_[x, y].T
    
    def plot_scan(self, n):
        n = base.getvector(n)
        for i in n:
            x, y = self.scanxy(i)
            plt.plot(x, y, '.', 'MarkerSize', 10)
            plt.pause(1)

    def pose(self, i):
        return self.vindex[i].coord
    
    def time(self, i):
        return self.vindex[i].time
    
    def plot(self, **kwargs):
        if not 'vertex' in kwargs:
            kwargs['vertex'] = dict(markersize=8, markerfacecolor='blue', markeredgecolor='None')
        if not 'edge' in kwargs:
            kwargs['edge'] = dict(linewidth=1, color='black')
        self.graph.plot(colorcomponents=False, **kwargs)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
    
    def scanmap(self, occgrid, maxrange=None):
        # note about maxrange timing

        bar = FillingCirclesBar('Converting', max=self.graph.n, 
            suffix = '%(percent).1f%% - %(eta)ds')

        grid1d = occgrid.ravel
        for i in range(0, self.graph.n, 5):
            
            xy = self.scanxy(i)
            r, theta = self.scan(i)
            if maxrange is not None:
                toofar = np.where(r > maxrange)[0]
                xy = np.delete(xy, toofar, axis=1)
            xyt = self.vindex[i].coord
            
            xy = SE2(xyt) * xy
            
            # start of each ray
            p1 = occgrid.w2g(xyt[:2])
            
            for p2 in occgrid.w2g(xy.T):

                # all cells along the ray
                try:
                    k = occgrid.line(p1, p2)
                except ValueError:
                    # silently ignore rays to points outside the grid map
                    continue
                
                # increment cells along the ray, these are free space
                grid1d[k[:-1]] += 1
                # decrement target cell
                grid1d[k[-1]] -= 1

            bar.next()

        grid1d[grid1d < 0] = -1
        grid1d[grid1d > 0] = 1
        bar.finish()
    
    def w2g(self, w):
        return np.round((w - self._centre) / self._cellsize) + self._ngrid / 2

    def g2w(self, g):
        return ( np.r_[g] - self._ngrid / 2) * self._cellsize + self._centre

    def plot_occgrid(self, w, block=True):
        
        bl = self.g2w([0,0])
        tr = self.g2w([w.shape[1], w.shape[0]])

        w = np.where(w < 0, -1, w)  # free cells are -1
        w = np.where(w > 0, 1, w)   # occupied cells are +1
        w = -w 

        # plot it
        plt.imshow(w, cmap='gray', extent=[bl[0], tr[0], bl[1], tr[1]])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show(block=block)

    
#   This source code is part of the graph optimization package
#   deveoped for the lectures of robotics2 at the University of Freiburg.
#
#     Copyright (c) 2007 Giorgio Grisetti, Gian Diego Tipaldi
#
#   It is licences under the Common Creative License,
#   Attribution-NonCommercial-ShareAlike 3.0
#
#   You are free:
#     - to Share - to copy, distribute and transmit the work
#     - to Remix - to adapt the work
#
#   Under the following conditions:
#
#     - Attribution. You must attribute the work in the manner specified
#       by the author or licensor (but not in any way that suggests that
#       they endorse you or your use of the work).
#
#     - Noncommercial. You may not use this work for commercial purposes.
#
#     - Share Alike. If you alter, transform, or build upon this work,
#       you may distribute the resulting work only under the same or
#       similar license to this one.
#
#   Any of the above conditions can be waived if you get permission
#   from the copyright holder.  Nothing in this license impairs or
#   restricts the author's moral rights.
#
#   This software is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied
#   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
#   PURPOSE.
    
    
# ls-slam.m
# this file is released under the creative common license

# solves a graph-based slam problem via least squares
# vmeans: matrix containing the column vectors of the poses of the vertices
# 	 the vertices are odrered such that vmeans[i] corresponds to the ith id
# eids:	 matrix containing the column vectors [idFrom, idTo]' of the ids of the vertices
# 	 eids[k] corresponds to emeans[k] and einfs[k].
# emeans: matrix containing the column vectors of the poses of the edges
# einfs:  3d matrix containing the information matrices of the edges
# 	 einfs(:,:,k) refers to the information matrix of the k-th edge.
# n:	 number of iterations
# newmeans: matrix containing the column vectors of the updated vertices positions
    
    def optimize(self, iterations = 10, animate = False, retain = False, **kwargs):
        
        eprev  =  math.inf
        for i in range(iterations):
            if animate:
                if not retain:
                    plt.clf()
                self.graph.plot(**kwargs)
                plt.pause(0.5)
            
            energy = self.linearize_and_solve()
            
            if energy >= eprev:
                break
            eprev = energy

    # linearizes and solves one time the ls-slam problem specified by the input
    # vmeans:   vertices positions at the linearization point
    # eids:     edge ids
    # emeans:   edge means
    # einfs:    edge information matrices
    # newmeans: new solution computed from the initial guess in vmeans
    def linearize_and_solve(self):
        t0  =  time.time()

        g = self.graph
        
        # H and b are respectively the system matrix and the system vector
        H = np.zeros((g.n * 3, g.n * 3))
        b = np.zeros((g.n * 3, 1))
        # this loop constructs the global system by accumulating in H and b the contributions
        # of all edges (see lecture)
        
        etotal = 0
        for edge  in g.edges():
            e, A, B = edge.linear_factors()
            omega  =  edge.info

            # compute the blocks of H^k
            bi  = -A.T @ omega @ e
            bj  = -B.T @ omega @ e
            Hii =  A.T @ omega @ A
            Hij =  A.T @ omega @ B
            Hjj =  B.T @ omega @ B
            
            v = edge.endpoints
            i = v[0].index
            j = v[1].index

            islice = slice(i*3, (i+1)*3)
            jslice = slice(j*3, (j+1)*3)

            # accumulate the blocks in H and b
            H[islice, islice] += Hii
            H[jslice, jslice] += Hjj
            H[islice, jslice] += Hij
            H[jslice, islice] += Hij.T

            b[islice,0] += bi
            b[jslice,0] += bj

            etotal += np.inner(e, e)

        # note that the system (H b) is obtained only from
        # relative constraints. H is not full rank.
        # we solve the problem by anchoring the position of
        # the the first vertex.
        # this can be expressed by adding the equation
        #  deltax(1:3,1) = 0
        # which is equivalent to the following

        H[0:3, 0:3] += np.eye(3)
        
        SH = sp.sparse.csr_matrix(H)
        deltax = sp.sparse.linalg.spsolve(SH, b)  # SH \ b
        
        # split the increments in nice 3x1 vectors and sum them up to the original matrix
        for vertex in g:
            i = vertex.index * 3
            vertex.coord += deltax[i: i+3]
            # normalize the angles between -PI and PI
            vertex.coord[2] = base.angdiff(vertex.coord[2])

        dt  =  time.time() - t0
        print(f"done in {dt*1e3:0.2f} msec.  Total cost {etotal:g}")

        return etotal

    #   This source code is part of the graph optimization package
    #   deveoped for the lectures of robotics2 at the University of Freiburg.
    #
    #     Copyright (c) 2007 Giorgio Grisetti, Gian Diego Tipaldi
    #
    #   It is licences under the Common Creative License,
    #   Attribution-NonCommercial-ShareAlike 3.0
    #
    #   You are free:
    #     - to Share - to copy, distribute and transmit the work
    #     - to Remix - to adapt the work
    #
    #   Under the following conditions:
    #
    #     - Attribution. You must attribute the work in the manner specified
    #       by the author or licensor (but not in any way that suggests that
    #       they endorse you or your use of the work).
    #
    #     - Noncommercial. You may not use this work for commercial purposes.
    #
    #     - Share Alike. If you alter, transform, or build upon this work,
    #       you may distribute the resulting work only under the same or
    #       similar license to this one.
    #
    #   Any of the above conditions can be waived if you get permission
    #   from the copyright holder.  Nothing in this license impairs or
    #   restricts the author's moral rights.
    #
    #   This software is distributed in the hope that it will be useful,
    #   but WITHOUT ANY WARRANTY; without even the implied
    #   warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
    #   PURPOSE.


if __name__ == "__main__":

    scan = PoseGraph('killian.g2o', laser=True, verbose=False)
    print(scan.graph.nc)
    # scan.plot()
    w = scan.scanmap(maxrange=40)
    scan.plot_occgrid(w)