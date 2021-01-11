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

        path = rtb.path_to_datafile(filename)

        if filename.endswith('.zip'):
            zf = zipfile.ZipFile(path, 'r')
            opener = zf.open
            filename = filename[:-4]
        else:
            opener = open
            filename = path

        with opener(filename, 'r') as f:

            toroformat = False
            nlaser = 0
            
            # indices into ROBOTLASER1 record for the 3x3 info matrix in column major
            # order
            g2o =  [0,  1,  2,  1,  3,  4,   2,  4,  5]
            toro = [0,  1, 4,  1,  2,  5,  4,  5,   3]
            
            # we keep an array self. = vindex(gi) to map g2o vertex index to PGraph vertex index
            
            vindex = {}
            firstlaser = True
            for line in f:
                # for zip file, we get data as bytes not str
                if isinstance(line, bytes):
                    line = line.decode()
                
                # is it a comment?
                if line.startswith('#'):
                    continue
                
                tokens = line.split(' ')
                
                # g2o format records
                if tokens[0] == 'VERTEX_SE2':
                    v = self.graph.add_vertex([float(x) for x in tokens[2:5]])
                    id = int(tokens[1])
                    vindex[id] = v
                    v.id = id

                    v.type = 'vertex'
                        
                elif tokens[0] ==  'VERTEX_XY':
                    v = self.graph.add_vertex([float(x) for x in tokens[2:4]])
                    id = int(tokens[1])
                    vindex[id] = v
                    v.id = id
                    v.type = 'landmark'
                        
                elif tokens[0] ==  'EDGE_SE2':                    
                    v1 = vindex[int(tokens[1])]
                    v2 = vindex[int(tokens[2])]

                    # create the edge
                    e = self.graph.add_edge(v1, v2)

                    # create the edge data as a structure
                    #  X  Y  T
                    #  3  4  5
                    e.mean = np.array([float(x) for x in tokens[3:6]])
                    
                    # IXX IXY IXT IYY IYT ITT
                    #   6   7   8   9  10  11
                    info = np.array([float(x) for x in tokens[6:12]])
                    e.info = np.reshape(info[g2o], (3,3))
                        
                ## TORO format records
                elif tokens[0] == 'VERTEX2':
                    toroformat = True
                    v = self.graph.add_vertex([float(x) for x in tokens[2:5]])
                    id = int(tokens[1])
                    vindex[id] = v
                    v.id = id
                    v.type = 'vertex'
                        
                elif tokens[0] == 'EDGE2':
                    toroformat = True
                    v1 = vindex[int(tokens[1])]
                    v2 = vindex[int(tokens[2])]

                    # create the edge
                    e = self.graph.add_edge(v1, v2)

                    # create the edge data as a structure
                    #  X  Y  T
                    #  3  4  5
                    e.mean = [float(x) for x in tokens[3:6]]
                    
                    # IXX IXY IXT IYY IYT ITT
                    #   6   7   8   9  10  11
                    info = np.array([float(x) for x in tokens[6:12]])
                    e.info = np.reshape(info[toro], (3,3))
                        
                elif tokens[0] == 'ROBOTLASER1':
                    if not laser:
                        continue
                    
                    # laser records are associated with the immediately preceding VERTEX record

                    # not quite sure what all the fields are
                    # 1 ?
                    # 2 min scan angle
                    # 3 scan range
                    # 4 angular increment
                    # 5 maximum range possible
                    # 6 ?
                    # 7 ?
                    # 8 N = number of beams
                    # 9 to 9+N laser range data
                    # 9+N+1 ?
                    # 9+N+2 ?
                    # 9+N+3 ?
                    # 9+N+4 ?
                    # 9+N+5 ?
                    # 9+N+6 ?
                    # 9+N+7 ?
                    # 9+N+8 ?
                    # 9+N+9 ?
                    # 9+N+10 ?
                    # 9+N+11 ?
                    # 9+N+12 timestamp (*nix timestamp)
                    # 9+N+13 laser type (str)
                    # 9+N+14  ?

                    if firstlaser:
                        nbeams = int(tokens[8])
                        lasermeta = tokens[2:6]
                        firstlaser = False

                    v.theta = np.arange(0, nbeams) * float(tokens[4]) + float(tokens[2])
                    v.range = np.array([float(x) for x in tokens[9:nbeams+9]])
                    v.time = float(tokens[21+nbeams])
                    nlaser+= 1
                else:
                    raise RuntimeError(f"Unexpected line  {line} in {filename}")

            if toroformat:
                print(f"loaded TORO/LAGO format file: {self.graph.n} nodes, {self.graph.ne} edges")
            else:
                print(f"loaded g2o format file: {self.graph.n} nodes, {self.graph.ne} edges")
            if nlaser > 0:
                lasermeta = [float(x) for x in lasermeta]
                
                self._angmin = lasermeta[0]
                self._angmax = sum(lasermeta[0:2])
                self._maxrange = lasermeta[3]

                fov = np.degrees([self._angmin, self._angmax])
                print(f"  {nlaser} laser scans: {nbeams} beams, fov {fov[0]:.1f}° to {fov[1]:.1f}°, max range {self._maxrange}")

        self.vindex = vindex
    
    def scan(self, i):
        v = self.vindex[i]
        return v.range, v.theta
    
    def scanxy(self, i):
        v = self.vindex[i]
        
        range, theta = self.scan(i)
        x = range * np.cos(theta)
        y = range * np.sin(theta)

        return np.c_[x,y]
    
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
        self.graph.plot(**kwargs)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
    
    def scanmap(self, centre=(75,50), ngrid=3000, cellsize=0.1, maxrange=None):
        
        self._centre = centre
        self._cellsize = cellsize
        self._ngrid = ngrid

        # h = waitbar(0, 'rendering a map')
        
        world = np.zeros((ngrid, ngrid), np.int32)
        for i in range(0, self.graph.n):
            
            # if i % 20 == 0:
            #     waitbar(i/self.graph.n, h)
            
            xy = self.scanxy(i)
            r,theta = self.scan(i)
            if maxrange is not None:
                xy = np.delete(xy, r > maxrange, axis=0)
            xyt = self.vindex[i].coord
            
            xy = SE2(xyt) * xy.T
            
            # start of each ray
            p1 = self.w2g(xyt[0:2])
            
            for end in xy.T:
                
                # end of each ray
                p2 = self.w2g(end[0:2])
                
                # all cells along the ray
                x, y = bresenham(p1, p2)

                try:
                    # decrement cells along the ray, these are free space
                    world[y[:-1],x[:-1]] = world[y[:-1],x[:-1]] - 1
                    # increment target cell
                    world[y[-1],x[-1]] = world[y[-1],x[-1]] + 1
                except IndexError:
                    # silently ignore rays to points outside the grid map
                    pass

        return world
    
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

def bresenham(p1, p2):
    
    # ensure all values are integer
    x1 = round(p1[0]); y1 = round(p1[1])
    x2 = round(p2[0]); y2 = round(p2[1])

    # compute the vertical and horizontal change
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steep = dy > dx
    
    if steep:
        # if slope > 1 swap the deltas
        t = dx
        dx = dy
        dy = t

    # The main algorithm goes here.
    if dy == 0:
        q = np.zeros((dx+1,1), np.int64)
    else:
        q = np.arange(np.floor(dx / 2), -dy * (dx + 1) + np.floor(dx / 2), -dy)
        q = np.r_[0, np.diff(np.mod(q, dx)) >= 0]
            
    if steep:
        if y1 <= y2:
            y = np.arange(y1, y2 + 1)
        else:
            y = np.arange(y1, y2 - 1, -1)

        if x1 <= x2:
            x = x1 + np.cumsum(q)
        else:
            x = x1 - np.cumsum(q)
    else:
        if x1 <= x2:
            x = np.arange(x1, x2 + 1)
        else:
            x = np.arange(x1, x2 - 1, -1)

        if y1 <= y2:
            y = y1 + np.cumsum(q)
        else:
            y = y1 - np.cumsum(q)
    
    return x, y




    
    
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
    
    
    # %ls-slam.m
    # %this file is released under the creative common license
    
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
    
    def optimize(self, iterations = 10, animate = False, retain = False):
        
        
        g2  =  PGraph(self.graph)  # deep copy
        
        eprev  =  math.inf
        for i in range(iterations):
            if animate:
                if not retain:
                    plt.clf()
                g2.plot()
                plt.pause(0.5)
            
            vmeans, energy = linearize_and_solve(g2)
            g2.setcoord(vmeans)
            
            if energy >= eprev:
                break
            eprev = energy
        
        self.graph = g2

        return g2


#computes the taylor expansion of the error function of the k_th edge
#vmeans: vertices positions
#eids:   edge ids
#emeans: edge means
#k:	 edge number
#e:	 e_k(x)
#A:	 d e_k(x) / d(x_i)
#B:	 d e_k(x) / d(x_j)
#function [e, A, B] = linear_factors(vmeans, eids, emeans, k)
def linear_factors(self, edge):
    # extract the ids of the vertices connected by the kth edge
    # 	id_i = eids(1,k)
    # 	id_j = eids(2,k)
    # extract the poses of the vertices and the mean of the edge
    #     v_i = vmeans(:,id_i)
    #     v_j = vmeans(:,id_j)
    #     z_ij = emeans(:,k)

    v  =  g.vertices(edge)
    v_i  =  g.coord(v(1))
    v_j  =  g.coord(v(2))
    z_ij  =  g.edata(edge).mean

    # compute the homoeneous transforms of the previous solutions
    zt_ij = base.trot2(z_ij[2], t=z_ij[0:2])
    vt_i = base.trot2(v_i[2], t=v_i[0:2])
    vt_j = base.trot2(v_j[2], t=v_j[0:2])

    # compute the displacement between x_i and x_j

    f_ij = base.trinv2(vt_i @ vt_j)

    # this below is too long to explain, to understand it derive it by hand
    theta_i = v_i[3]
    ti = v_i[0:2,0]
    tj = v_j[0:2,0]
    dt_ij = tj - ti

    si = sin(theta_i)
    ci = cos(theta_i)

    A =  np.array([
        [-ci, -si, [-si,  ci] @ dt_ij],
        [ si, -ci, [-ci, -si] @ dt_ij],
        [  0,   0, -1 ],
            ])
    B  = np.array([
        [  ci, si, 0],
        [ -si, ci, 0],
        [  0,   0, 1 ],
            ])

    ztinv = base.trinv2(zt_ij)
    T = ztinv @ f_ij
    e = np.r_[base.transl2(T), base.angle(T)]
    ztinv[0:2,2] =  0
    A = ztinv @ A
    B = ztinv @ B

    return e, A, B 



# linearizes and solves one time the ls-slam problem specified by the input
# vmeans:   vertices positions at the linearization point
# eids:     edge ids
# emeans:   edge means
# einfs:    edge information matrices
# newmeans: new solution computed from the initial guess in vmeans
def linearize_and_solve(self):
    t0  =  time.time()
    print('solving')
    
    # H and b are respectively the system matrix and the system vector
    H = np.zeros((self.n * 3, self.n * 3))
    b = zeros(g.n * 3,1)
    # this loop constructs the global system by accumulating in H and b the contributions
    # of all edges (see lecture)
    fprintf('.')
    
    etotal  =  0
    for edge  in self.edges:
        
        e, A, B = self.linear_factors(edge)
        omega  =  g.edata(edge).info
        # compute the blocks of H^k
        
        # not quite sure whey SE3 is being transposed, what does that mean?
        b_i   =  -A.T @ omega @ e
        b_j   =  -B.T @ omega @ e
        H_ii  =   A.T @ omega @ A
        H_ij  =   A.T @ omega @ B
        H_jj  =   B.T @ omega @ B
        
        v = g.vertices(edge)
        
        i = v[0]; j = v[1]

        islice = slice(i*3, (i+1)*3)
        jslice = slice(j*3, (j+1)*3)

        # accumulate the blocks in H and b

        H[islice, islice] += Hii
        H[jslice, jslice] += Hjj
        H[islice, jslice] += Hij
        H[jslice, islice] += Hij.T

        b[islice,0] += b_i
        b[jslice,0] += b_j

        etotal  =  etotal + np.inner(e, e)
        printf('.', end=None)

    
    # %note that the system (H b) is obtained only from
    # %relative constraints. H is not full rank.
    # %we solve the problem by anchoring the position of
    # %the the first vertex.
    # %this can be expressed by adding the equation
    #  deltax(1:3,1) = 0
    # which is equivalent to the following

    H[0:3,0:3] += np.eye(3)
    
    SH = sp.bsr_sparse(H)
    print('.', end=None)
    deltax = sp.spsolve(SH, b)  # SH \ b
    print('.', end=None)
    
    # split the increments in nice 3x1 vectors and sum them up to the original matrix
    newmeans  =  self.coord() + np.reshape(deltax, (3, self.n))
    
    # normalize the angles between -PI and PI
    # for (i = 1:size(newmeans,2))
    #     s = sin(newmeans(3,i))
    #     c = cos(newmeans(3,i))
    #     newmeans(3,i) = atan2(s,c)
    newmeans[3,:] = base.angdiff(newmeans[3,:])

    dt  =  time.time() - t0
    print(f"done in {dt:0.2f} sec.  Total cost {etotal}")


    return newmeans, energy

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



# #computes the pose vector v from an homogeneous transform A
# function v = t2v(A)
#     v(1:2, 1) = A(1:2,3)
#     v(3,1) = atan2(A(2,1),A(1,1))
# end


if __name__ == "__main__":

    scan = PoseGraph('killian.g2o', laser=True, verbose=False)
    print(scan.graph.nc)
    # scan.plot()
    w = scan.scanmap(maxrange=40)
    scan.plot_occgrid(w)