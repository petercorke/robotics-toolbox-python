from abc import ABC
import numpy as np
import scipy as sp
from math import pi, sin, cos
import matplotlib.pyplot as plt
from spatialmath import base
import roboticstoolbox as rtb

class LandmarkMap:
    """
    LandmarkMap Map of planar point landmarks

    A LandmarkMap object represents a square 2D environment with a number of landmark
    landmark pointself.

    Methods::
      plot      Plot the landmark map
      landmark   Return a specified map landmark
      display   Display map parameters in human readable form
      char      Convert map parameters to human readable string

    Properties::
      map         Matrix of map landmark coordinates 2xN
      dim         The dimensions of the map region x,y in [-dim,dim]
      nlandmarks   The number of map landmarks N

    Examples::

    To create a map for an area where X and Y are in the range -10 to +10 metres
    and with 50 random landmark points
           map = LandmarkMap(50, 10)
    which can be displayed by
           self.plot()

    Reference::

      Robotics, Vision & Control, Chap 6,
      Peter Corke,
      Springer 2011

    See also RangeBearingSensor, EKF.
    """
    # TODO:
    # add a name property, show in char()

    #     properties
    #         map    # map landmarks
    #         dim         # map dimension
    #         nlandmarks   # number of landmarks in map

    #         verbose
    #     end


    def __init__(self, nlandmarks, dim=10, verbose=True, seed=None):
        """
        Create a map of point landmark landmarks
        %
        # M = LandmarkMap(N, DIM, OPTIONS) is a LandmarkMap object that represents N random point landmarks
        # in a planar region bounded by +/-DIM in the x- and y-directionself.
        %
        # Options::
        # 'verbose'    Be verbose
            
        
        %# TODO: dim can be a 4-vector
        """
    
        self._nlandmarks = nlandmarks
        self._dim = base.expand_dims(dim)

        random = np.random.default_rng(seed)
        x = random.uniform(self._dim[0], self._dim[1], nlandmarks)
        y = random.uniform(self._dim[2], self._dim[3], nlandmarks)

        self._map = np.c_[x, y].T
        self._verbose = verbose


    def __str__(self):
    # s = M.char() is a string showing map parameters in 
    # a compact human readable format. 
        s = f"LandmarkMap object with {self._nlandmarks} landmarks, dim=" + str(self._dim)
        return s

    @property
    def nlandmarks(self):
        return self._nlandmarks

    @property
    def x(self):
        return self._map[0,:]

    @property
    def y(self):
        return self._map[1,:]

    @property
    def xy(self):
        return self._map

    def landmark(self, k):
        """
        Get k'th landmarks from map
        %
        # F = M.landmark(K) is the coordinate (2x1) of the K'th landmark (landmark).
        """
        return self._map[:,k]

    def plot(self, block=False, **kwargs):
        """
        Plot the map
        %
        # M.plot() plots the landmark map in the current figure, as a square
        # region with dimensions given by the M.dim property.  Each landmark
        # is marked by a black diamond.
        %
        # M.plot(LS) as above, but the arguments LS
        # are passed to plot and override the default marker style.
        %
        # Notes::
        # - The plot is left with HOLD ON.
        """
        ax = base.plotvol2(self._dim)
        ax.set_aspect('equal')

        ax.set_xlabel('x')
        ax.set_ylabel('y')

        if len(kwargs) == 0:
            kwargs = {
                'linewidth': 0,
                'marker': 'P',
                'color': 'k',
            }

        plt.plot(self._map[0,:], self._map[1,:], **kwargs)
        plt.grid(True)
        plt.show(block=block)

if __name__ == "__main__":
    import unittest

