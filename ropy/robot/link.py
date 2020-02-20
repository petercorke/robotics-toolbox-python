#! /usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class Link(object):
    """
    A link superclass for all link types
    
    Attributes:
    --------
        theta : float
            kinematic: joint angle
        d : float
            kinematic: link offset
        alpha : float
            kinematic: link twist
        a : float
            kinematic: link length
        sigma : int
            kinematic: 0 if revolute, 1 if prismatic
        mdh : int
            kinematic: 0 if standard D&H, else 1
        offset : float
            kinematic: joint variable offset
        qlim : float np.ndarray(1,2)
            kinematic: joint variable limits [min max]
        flip : bool
            joint moves in opposite direction

    See Also
    --------
    ropy.robot.Revolute : A revolute link class
    """

    def __init__(
            self, 
            theta = 0, 
            d = 0,
            alpha = 0,
            a = 0,
            sigma = 0,
            mdh = 0,
            offset = 0,
            qlim = 0,
            flip = False
            ):

        self.theta = theta
        self.d = d
        self.alpha = alpha
        self.a = a
        self.sigma = sigma
        self.mdh = mdh
        self.offset = offset
        self.flip = flip   
        self.qlim = qlim        


    def A(self, q):
        """
        A Link transform matrix. T = L.A(Q) is the link homogeneous
        transformation matrix (4x4) corresponding to the link variable Q
        which is either the Denavit-Hartenberg parameter THETA (revolute)
        or D (prismatic)

        Notes:
        - For a revolute joint the THETA parameter of the link is ignored,
          and Q used instead.
        - For a prismatic joint the D parameter of the link is ignored, and
          Q used instead.
        - The link offset parameter is added to Q before computation of the
          transformation matrix.
        
        Parameters
        ----------
        q : float
            Joint angle (radians)

        Returns
        -------
        T : float numpy.ndarray((4, 4))
            link homogeneous transformation matrix
        
        See Also
        --------
        ropy.robot.Revolute : A revolute link class
        """

        sa = np.sin(self.alpha)
        ca = np.cos(self.alpha)

        if self.flip:
            q = -q + self.offset
        else:
            q = q + self.offset


        if self.sigma == 0:
            # revolute
            st = np.sin(q)
            ct = np.cos(q)
            d = self.d
        else:
            # prismatic
            st = np.sin(self.theta)
            ct = np.cos(self.theta)
            d = q

        
        if self.mdh == 0:
            # standard DH
            T = np.array([ [ ct,  -st*ca,   st*sa,   self.a*ct  ],
                           [ st,   ct*ca,  -ct*sa,   self.a*st  ],
                           [ 0,    sa,      ca,      d          ],
                           [ 0,    0,       0,       1          ] ])
        else:
            # modified DH
            T = np.array([ [ ct,      -st,       0,     self.a  ],
                           [ st*ca,    ct*ca,   -sa,   -sa*d    ],
                           [ st*sa,    ct*sa,    ca,    ca*d    ],
                           [ 0,        0,        0,     1       ] ])

        return T



class Revolute(Link):
    """
    A class for revolute link types
    
    Attributes:
    --------
        theta : float
            kinematic: joint angle
        d : float
            kinematic: link offset
        alpha : float
            kinematic: link twist
        a : float
            kinematic: link length
        sigma : int
            kinematic: 0 if revolute, 1 if prismatic
        mdh : int
            kinematic: 0 if standard D&H, else 1
        offset : float
            kinematic: joint variable offset
        qlim : float np.ndarray(1,2)
            kinematic: joint variable limits [min max]
        flip : bool
            joint moves in opposite direction

    See Also
    --------
    ropy.robot.Revolute : A revolute link class
    """

    def __init__(
            self, 
            theta = 0, 
            d = 0,
            alpha = 0,
            a = 0,
            sigma = 0,
            mdh = 0,
            offset = 0,
            qlim = 0,
            flip = False
            ):

        super(Revolute, self).__init__(theta, d, alpha, a, sigma, mdh, offset, qlim, flip)

        if self.d is None:
            self.d = 0

        self.is_revolute = True
        
        if self.theta != 0:
            raise ValueError('Theta cannot be specified for a revolute link')