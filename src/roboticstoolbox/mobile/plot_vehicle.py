"""
Python ReedShepp Planner
@Author: Kristian Gibson
TODO: Comments + Sphynx Docs Structured Text
TODO: Bug-fix, testing
TODO: Add support for extra words
      here: http://planning.cs.uiuc.edu/node822.html
      based the original article here: https://projecteuclid.org/euclid.pjm/1102645450

Not ready for use yet.
"""

from numpy import disp
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
from spatialmath.base.transforms2d import *
from spatialmath.base.vectors import *


class PlotVehicle:
    def __init__(self, covar=None):
        self._covar = np.array([])

    @property
    def l(self):
        return self._l

    # Example
    def init(self, x0=None):
        if x0 is not None:
            self._x = x0
        else:
            self._x = self._x0

        self._x_hist = np.array([])

        if self._driver is not None:
            self._driver.init()  # TODO: make this work?

        self._v_handle = np.array([])


def col_norm(x):
    y = np.array([])
    if x.ndim > 1:
        x = np.column_stack(x)
        for vector in x:
            y = np.append(y, np.linalg.norm(vector))
    else:
        y = np.linalg.norm(x)
    return y
