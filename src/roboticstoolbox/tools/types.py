#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from collections.abc import Sequence

from numpy import ndarray

NDArray = ndarray

# a list, tuple, range or any other sequence of numbers
PyArrayLike = Sequence[float]

# the argument type for anything array-like, use NDArray for returned arrays
ArrayLike = NDArray | PyArrayLike
