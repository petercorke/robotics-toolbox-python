#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from typing import Tuple, Union, List, Set
from numpy import ndarray

NDArray = ndarray

PyArrayLike = Union[List[float], Tuple[float], Set[float]]

ArrayLike = Union[NDArray, PyArrayLike]
