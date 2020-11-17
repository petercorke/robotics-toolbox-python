#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import numpy as np


def null(A, atol=1e-13, rtol=0):
    '''
    Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of A.

    :param A: A should be at most 2D. A 1-D array with length k will be
        treated as a 2D with shape (1, k)
    :type A: ndarray
    :param atol: The absolute tolerance for a zero singular value. Singular
        values smaller than `atol` are considered to be zero.
    :type atol: float
    :param rtol: The relative tolerance. Singular values less than rtol*smax
        are considered to be zero, where smax is the largest singular value.
    :type rtol: float

    :notes:
        - If both `atol` and `rtol` are positive, the combined tolerance is
          the maximum of the two; that is:
          tol = max(atol, rtol * smax)
        - Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    :return ns: If A is an array with shape (m, k), then ns will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of A.  The columns of ns are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    :rtype ns: ndarray

    '''

    A = np.atleast_2d(A)
    u, s, vh = np.linalg.svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns
