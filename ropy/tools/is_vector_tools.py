# #! /usr/bin/env python

# import numpy as np


# def is_vector(v):
#     """
#     is_vector returns True if v is a 1-by-n or n-by-1 vector, and False otherwise

#     Parameters
#     ----------
#     v : numpy.ndarray
#         Input array to ckeck

#     Returns
#     -------
#     is_vector : bool
#         True if v is a 1-by-n or n-by-1 vector, True for a 1-by-1

#     Examples
#     --------
#     >>> is_vec = is_vector(np.array([0, 0]))

#     >>> True

#     See Also
#     --------
#     ropy.tools.is_column : returns True if v is an n-by-1 vector
#     ropy.tools.is_row : returns True if v is an 1-by-n vector
#     """

#     if not isinstance(v, np.ndarray):
#         raise TypeError('Input v must be a numpy array')

#     if np.ndim(v) == 1:
#         vec = np.expand_dims(v, 0)
#     else:
#         vec = v
    
#     if np.ndim(vec) == 2:
#         x = np.size(vec, 0)
#         y = np.size(vec, 1)

#         if (x == 1 and y >= 1) or (x >= 1 and y == 1):
#             return True
#         else:
#             return False
#     else:
#         return False



# def is_row(v):
#     """
#     is_row returns True if input is a row vector.

#     Parameters
#     ----------
#     v : numpy.ndarray
#         Input array to ckeck

#     Returns
#     -------
#     is_row : bool
#         True if v is a 1-by-n vector, and False otherwise. A 1x1 array 
#         will return True

#     Examples
#     --------
#     >>> is_row = is_row(np.array([0, 0]))

#     >>> True

#     See Also
#     --------
#     ropy.tools.is_vector : returns True if v is a 1-by-n or n-by-1 vector
#     ropy.tools.is_column : returns True if v is an n-by-1 vector
#     """

#     if not isinstance(v, np.ndarray):
#         raise TypeError('Input v must be a numpy array')

#     if not is_vector(v):
#         return False
    
#     if np.ndim(v) == 1:
#         return True
    
#     if np.ndim(v) == 2:

#         x = np.size(v, 0)

#         if (x == 1):
#             return True
#         else:
#             return False

#     else:
#         return False



# def is_column(v):
#     """
#     is_column returns True if v is an n-by-1 vector.

#     Parameters
#     ----------
#     v : numpy.ndarray
#         Input array to ckeck

#     Returns
#     -------
#     is_column : bool
#         True if v is an n-by-1 vector, and False otherwise. A 1x1 array
#         will return True

#     Examples
#     --------
#     >>> is_column = is_column(np.array([0, 0]))

#     >>> Fasle

#     See Also
#     --------
#     ropy.tools.is_vector : returns True if v is a 1-by-n or n-by-1 vector
#     ropy.tools.is_row : returns True if v is an 1-by-n vector
#     """

#     if not isinstance(v, np.ndarray):
#         raise TypeError('Input v must be a numpy array')

#     if not is_vector(v):
#         return False

#     if np.ndim(v) == 1:
#         if v.size == 1:
#             return True
#         else:
#             return False

#     if np.ndim(v) == 2:

#         y = np.size(v, 1)

#         if (y == 1):
#             return True
#         else:
#             return False

#     else:
#         return False