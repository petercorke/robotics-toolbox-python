#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:41:25 2020

@author: corkep
"""
import numpy as np
import math
import unittest

def matrix(m, shape):
    assert ismatrix(m, shape)

def ismatrix(m, shape):
    return type(m) == np.ndarray and m.shape == shape

 #and not np.iscomplex(m) checks every element, would need to be not np.any(np.iscomplex(m)) which seems expensive

def getvector(v, dim=None, out='array'):
    if isinstance(v, (int,float)): # handle scalar case
        v = [v]
        
    if isinstance(v, (list,tuple)):
        if dim and v and len(v) != dim:
            raise ValueError("incorrect vector length")
        if out == 'sequence':
            return v
        elif out == 'array':
            return np.array(v)
        elif out == 'row':
            return np.array(v).reshape(1, -1)
        elif out == 'col':
            return np.array(v).reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")
    elif isinstance(v, np.ndarray):
        s = v.shape
        if dim:
            if not ( s == (dim,) or s == (1,dim) or s == (dim,1) ):
                raise ValueError("incorrect vector length")

        v = v.flatten()

        if out == 'sequence':
            return list(v.flatten())
        elif out == 'array':
            return v
        elif out == 'row':
            return v.reshape(1, -1)
        elif out == 'col':
            return v.reshape(-1, 1)
        else:
            raise ValueError("invalid output specifier")
    else:
        raise ValueError("invalid input type")

def vector(v, dim):
    assert isvector(v, dim)
    
def isvector(v, dim=None, out='sequence'):
    if isinstance(v, (list, tuple)) and (dim is None or len(v) == dim) and all(map(lambda x: isinstance(x, (int, float)), v)):
        return True  # list or tuple
    
    if isinstance(v, np.ndarray):
        s = v.shape
        if dim is None:
            return (len(s) == 1 and s[0] > 0) or (s[0] == 1 and s[1] > 0) or (s[0] > 0 and s[1] == 1)
        else:
            return s == (dim,) or s == (1,dim) or s == (dim,1)
    
    if (dim is None or dim == 1) and isinstance(v, (int,float)):
        return True
    
    return False
        
def isscalar(v):
    return isinstance(v, (int,float))
    
def getunit(v, u):
    if u == "rad":
        return v
    elif u == "deg":
        if isinstance(v, np.ndarray) or np.isscalar(v):
            return v * math.pi / 180
        else:
            return [x * math.pi / 180 for x in v]
    else:
        raise ValueError("invalid angular units")
        
def isnumberlist(l):
    return isinstance(l, (list,tuple)) and len(l) > 0 and all( map( lambda x: isinstance(x, (float, int)), l) )

def isvectorlist(l, n):
    return isinstance(l, (list,tuple)) and len(l) > 0 and all( map( lambda x: isinstance(x, np.ndarray) and len(x) == n, l) )
        
        
if __name__ == '__main__':

    import numpy.testing as nt
        
    class Test_check(unittest.TestCase):
        
        def test_unit(self):
            
            nt.assert_equal(getunit(5, 'rad'), 5)
            nt.assert_equal(getunit(5, 'deg'), 5*math.pi/180.0)
            nt.assert_equal(getunit([3,4,5], 'rad'), [3,4,5])
            nt.assert_equal(getunit([3,4,5], 'deg'), [x*math.pi/180.0 for x in [3,4,5]])
            nt.assert_equal(getunit((3,4,5), 'rad'), [3,4,5])
            nt.assert_equal(getunit((3,4,5), 'deg'), [x*math.pi/180.0 for x in [3,4,5]])

            nt.assert_equal(getunit(np.array([3,4,5]), 'rad'), [3,4,5])
            nt.assert_equal(getunit(np.array([3,4,5]), 'deg'), [x*math.pi/180.0 for x in [3,4,5]])
            
        def test_isvector(self):
            # no length specified
            nt.assert_equal(isvector(2), True)
            nt.assert_equal(isvector(2.0), True)
            nt.assert_equal(isvector([1,2,3]), True)
            nt.assert_equal(isvector((1,2,3)), True)
            nt.assert_equal(isvector(np.array([1,2,3])), True)
            nt.assert_equal(isvector(np.array([[1,2,3]])), True)
            nt.assert_equal(isvector(np.array([[1],[2],[3]])), True)
            
            # length specified
            nt.assert_equal(isvector(2, 1), True)
            nt.assert_equal(isvector(2.0, 1), True)
            nt.assert_equal(isvector([1,2,3], 3), True)
            nt.assert_equal(isvector((1,2,3), 3), True)
            nt.assert_equal(isvector(np.array([1,2,3]), 3), True)
            nt.assert_equal(isvector(np.array([[1,2,3]]), 3), True)
            nt.assert_equal(isvector(np.array([[1],[2],[3]]), 3), True)
            
            # wrong length specified
            nt.assert_equal(isvector(2, 4), False)
            nt.assert_equal(isvector(2.0, 4), False)
            nt.assert_equal(isvector([1,2,3], 4), False)
            nt.assert_equal(isvector((1,2,3), 4), False)
            nt.assert_equal(isvector(np.array([1,2,3]), 4), False)
            nt.assert_equal(isvector(np.array([[1,2,3]]), 4), False)
            nt.assert_equal(isvector(np.array([[1],[2],[3]]), 4), False)
            
        def test_isvector(self):
            l = [1,2,3]
            nt.assert_raises(AssertionError, vector, l, 4 )
            
        def test_getvector(self):
            l = [1,2,3]
            t = (1,2,3)
            a = np.array(l)
            r = np.array([[1,2,3]])
            c = np.array([[1],[2],[3]])
            
            # input is list
            v = getvector(l)
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(l, 3)
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(l, out='sequence')
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(l, 3, out='sequence')
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(l, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(l, 3, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(l, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(l, 3, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(l, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            v = getvector(l, 3, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            nt.assert_raises(ValueError, getvector, l, 4 )
            nt.assert_raises(ValueError, getvector, l, 4, 'sequence' )
            nt.assert_raises(ValueError, getvector, l, 4, 'array' )
            nt.assert_raises(ValueError, getvector, l, 4, 'row' )
            nt.assert_raises(ValueError, getvector, l, 4, 'col' )
            
            # input is tuple
            
            v = getvector(t)
            nt.assert_equal(isinstance(v, tuple), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(t, 3)
            nt.assert_equal(isinstance(v, tuple), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(t, out='sequence')
            nt.assert_equal(isinstance(v, tuple), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(t, 3, out='sequence')
            nt.assert_equal(isinstance(v, tuple), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(t, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(t, 3, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(t, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(t, 3, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(t, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            v = getvector(t, 3, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            nt.assert_raises(ValueError, getvector, t, 4 )
            nt.assert_raises(ValueError, getvector, t, 4, 'sequence' )
            nt.assert_raises(ValueError, getvector, t, 4, 'array' )
            nt.assert_raises(ValueError, getvector, t, 4, 'row' )
            nt.assert_raises(ValueError, getvector, t, 4, 'col' )
            
            # input is array
            
            v = getvector(a)
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(a, 3)
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(a, out='sequence')
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(a, 3, out='sequence')
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(a, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(a, 3, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(a, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(a, 3, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(a, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            v = getvector(a, 3, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            nt.assert_raises(ValueError, getvector, a, 4 )
            nt.assert_raises(ValueError, getvector, a, 4, 'sequence' )
            nt.assert_raises(ValueError, getvector, a, 4, 'array' )
            nt.assert_raises(ValueError, getvector, a, 4, 'row' )
            nt.assert_raises(ValueError, getvector, a, 4, 'col' )

            # input is row
            
            v = getvector(r)
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(r, 3)
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(r, out='sequence')
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(r, 3, out='sequence')
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(r, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(r, 3, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(r, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(r, 3, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(r, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            v = getvector(r, 3, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            nt.assert_raises(ValueError, getvector, r, 4 )
            nt.assert_raises(ValueError, getvector, r, 4, 'sequence' )
            nt.assert_raises(ValueError, getvector, r, 4, 'array' )
            nt.assert_raises(ValueError, getvector, r, 4, 'row' )
            nt.assert_raises(ValueError, getvector, r, 4, 'col' )

            # input is col
            
            v = getvector(c)
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(c, 3)
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(c, out='sequence')
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(c, 3, out='sequence')
            nt.assert_equal(isinstance(v, list), True)
            nt.assert_equal(len(v), 3)
            
            v = getvector(c, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(c, 3, out='array')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,))
            
            v = getvector(c, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(c, 3, out='row')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (1,3))
            
            v = getvector(c, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            v = getvector(c, 3, out='col')
            nt.assert_equal(isinstance(v, np.ndarray), True)
            nt.assert_equal(v.shape, (3,1))
            
            nt.assert_raises(ValueError, getvector, c, 4 )
            nt.assert_raises(ValueError, getvector, c, 4, 'sequence' )
            nt.assert_raises(ValueError, getvector, c, 4, 'array' )
            nt.assert_raises(ValueError, getvector, c, 4, 'row' )
            nt.assert_raises(ValueError, getvector, c, 4, 'col' )            
            

        
        def test_isnumberlist(self):
            nt.assert_equal(isnumberlist([1]), True)
            nt.assert_equal(isnumberlist([1,2]), True)
            nt.assert_equal(isnumberlist((1,)), True)
            nt.assert_equal(isnumberlist((1,2)), True)
            nt.assert_equal(isnumberlist(1), False)
            nt.assert_equal(isnumberlist([]), False)
            nt.assert_equal(isnumberlist(np.array([1,2,3])), False)
            
            
    unittest.main()    
    