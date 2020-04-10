# Created by: Aditya Dua
# 13 June, 2017

import numpy as np
from abc import ABC, abstractmethod
from collections import UserList
import copy

# colored printing of matrices to the terminal
#   colored package has much finder control than colorama, but the latter is available by default with anaconda
try:
    from colored import fg, bg, attr
    _color = True
    print('using colored output')
except ImportError:
    _color = False
    fg = lambda : ''
    bg = lambda : ''
    attr = lambda : ''
    
# try:
#     import colorama
#     colorama.init()
#     print('using colored output')
#     from colorama import Fore, Back, Style

# except:
#     class color:
#         def __init__(self):
#             self.RED = ''
#             self.BLUE = ''
#             self.BLACK = ''
#             self.DIM = ''
        
# print(Fore.RED + '1.00 2.00 ' + Fore.BLUE + '3.00')
# print(Fore.RED + '1.00 2.00 ' + Fore.BLUE + '3.00')
# print(Fore.BLACK + Style.DIM + '0 0 1')


class SuperPose(UserList, ABC):
    # inherits from:
    #  UserList, gives list-like functionality
    #  ABC, defines an abstract class, can't be instantiated
    
#    @property
#    def length(self):
#        """
#        Property to return number of matrices in pose object
#        :return: int
#        """
#        return len(self._list)
#
#    @property
#    def data(self):
#        """
#        Always returns a list containing the matrices of the pose object.
#        :return: A list of matrices.
#        """
#        return self._list
#
#
#    def is_equal(self, other):
#        if (type(self) is type(other)) and (self.length == other.length):
#            for i in range(self.length):
#                try:
#                    npt.assert_almost_equal(self.data[i], other.data[i])
#                except AssertionError:
#                    return False
#            return True
#
#    def append(self, item):
#        check_args.super_pose_appenditem(self, item)
#        if type(item) is np.matrix:
#            self._list.append(item)
#        else:
#            for each_matrix in item:
#                self._list.append(each_matrix)
#
#    def tr_2_rt(self):
#        assert isinstance(self, pose.SE2) or isinstance(self, pose.SE3)
#
#    def t_2_r(self):
#        assert isinstance(self, pose.SE2) or isinstance(self, pose.SE3)
#        for each_matrix in self:
#            pass  # TODO
    
    def __init__(self):
        # handle common cases
        #  deep copy
        #  numpy array
        #  list of numpy array
        # validity checking??
        print('super_pose constructor')
        super().__init__()   # enable UserList superpowers
        
    def arghandler(self, arg):
        if type(arg) is np.ndarray:
            # it's a numpy array
            print('construct from ndarray', arg)
            assert arg.shape == self.shape, 'array must have valid shape for the class'
            assert type(self).isvalid(arg), 'array must have valid value for the class'
            self.data.append(arg)
        elif type(arg) is list:
            # construct from a list
            s = self.shape
            check = type(self).isvalid
            assert all( map( lambda x: x.shape==s and check(x), arg) ), 'all elements of list must have valid shape and value for the class'
            self.data = arg           
        elif type(self) == type(arg):
            # it's an SO2 type, do copy
            print('copy constructor')
            self.data.append(arg.data.copy())
        else:
            raise ValueError('bad argument to SO2 constructor')
            
    def append(self, x):
        print('in append method')
        if not type(self) == type(x):
            raise ValueError("cant append different type of pose object")
        if len(x) > 1:
            raise ValueError("cant append a pose sequence - use extend")
        super().append(x.A)
        
    @property
    def A(self):
        # get the underlying numpy array
        if len(self.data) == 1:
            return self.data[0]
        else:
            return self.data

    def __getitem__(self, i):
        print('getitem', i)
        #return self.__class__(self.data[i])
        return self.__class__(self.data[i])

    #----------------------- tests
    @property
    def isSO(self):
        return type(self).__name__ == 'SO2' or type(self).__name__ == 'SO3'
    
    @property
    def isSE(self):
        return type(self).__name__ == 'SE2' or type(self).__name__ == 'SE3'
    
    @property
    def isSO(self):
        return type(self).__name__ == 'SO2' or type(self).__name__ == 'SO3'
    
    @property
    def N(self):
        if type(self).__name__ == 'SO2' or type(self).__name__ == 'SE2':
            return 2
        else:
            return 3
        
    # compatibility methods

    def isrot(self):
        return type(self).__name__ == 'SO3'

    def isrot2(self):
        return type(self).__name__ == 'SO2'

    def ishom(self):
        return type(self).__name__ == 'SE3'

    def ishom2(self):
        return type(self).__name__ == 'SE2'
    
    #----------------------- properties
    @property
    def shape(self):
        if   type(self).__name__ == 'SO2':
            return (2,2)
        elif type(self).__name__ == 'SO3':
            return (3,3)
        elif type(self).__name__ == 'SE2':
            return (3,3)
        elif type(self).__name__ == 'SE3':
            return (4,4)
    
    

    
    def about(self):
        print(type(self).__name__)
#
#    def render(self):
#        pass
#
#    def trprint(self):
#        pass  # TODO
#
#    def trplot(self):
#        pass  # TODO
#
#    def trplot2(self):
#        pass  # TODO
#
#    def tranimate(self):
#        pass  # TODO


    #----------------------- arithmetic

    def __mul__(self, other):
        assert type(self) == type(other), 'operands to * are of different types'
        return self._op2(other, lambda x, y: x @ y )

    def __rmul__(x, y):
        raise NotImplemented()
        
    def __imul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, n):
        assert type(n) is int, 'exponent must be an int'
        return self.__class__([np.linalg.matrix_power(x, n) for x in self.data])
    
    def __ipow__(self, n):
        return self.__pow__(n)
                    

    def __truediv__(self, other):
        assert type(self) == type(other), 'operands to * are of different types'
        return self._op2(other, lambda x, y: x @ np.linalg.inv(y) )
    

    def __add__(self, other):
        # results is not in the group, return an array, not a class
        assert type(self) == type(other), 'operands to * are of different types'
        return self._op2(other, lambda x, y: x + y )

    def __sub__(self, other):
        # results is not in the group, return an array, not a class
        # TODO allow class +/- a conformant array
        assert type(self) == type(other), 'operands to * are of different types'
        return self._op2(other, lambda x, y: x - y )
    

    def __eq__(self, other):
        assert type(self) == type(other), 'operands to == are of different types'
        return self._op2(other, lambda x, y: np.allclose(x, y) )
    
    def __ne__(self, other):
        return [not x for x in self == other]
    
    def _op2(self, other, op):
        
        if len(self) == 1:
            if len(other) == 1:
                return op(self.A, other.A)
            else:
                print('== 1xN')
                return [op(self.A @ x.A) for x in other]
        else:
            if len(other) == 1:
                print('== Nx1')
                return [op(x.A @ other.A) for x in self]
            elif len(self) == len(other):
                print('== NxN')
                return [op(x.A @ y.A) for (x,y) in zip(self.A, self.other)]
            else:
                raise ValueError('length of lists to == must be same length')
    
    # @classmethod
    # def rand(cls):
    #     obj = cls(uniform(0, 360), unit='deg')
    #     return obj
                
     #----------------------- functions
                
    def exp(self, arg):
        pass
    
    def log(self, arg):
        pass
    
    def interp(self, arg):
        pass
                
    #----------------------- i/o stuff
    
    def print(self):
        if self.N == 2:
            trprint2(self.A)
        else:
            trprint(self.A)

    def plot(self):
        if self.N == 2:
            trplot2(self.A)
        else:
            trplot(self.A)
        
    def __repr__(self):
        #print('in __repr__')
        if len(self) >= 1:
            str = ''
            for each in self.data:
                str += np.array2string(each) + '\n\n'
            return str.rstrip("\n")  # Remove trailing newline character
        else:
             raise ValueError('no elements in the value list')

    def __str__(self):
        #print('in __str__')
        def mformat(self, X):
            # X is an ndarray value to be display
            # self provides set type for formatting
            print(self.A)
            out = ''
            n = self.N  # dimension of rotation submatrix
            for rownum, row in enumerate(X):
                rowstr = '  '
                # format the columns
                for colnum, element in enumerate(row):
                    s = '{:< 10g}'.format(element)

                    if rownum < n:
                        if colnum < n:
                            # rotation part
                            s = fg('red') + bg('grey_93') + s + attr(0)
                        else:
                            # translation part
                            s = fg('blue') + bg('grey_93') + s + attr(0)
                    else:
                        # bottom row
                        s = fg('grey_50') + bg('grey_93') + s + attr(0)
                    rowstr += s
                out += rowstr + bg('grey_93') + '  ' + attr(0) + '\n'
            return out

        output_str = ''

        if len(self.data) == 1:
            # single matrix case
            output_str = mformat(self, self.A)
        elif len(self.data) > 1:
            # sequence case
            for count, X in enumerate(self.data):
                # add separator lines and the index
                output_str += fg('green') + '[{:d}] =\n'.format(count) + attr(0) + mformat(self, X)
        else:
            raise ValueError('no elements in the value list')


        
        return output_str
