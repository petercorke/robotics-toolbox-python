#DIFF2TR Convert a differential to a homogeneous transform
from numpy import *
def diff2tr(d):
    d = mat(d).flatten().T
    return mat([[0,       -d[5,0],   d[4,0],  d[0,0]],\
                [d[5,0],        0,  -d[3,0],  d[1,0]],\
                [-d[4,0],  d[3,0],        0,  d[2,0]],\
                [0,             0,        0,      0]])
