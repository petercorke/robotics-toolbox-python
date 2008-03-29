import matplotlib.axes3d as p3
from numpy import * # for outer and arange
import pylab as p   # for figure
from robot.Quaternion import *

def trplot(r, name=''):
    '''
    Plot a rotation as a set of axes aligned with the x, y and z
    directions of a frame rotated by C{r}.
    '''

    if type(r) is matrix:
        q = quaternion(r);
    elif isinstance(r, quaternion):
        q = r;
    else:
        raise ValueError;

    x = q * mat([1,0,0]);
    y = q * mat([0,1,0]);
    z = q * mat([0,0,1]);

    fig=p.figure()
    ax=p3.Axes3D(fig)
    ax.plot3d([0,x[0,0]], [0,x[0,1]], [0,x[0,2]], color='red')
    ax.plot3d([0,y[0,0]], [0,y[0,1]], [0,y[0,2]], color='green')
    ax.plot3d([0,z[0,0]], [0,z[0,1]], [0,z[0,2]], color='blue')
    p.show()

if __name__ == "__main__":
    from robot.transform import *
    trplot( rotx(0.2) );
