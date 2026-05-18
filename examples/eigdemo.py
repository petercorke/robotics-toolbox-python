"""
Eigenvalue demonstration

The revolving red hand is the input vector and the blue hand is the linearly
transformed vector.

Four times every revolution the two hands are parallel (or anti-parallel),
twice to each eigenvector of the matrix A.  The ratio of lengths, blue hand
over red hand, is the corresponding eigenvalue.  The eigenvalue will be
negative if the hands are anti-parallel.

"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi, sin, cos


def eigdemo(A):

    print('matrix A =')
    print(A)
    e, x = np.linalg.eig(A)

    print(f"λ1 = {e[0]:.3f}, x1 = {np.real(x[:,0].flatten())}")
    print(f"λ2 = {e[1]:.3f}, x2 = {np.real(x[:,1].flatten())}")

    s = np.max(np.abs(e))

    fig, ax = plt.subplots()
    plt.axis([-s, s, -s, s])
    plt.grid(True)
    plt.title('Eigenvector demonstration')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.xlim(-s, s)
    plt.ylim(-s, s)
    ax.set_aspect('equal')

    l1, = plt.plot([0, 0], [0, 0], color='r', linewidth=3)  # input vector
    l2, = plt.plot([0, 0], [0, 0], color='b', linewidth=3)  # transformed vector

    plt.plot([0, x[0, 0]], [0, x[1, 0]], color='k', linewidth=1)
    plt.plot([0, x[0, 1]], [0, x[1, 1]], color='k', linewidth=1)
    

    plt.legend(['$x$', r'${\bf A} x$', r'$x_1$', r'$x_2$'])

    print("\nto exit: type q, or close the window")

    def animate(theta):

        x = np.r_[cos(-theta), sin(-theta)]
        y = A @ x

        l1.set_xdata([0, x[0]])
        l1.set_ydata([0, x[1]])

        l2.set_xdata([0, y[0]])
        l2.set_ydata([0, y[1]])

        return l1, l2


    myAnimation = animation.FuncAnimation(
        fig, animate, frames=np.linspace(
            0, 2 * pi, 400), blit=True, interval=20, repeat=True)

    plt.show(block=True)

def main():

    def help():
        print("eigdemo          # uses default matrix [1 2; 3 4]")
        print("eigdemo a b c d  # uses matrix [a b; c d]")
        sys.exit(0)
    
    if sys.argv in ("-h", "--help", "help"):
        help()

    if len(sys.argv) == 5:

        try:
            vec = [float(a) for a in sys.argv[1:5]]
            A = np.reshape(vec, (2,2))
        except:
            help()

    elif len(sys.argv) == 1:
        A = np.array([
            [1, 2],
            [3, 3]
        ])
    
    else:
        help()


    eigdemo(A)

if __name__ == "__main__":
    main()