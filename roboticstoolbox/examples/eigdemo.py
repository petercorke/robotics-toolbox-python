"""
Eigenvalue demonstration

The revolving red hand is the input vector and the blue hand is the linearly
transformed vector.

Four times every revolution the two hands are parallel (or anti-parallel),
twice to each eigenvector of the matrix A.  The ratio of lengths, blue hand
over red hand, is the corresponding eigenvalue.  The eigenvalue will be
negative if the hands are anti-parallel.

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi, sin, cos

A = np.array([
    [1, 2],
    [3, 3]
])

e, x = np.linalg.eig(A)
print(e)
print(x)

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

l1, = plt.plot([0, 0], [0, 0], color='r', linewidth=1.5)  # input vector
l2, = plt.plot([0, 0], [0, 0], color='b', linewidth=1.5)  # transformed vector

plt.legend(['$x$', r'${\bf A} x$'])


def animate(theta):

    x = np.r_[cos(theta), sin(theta)]
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
