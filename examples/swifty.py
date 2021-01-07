#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
import spatialmath as sm
import numpy as np

# Launch the simulator Swift
# env = rtb.backends.Swift()
# env.launch()

# # Create a Panda robot object
# robot = rtb.models.Puma560()
# env.add(robot)

# env.hold()

# print(panda)
# print(panda.base_link)
# print(panda.ee_links)

# path, n = panda.get_path(panda.base_link, panda.ee_links[0])

# q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
# panda.q = q1

# print(panda.fkine())

# for link in path:
#     print(link.name)

# print(panda.get_path(panda.base_link, panda.ee_links[0])[0])

# print(panda.links[5].A(0))

# # Set joint angles to ready configuration
# panda.q = panda.qr

# Add the Panda to the simulator
# env.add(panda)


# while 1:
#     pass

# panda = rtb.models.Panda()
# robot = rtb.models.DH.Panda()

# panda.q = panda.qr

# T = panda.fkine(panda.qr)
# # T = sm.SE3(0.8, 0.2, 0.1) * sm.SE3.OA([0, 1, 0], [0, 0, -1])

# sol = robot.ikine_LMS(T)         # solve IK


# qt = rtb.jtraj(robot.qz, sol.q, 50)

# env = rtb.backends.Swift()  # instantiate 3D browser-based visualizer
# env.launch()                # activate it
# env.add(panda)              # add robot to the 3D scene
# for qk in qt.q:             # for each joint configuration on trajectory
#     panda.q = qk          # update the robot state
#     env.step()            # update visualization

# print(panda.fkine(panda.q))
# print(T)

# r = rtb.models.Puma560()
# # r.q = [-0.5, -0.5, 0.5, 0.5, 0.5, 0.5]
# env = rtb.backends.Swift()
# env.launch()
# env.add(r)
# env.hold()

r = rtb.models.ETS.Panda()
r.q = r.qr

q1 = r.qr
q2 = q1 + 0.1


def derivative(f, a, method='central', h=0.01):
    '''Compute the difference formula for f'(a) with step size h.

    Parameters
    ----------
    f : function
        Vectorized function of one variable
    a : number
        Compute derivative at x = a
    method : string
        Difference formula: 'forward', 'backward' or 'central'
    h : number
        Step size in difference formula

    Returns
    -------
    float
        Difference formula:
            central: f(a+h) - f(a-h))/2h
            forward: f(a+h) - f(a))/h
            backward: f(a) - f(a-h))/h
    '''
    if method == 'central':
        return (f(a + h) - f(a - h))/(2*h)
    elif method == 'forward':
        return (f(a + h) - f(a))/h
    elif method == 'backward':
        return (f(a) - f(a - h))/h
    else:
        raise ValueError("Method must be 'central', 'forward' or 'backward'.")


# Numerical hessian wrt q[0]
# d = derivative(lambda x: r.jacob0(np.r_[x, q1[1:]]), q1[0], h=0.1)
# print(np.round(h1[:, :, 0], 3))
# print(np.round(d, 3))

# Numerical third wrt q[0]
d = derivative(lambda x: r.hessian0(np.r_[x, q1[1:]]), q1[0], h=0.01)
print(np.round(d[3:, :, 0], 3))
print(np.round(r.third(q1)[3:, :, 0, 0], 3))

l = r.deriv(q1, 3)
print(np.round(l[3:, :, 0, 0], 3))

# def runner():
#     for i in range(10000):
#         r.jacob0(r.q)
#         # r.hessian0(r.q)


# import cProfile
# cProfile.run('runner()')
