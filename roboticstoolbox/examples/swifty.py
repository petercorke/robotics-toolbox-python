# #!/usr/bin/env python
# """
# @author Jesse Haviland
# """

# import roboticstoolbox as rtb
# import spatialmath as sm
# import numpy as np
# import numpy.testing as nt

# import matplotlib
# import matplotlib.pyplot as plt

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# plt.style.use('ggplot')
# matplotlib.rcParams['font.size'] = 4.5
# matplotlib.rcParams['lines.linewidth'] = 0.5
# matplotlib.rcParams['xtick.major.size'] = 1.5
# matplotlib.rcParams['ytick.major.size'] = 1.5
# matplotlib.rcParams['axes.labelpad'] = 1
# plt.rc('grid', linestyle="-", color='#dbdbdb')

# # Launch the simulator Swift
# # env = rtb.backends.Swift()
# # env.launch()

# # # Create a Panda robot object
# # robot = rtb.models.Puma560()
# # env.add(robot)

# # env.hold()

# # print(panda)
# # print(panda.base_link)
# # print(panda.ee_links)

# # path, n = panda.get_path(panda.base_link, panda.ee_links[0])

# # q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
# # panda.q = q1

# # print(panda.fkine())

# # for link in path:
# #     print(link.name)

# # print(panda.get_path(panda.base_link, panda.ee_links[0])[0])

# # print(panda.links[5].A(0))

# # # Set joint angles to ready configuration
# # panda.q = panda.qr

# # Add the Panda to the simulator
# # env.add(panda)


# # while 1:
# #     pass

# # panda = rtb.models.Panda()
# # robot = rtb.models.DH.Panda()

# # panda.q = panda.qr

# # T = panda.fkine(panda.qr)
# # # T = sm.SE3(0.8, 0.2, 0.1) * sm.SE3.OA([0, 1, 0], [0, 0, -1])

# # sol = robot.ikine_LMS(T)         # solve IK


# # qt = rtb.jtraj(robot.qz, sol.q, 50)

# # env = rtb.backends.Swift()  # instantiate 3D browser-based visualizer
# # env.launch()                # activate it
# # env.add(panda)              # add robot to the 3D scene
# # for qk in qt.q:             # for each joint configuration on trajectory
# #     panda.q = qk          # update the robot state
# #     env.step()            # update visualization

# # print(panda.fkine(panda.q))
# # print(T)

# # r = rtb.models.Puma560()
# # # r.q = [-0.5, -0.5, 0.5, 0.5, 0.5, 0.5]
# # env = rtb.backends.Swift()
# # env.launch()
# # env.add(r)
# # env.hold()

# # r = rtb.models.ETS.Panda()
# # r.q = r.qr

# # q2 = [1, 2, -1, -2, 1, 1, 2]
# # Tep = r.fkine(q2)
# # Tep = sm.SE3(0.7, 0.2, 0.1) * sm.SE3.OA([0, 1, 0], [0, 0, -1])

# # import cProfile
# # cProfile.run('qp = r.ikine_mmc(Tep)')

# # print(r.fkine(q2))
# # print(r.fkine(qp))

# # q1 = r.qr
# # q2 = q1 + 0.1


# # def derivative(f, a, method='central', h=0.01):
# #     '''Compute the difference formula for f'(a) with step size h.

# #     Parameters
# #     ----------
# #     f : function
# #         Vectorized function of one variable
# #     a : number
# #         Compute derivative at x = a
# #     method : string
# #         Difference formula: 'forward', 'backward' or 'central'
# #     h : number
# #         Step size in difference formula

# #     Returns
# #     -------
# #     float
# #         Difference formula:
# #             central: f(a+h) - f(a-h))/2h
# #             forward: f(a+h) - f(a))/h
# #             backward: f(a) - f(a-h))/h
# #     '''
# #     if method == 'central':
# #         return (f(a + h) - f(a - h))/(2*h)
# #     elif method == 'forward':
# #         return (f(a + h) - f(a))/h
# #     elif method == 'backward':
# #         return (f(a) - f(a - h))/h
# #     else:
# #         raise ValueError("Method must be 'central', 'forward' or 'backward'.")

# # # Numerical hessian wrt q[0]
# # # d = derivative(lambda x: r.jacob0(np.r_[x, q1[1:]]), q1[0], h=0.1)
# # # print(np.round(h1[:, :, 0], 3))
# # # print(np.round(d, 3))

# # # Numerical third wrt q[0]
# # # d = derivative(lambda x: r.hessian0(np.r_[x, q1[1:]]), q1[0], h=0.01)
# # # print(np.round(d[:, :, 0], 3))
# # # print(np.round(r.third(q1)[:, :, 0, 0], 3))

# # # l = r.partial_fkine0(q1, 3)
# # # print(np.round(l[:, :, 0, 0], 3))


# # # Numerical fourth wrt q[0]
# # # d = derivative(lambda x: r.third(np.r_[x, q1[1:]]), q1[0], h=0.01)
# # # print(np.round(d[:, :, 0, 0], 3))

# # # l = r.partial_fkine0(q1, 4)
# # # print(np.round(l[:, :, 0, 0, 0], 3))

# # j = r.jacob0(r.q)
# # def runner():
# #     for i in range(1):
# #         r.partial_fkine0(r.q, 7)
# #         # r.hessian0(r.q, j)


# # import cProfile
# # cProfile.run('runner()')


# fig, ax = plt.subplots()
# fig.set_size_inches(8, 8)

# ax.set(xlabel='Manipulability', ylabel='Condition')
# ax.grid()
# plt.grid(True)
# # ax.set_xlim(xmin=0, xmax=3.1)
# # ax.set_ylim(ymin=0, ymax=0.11)

# plt.ion()
# plt.show()

# rng = np.random.default_rng(1)
# r = rtb.models.ETS.GenericSeven()


# def randn(a, b, size):
#     return (b - a) * rng.random(size) + a


# def rand_q():
#     return randn(np.pi / 2, -np.pi / 2, (7,))


# def rand_v():
#     v = randn(0, 1, (3,))
#     v = v / np.linalg.norm(v)

#     return np.r_[v, 0, 0, 0]
#     # return np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0])


# x = []
# y = []
# rands = 100
# v = np.zeros((rands, 6))

# for i in range(rands):
#     v[i, :] = rand_v()

# for i in range(10000000):

#     q = rand_q()
#     # v = rand_v()

#     J = r.jacob0(q)
#     Jt = J[:3, :]
#     # H = r.hessian0(q)
#     # Ht = H[:3, :, :]

#     q_n = [10000, 0]
#     q_m = [10000, 0]
#     q_mn = 0

#     # cond = np.linalg.cond(J[:3, :])
#     m = r.manipulability(J=J, axes='trans')
#     # infn = np.linalg.norm(Jt, 2)
#     psi = (np.cbrt(np.linalg.det(Jt @ np.transpose(Jt)))) / \
#         (np.trace(Jt @ np.transpose(Jt)) / 3)

#     for j in range(rands):
#         qd = np.linalg.pinv(J) @ v[j, :]

#         if np.max(qd) > q_m[1]:
#             q_m[1] = np.max(qd)

#         if np.min(qd) < q_m[0]:
#             q_m[0] = np.min(qd)

#         if np.linalg.norm(qd) > q_n[1]:
#             q_n[1] = np.linalg.norm(qd)
#         elif np.linalg.norm(qd) < q_n[0]:
#             q_n[0] = np.linalg.norm(qd)

#         q_mn += np.linalg.norm(qd)

#     q_mn /= rands

#     # # ax.plot(m, np.log10(cond), 'o', color='black')
#     # ax.plot(m, infn, 'o', color='black')
#     # # ax.plot(1, 0.002, 'o', color='black')

#     x.append(m)
#     y.append(np.log10(q_n[0]))
#     # y.append(psi)

#     # ax.plot(m, np.log10(q_m[1]), 'o', color='black')

#     if len(x) % 100 == 0:
#         ax.hist2d(x, y, bins=100)
#         plt.pause(0.001)


# plt.ioff()
# plt.show()
