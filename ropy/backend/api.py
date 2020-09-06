# import zerorpc
# import ropy as rp
# # import numpy as np


# panda = rp.PandaMDH()
# panda.q = panda.qr

# # q = [0, -0.3, 0, -2.2, 0, 2.0, np.pi/4]
# q = panda.qr.tolist()

# T = panda.fkine(q)
# # print(T)

# # q1, success, err = panda.ikine(T)
# # q1 = q1.tolist()
# # T2 = panda.fkine(q1)
# # print(T * T2)

# q1, err, success = panda.ikcon(T)
# q1 = q1.tolist()
# T2 = panda.fkine(q1)
# print(T2)

# # q2 = [-0.0011,    0.9013,    0.0210,   -0.0698,   -0.0199,    0.9620,    0.7861]
# # T2 = panda.fkine(q2)
# # print(T2)



# # q = [0, 0, 0, -1, 0, 0, 0]
# # q = panda.q.tolist()

# l = []
# for i in range(panda.n):
#     # m = min(0, i-1)
#     # l.append(panda.A([m, i]).A.tolist())
#     # lib = panda.A(i).A
#     # lie = panda.links[i].A(panda.q[i]).A
#     # l.append(lib.tolist())
#     # l.append(lie.tolist())
#     li = [panda.links[i].sigma, panda.links[i].mdh, panda.links[i].theta, panda.links[i].d, panda.links[i].a, panda.links[i].alpha]
#     l.append(li)


# # l[3][3] = 0.1

# # l1 = panda.A([0, 1])



# sim = zerorpc.Client()
# sim.connect("tcp://127.0.0.1:4242")

# ob = ["SerialLinkMDH", l]
# id = sim.robot(ob)

# q_ob = [id, q1]
# sim.q(q_ob)

# # qd = [0, 0, 0, 0.1, 0, 0, 0]
# # qd_ob = [id, qd]
# # sim.qd(qd_ob)


# # v = np.array([[0.01, 0.01, 0.01, 0, 0, 0]]).T

# # for i in range(1000):
# #     panda.q = sim.get_q(id)
# #     qd = np.linalg.pinv(panda.jacobe()) @ v
# #     qd = qd.tolist()
# #     sim.qd([id, qd])

# #     sim.step(1)



# # import sys
# # import zerorpc
# # import numpy as np
# # from subprocess import call, Popen


# # class RopyApi(object):

# #     def __init__(self):
# #         self.c = zerorpc.Client()
# #         self.c.connect("tcp://127.0.0.1:4242")
# #         arr = np.eye(4).tolist()
# #         a = self.c.hello(arr)
# #         print(a)

# #     def draw_robot(self, poses):
# #         """based on the input text, return the int result"""
# #         self.c.draw_robot(pose)


# # if __name__ == '__main__':
# #     RopyApi()