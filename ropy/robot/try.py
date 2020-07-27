
import ropy as rp
import spatialmath as sm
import numpy as np
import ctypes
import frne

import time

# puma = rp.Puma560()

# L = np.zeros(24*9)

# for i in range(puma.n):
#     j = i * 24
#     L[j] = puma.links[i].alpha
#     L[j+1] = puma.links[i].a
#     L[j+2] = puma.links[i].theta
#     L[j+3] = puma.links[i].d
#     L[j+4] = puma.links[i].sigma
#     L[j+5] = puma.links[i].offset
#     L[j+6] = puma.links[i].m
#     L[j+7:j+10] = puma.links[i].r.flatten()
#     L[j+10:j+19] = puma.links[i].I.flatten()
#     L[j+19] = puma.links[i].Jm
#     L[j+20] = puma.links[i].G
#     L[j+21] = puma.links[i].B
#     L[j+22:j+24] = puma.links[i].Tc.flatten()


# r = frne.init(puma.n, puma.mdh, L, puma.gravity[:, 0])
# frne.point(r)
# frne.delete(r)
# frne.point(r)
# t1 = time.time()

# z = np.zeros(6)
# o = np.ones(6)


# tau = puma.rne(o, o, puma.qn)
# print(tau)




puma = rp.Puma560()
puma.q = puma.qn
qd = [1, 2, 3, 1, 2, 3]


print(np.round(puma.coriolis(qd, puma.qn), 5))

# z = np.zeros(6)
# o = np.ones(6)

# t0 = puma.rne(z, z, puma.qn)
# t1 = puma.rne(z, o, puma.qn)

# print(puma._rne_changed)
# puma.gravity = [0, 0, 9.81]
# print(puma._rne_changed)
# t2 = puma.rne(z, o, puma.qn)


# t2 = time.time()
# puma.rne(1, 2, 3, 4, 5)
# t3 = time.time()
# puma.rne(1, 2, 3, 4, 5)
# t4 = time.time()
# puma.delete_rne()
# t5 = time.time()
# puma.rne(1, 2, 3, 4, 5)
# t6 = time.time()

# print(t2-t1)
# print(t3-t2)
# print(t4-t3)
# print(t6-t5)

# print(r)

# panda = rp.PandaMDH()
# q = np.array([0, -0.3, 0, -2.2, 0, 2.0, np.pi/4])
# T = panda.fkine(q)

# qe, success, err = panda.ikine(T)

# print(qe)
# print(success)
# print(err)



# l0 = rp.Revolute(d=2.0)
# l1 = rp.Prismatic(theta=1.0)
# r0 = rp.SerialLink([l0, l1])

# qa5, success, err = r0.ikine(T, mask=[1, 1, 0, 0, 0, 0])

# print(success)
# print(err)


# l0 = rp.Revolute(alpha=np.pi/2)
# l1 = rp.Revolute(a=0.4318)
# l2 = rp.Revolute(d=0.15005, a=0.0203, alpha=-np.pi/2)
# r0 = rp.SerialLink([l0, l1, l2])
# q = [1, 1, 1]
# T = r0.fkine(q)

# qr = r0.ikine3(T)
# print(qr)

# puma = rp.Puma560()
# q = puma.qr
# T = puma.fkine(q)

# # puma.ikine6s(T)

# qr = puma.ikunc(T)
# print(qr)


# # rrp
# l0 = rp.Revolute(alpha=-np.pi/2)
# l1a = rp.Revolute(alpha=np.pi/2)
# l2a = rp.Prismatic()
# l3 = rp.Revolute(alpha=-np.pi/2)
# l4 = rp.Revolute(alpha=np.pi/2)
# l5 = rp.Revolute()
# rrp = rp.SerialLink([l0, l1a, l2a, l3, l4, l5])

# q = [1, 1, 1, 1, 1, 1]
# T = rrp.fkine(q)

# rrp.ikine6s(T)

# # simple
# l1b = rp.Revolute()
# l2b = rp.Revolute(alpha=np.pi/2)
# sim = rp.SerialLink([l0, l1b, l2b, l3, l4, l5])

# q = [1, 1, 1, 1, 1, 1]
# T = sim.fkine(q)
# sim.ikine6s(T)

# # offset
# l1c = rp.Revolute(d=1.0)
# off = rp.SerialLink([l0, l1c, l2b, l3, l4, l5])

# q = [1, 1, 1, 1, 1, 1]
# T = off.fkine(q)
# off.ikine6s(T)
