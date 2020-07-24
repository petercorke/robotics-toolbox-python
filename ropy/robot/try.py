
import ropy as rp
import spatialmath as sm
import numpy as np

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

puma = rp.Puma560()
q = puma.qr
T = puma.fkine(q)

# puma.ikine6s(T)

qr = puma.ikunc(T)
print(qr)


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
