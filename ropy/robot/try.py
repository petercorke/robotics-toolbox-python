
import ropy as rp
import spatialmath as sm
import numpy as np

# t = sm.SE3([np.eye(4) for i in range(5)])

# t. t[1] * sm.SE3.Rx(2)

# print(t)



# l1 = rp.Link(alpha=1.0)
# l2 = rp.Link(alpha=2.0)
# l3 = rp.Link(alpha=3.0)
# l4 = rp.Link(alpha=4.0)

# r1 = rp.SerialLink([l1, l2])
# r2 = rp.SerialLink([l1, l2])

# links = [l1, r1]

# r3 = rp.SerialLink(links)
# # print(l1)
# repr(l1)

# print(r3.links)
# r4 = r3 + l1
# r5 = l1 + r3


# print(r4.links)
# print(r5.links)
# print(r3.links)

# q = np.pi * np.ones((4, 1))
# print(r4.islimit(q))


# ls = [l1, l2, l3, l4]

# print(r1.isspherical())




l0 = rp.Prismatic()
l1 = rp.Revolute()
l2 = rp.Prismatic(theta=2.0)
l3 = rp.Revolute()

q = np.array([[1], [2], [3], [4]])
qq = np.c_[q, q]

r0 = rp.SerialLink([l0, l1, l2, l3])

print(r0.fkine(q).A)
