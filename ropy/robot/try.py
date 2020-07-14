
import ropy as rp
import spatialmath as sm
import numpy as np

l1 = rp.Link(alpha=1.0)
l2 = rp.Link(alpha=2.0)
l3 = rp.Link(alpha=3.0)
l4 = rp.Link(alpha=4.0)

r1 = rp.SerialLink([l1, l2])
r2 = rp.SerialLink([l1, l2])

links = [l1, r1]

r3 = rp.SerialLink(links)
# print(l1)
repr(l1)

print(r3.links)
r4 = r3 + l1
r5 = l1 + r3


print(r4.links)
print(r5.links)
print(r3.links)

q = np.pi * np.ones((4, 1))
print(r4.islimit(q))


ls = [l1, l2, l3, l4]

print(r1.isspherical())
