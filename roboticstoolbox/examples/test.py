import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

# q0 = np.array(
#     [
#         -1.66441371,
#         -1.20998727,
#         1.04248366,
#         -2.10222463,
#         1.05097407,
#         1.41173279,
#         0.0053529,
#     ]
# )

# tol = 1e-6

# panda = rtb.models.Panda().ets()

# Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

# solver = rtb.IK_QP()

# sol = panda.ik_LM(Tep, tol=tol, q0=q0, method="chan")

robot = rtb.models.KinovaGen3()

glink = rtb.Link(
    rtb.ET.tz(0.12),
    name="gripper_link",
    m=0.831,
    r=[0, 0, 0.0473],
    parent=robot.links[-1],
)

robot.links.append(glink)


n = len(robot.links)

print(n)

# allocate intermediate variables
Xup = sm.SE3.Alloc(n)
Xtree = sm.SE3.Alloc(n)

# The body velocities, accelerations and forces
v = sm.SpatialVelocity.Alloc(n)
a = sm.SpatialAcceleration.Alloc(n)
f = sm.SpatialForce.Alloc(n)

# The joint inertia
I = sm.SpatialInertia.Alloc(n)  # noqa: E741

# Joint motion subspace matrix
s = []

q = robot.qr
qd = np.zeros(n)
qdd = np.zeros(n)

Q = np.zeros(n)

# for link in robot.links:
#     print()
#     print(link.name)
#     print(link.isjoint)
#     print(link.m)

# for link in robot.grippers[0].links:
#     print()
#     print(link.name)
#     print(link.isjoint)
#     print(link.m)

for i, link in enumerate(robot.links):

    # Allocate the inertia
    I[i] = sm.SpatialInertia(link.m, link.r, link.I)

    # Compute the link transform
    Xtree[i] = sm.SE3(link.Ts, check=False)  # type: ignore

    # Append the variable axis of the joint to s
    if link.v is not None:
        s.append(link.v.s)
    else:
        s.append(np.zeros(6))

a_grav = -sm.SpatialAcceleration([0, 0, 9.81])

# Forward recursion
for i, link in enumerate(robot.links):

    if link.jindex is None:
        qi = 0
        qdi = 0
        qddi = 0
    else:
        qi = q[link.jindex]
        qdi = qd[link.jindex]
        qddi = qdd[link.jindex]

    vJ = sm.SpatialVelocity(s[i] * qdi)

    # Transform from parent(j) to j
    if link.isjoint:
        Xup[i] = sm.SE3(link.A(qi)).inv()
    else:
        Xup[i] = sm.SE3(link.A()).inv()

    if link.parent is None:
        v[i] = vJ
        a[i] = Xup[i] * a_grav + sm.SpatialAcceleration(s[i] * qddi)
    else:
        jp = link.parent.jindex  # type: ignore
        v[i] = Xup[i] * v[i - 1] + vJ
        a[i] = Xup[i] * a[i - 1] + sm.SpatialAcceleration(s[i] * qddi) + v[i] @ vJ

    f[i] = I[i] * a[i] + v[i] @ (I[i] * v[i])

# backward recursion
for i in reversed(range(n)):

    link = robot.links[i]

    Q[i] = sum(f[i].A * s[i])

    if link.parent is not None:
        f[i - 1] = f[i - 1] + Xup[i] * f[i]

print(np.round(Q, 2))


for link in robot.links:
    print()
    print(link.name)
    print(link.isjoint)
    print(link.m)
    print(link.r)
    print(link.I)
