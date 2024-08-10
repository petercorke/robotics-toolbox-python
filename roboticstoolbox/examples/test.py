import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm


def rne(robot, q, qd, qdd):
    n = len(robot.links)

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
            v[i] = Xup[i] * v[i - 1] + vJ
            a[i] = Xup[i] * a[i - 1] + sm.SpatialAcceleration(s[i] * qddi) + v[i] @ vJ

        f[i] = I[i] * a[i] + v[i] @ (I[i] * v[i])

    # backward recursion
    for i in reversed(range(n)):

        link = robot.links[i]

        Q[i] = sum(f[i].A * s[i])

        if link.parent is not None:
            f[i - 1] = f[i - 1] + Xup[i] * f[i]

    return Q


def rne_eff(robot, q, qd, qdd):
    n = len(robot.links)

    # allocate intermediate variables
    Xup = np.empty((n, 4, 4))

    Xtree = np.empty((n, 4, 4))

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

    for i, link in enumerate(robot.links):

        # Allocate the inertia
        I[i] = sm.SpatialInertia(link.m, link.r, link.I)

        # Compute the link transform
        Xtree[i] = link.Ts

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
            Xup[i] = np.linalg.inv(link.A(qi))
        else:
            Xup[i] = np.linalg.inv(link.A())

        if link.parent is None:
            v[i] = vJ
            a[i] = Xup[i] * a_grav + sm.SpatialAcceleration(s[i] * qddi)
        else:
            v[i] = Xup[i] * v[i - 1] + vJ
            a[i] = Xup[i] * a[i - 1] + sm.SpatialAcceleration(s[i] * qddi) + v[i] @ vJ

        f[i] = I[i] * a[i] + v[i] @ (I[i] * v[i])

    # backward recursion
    for i in reversed(range(n)):

        link = robot.links[i]

        Q[i] = sum(f[i].A * s[i])

        if link.parent is not None:
            f[i - 1] = f[i - 1] + Xup[i] * f[i]

    return Q


robot = rtb.models.KinovaGen3()

glink = rtb.Link(
    rtb.ET.tz(0.12),
    name="gripper_link",
    m=0.831,
    r=[0, 0, 0.0473],
    parent=robot.links[-1],
)

robot.links.append(glink)

# for link in robot.links:
#     print()
#     print(link.name)
#     print(link.isjoint)
#     print(link.m)
#     print(link.r)
#     print(link.I)

# q = rne(robot, robot.qr, np.zeros(7), np.zeros(7))
# q = robot.qr
# qd = np.zeros(7)
# qdd = np.zeros(7)

# for i in range(1000):
#     tau = robot.rne(q, qd, qdd)
#     # q = rne(robot, robot.qr, np.zeros(7), np.zeros(7))
#     # print(i)


# print(np.round(tau, 2))

# [ 0.    0.   14.2   0.12 -9.22  0.   -4.47 -0.    0.    0.    0.  ]

l1a = rtb.Link(ets=rtb.ETS(rtb.ET.Rx()), m=1, r=[0.5, 0, 0], name="l1")
l2a = rtb.Link(ets=rtb.ETS(rtb.ET.Rz()), m=1, r=[0, 0.5, 0], parent=l1a, name="l2")
l3a = rtb.Link(ets=rtb.ETS(rtb.ET.Ry()), m=1, r=[0.0, 0, 0.5], parent=l2a, name="l3")
robota = rtb.Robot([l1a, l2a, l3a], name="simple 3 link a")

l1b = rtb.Link(ets=rtb.ETS(rtb.ET.Rx()), m=1, r=[0.5, 0, 0], name="l1")
l2b = rtb.Link(ets=rtb.ETS(rtb.ET.Rz(0.5)), m=1, r=[0, 0.5, 0], parent=l1b, name="l2")
l3b = rtb.Link(ets=rtb.ETS(rtb.ET.Ry()), m=1, r=[0.0, 0, 0.5], parent=l2b, name="l3")
robotb = rtb.Robot([l1b, l2b, l3b], name="simple 3 link b")

l1c = rtb.Link(ets=rtb.ETS(rtb.ET.tx(0.5)), m=1, r=[0.5, 0, 0], name="l1")
l2c = rtb.Link(ets=rtb.ETS(rtb.ET.Rx()), m=1, r=[0, 0.5, 0], parent=l1c, name="l2")

# Branch 1
l3c = rtb.Link(ets=rtb.ETS(rtb.ET.tz(0.5)), m=1, r=[0.0, 0, 0.5], parent=l2c, name="l3")
l4c = rtb.Link(ets=rtb.ETS(rtb.ET.Rz()), m=1, r=[0.0, 0, 0.5], parent=l3c, name="l4")

# Branch 2
l5c = rtb.Link(ets=rtb.ETS(rtb.ET.tz(0.5)), m=1, r=[0.0, 0, 0.5], parent=l2c, name="l5")
l6c = rtb.Link(ets=rtb.ETS(rtb.ET.Rz()), m=1, r=[0.0, 0, 0.5], parent=l5c, name="l6")

robotc = rtb.Robot([l1c, l2c, l3c, l4c, l5c, l6c], name="branch 3 link c")

za = np.array([0.5, 0.5, 0.5])
zb = np.array([0.5, 0.5])
zc = np.array([0.5, 0.5, 0.5])

print("\nRobot A")
taua = robota.rne(za, za, za)
print(taua)

print("\nRobot B")
taub = robotb.rne(zb, zb, zb)
print(taub)

print("\nRobot C")
tauc = robotc.rne(zc, zc, zc)
print(tauc)


# l1 = rtb.Link(ets=rtb.ETS(rtb.ET.Ry()), m=1, r=[0.5, 0, 0], name="l1")
# l2 = rtb.Link(ets=rtb.ETS(rtb.ET.tx(1)), m=1, r=[0.5, 0, 0], parent=l1, name="l2")
# l3 = rtb.Link(ets=rtb.ETS([rtb.ET.Ry()]), m=0, r=[0.0, 0, 0], parent=l2, name="l3")
# robot = rtb.Robot([l1, l2, l3], name="simple 3 link")
# z = np.zeros(robot.n)

# # check gravity load
# tau = robot.rne(z, z, z) / 9.81
# print(tau)

# # nt.assert_array_almost_equal(tau, np.r_[-2, -0.5])

# print("\n\nNew Robot\n")
# l1 = rtb.Link(ets=rtb.ETS(rtb.ET.Ry()), m=1, r=[0.5, 0, 0], name="l1")
# l2 = rtb.Link(
#     ets=rtb.ETS([rtb.ET.tx(1), rtb.ET.Ry()]), m=1, r=[0.5, 0, 0], parent=l1, name="l2"
# )
# robot = rtb.Robot([l1, l2], name="simple 2 link")
# z = np.zeros(robot.n)

# # check gravity load
# tau = robot.rne(z, z, z) / 9.81
# print(tau)
