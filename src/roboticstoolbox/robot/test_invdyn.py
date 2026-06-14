import numpy as np
from spatialmath import Twist3, SE3
import roboticstoolbox as rtb
from roboticstoolbox import ETS as E
from spatialmath import (
    SpatialAcceleration,
    SpatialVelocity,
    SpatialInertia,
    SpatialForce,
)
from math import pi

import roboticstoolbox as rtb


# inverse dynamics (recursive Newton-Euler) using spatial vector notation
def ID(robot, q, qd, qdd):

    n = robot.n

    # allocate intermediate variables
    Xup = SE3.Alloc(n)
    Xtree = SE3.Alloc(n)

    v = SpatialVelocity.Alloc(n)
    a = SpatialAcceleration.Alloc(n)
    f = SpatialForce.Alloc(n)
    I = SpatialInertia.Alloc(n)
    s = [None for i in range(n)]  # joint motion subspace
    Q = np.zeros((n,))  # joint torque/force

    # initialize intermediate variables
    for j, link in enumerate(robot):
        I[j] = SpatialInertia(m=link.m, r=link.r)
        Xtree[j] = link.Ts
        s[j] = link.v.s

    a_grav = SpatialAcceleration(robot.gravity)

    # forward recursion
    for j in range(0, n):
        vJ = SpatialVelocity(s[j] * qd[j])

        # transform from parent(j) to j
        Xup[j] = robot[j].A(q[j]).inv()

        if robot[j].parent is None:
            v[j] = vJ
            a[j] = Xup[j] * a_grav + SpatialAcceleration(s[j] * qdd[j])
        else:
            jp = robot[j].parent.jindex
            v[j] = Xup[j] * v[jp] + vJ
            a[j] = Xup[j] * a[jp] + SpatialAcceleration(s[j] * qdd[j]) + v[j] @ vJ

        f[j] = I[j] * a[j] + v[j] @ (I[j] * v[j])

    # backward recursion
    for j in reversed(range(0, n)):
        Q[j] = f[j].dot(s[j])

        if robot[j].parent is not None:
            jp = robot[j].parent.jindex
            f[jp] = f[jp] + Xup[j] * f[j]

    return Q


if __name__ == "__main__":
    # robot = rtb.models.URDF.UR5()

    # this has reverse axis rotation to DH twolink model

    l1 = rtb.ELink(ets=E.ry(), m=1, r=[0.5, 0, 0], name="l1")
    l2 = rtb.ELink(ets=E.tx(1) * E.ry(), m=1, r=[0.5, 0, 0], parent=l1, name="l2")
    robot = rtb.ERobot([l1, l2], name="simple 2 link")

    print(robot)

    print(robot.dyntable())
    print(robot.gravity)
    print(robot[1].v)
    robot.dyntable()

    z = np.zeros(robot.n)

    # should be [-2g -0.5g]
    print(robot.fkine([0, 0]))
    tau = ID(robot, [0, 0], z, z)
    print("tau", tau)

    # should be [-1.5g 0]
    tau = ID(robot, [0, -pi / 2], z, z)
    print("tau", tau)

    # should be [-1.5 0.5]
    # tau = ID(robot, [pi/2, -pi/2], [2,-3], [3,4])
    robot.gravity = [0, 0, 0]
    tau = ID(robot, [0, -pi / 2], [-1, -1], z)
    print("tau", tau)

    # should be [0.5 0.5]
    tau = ID(robot, [0, -pi / 2], [1, -1], z)
    print("tau", tau)

    # should be [3.25 1]
    tau = ID(robot, [0, 0], z, [-1, -1])
    print("tau", tau)

    # should be [1.75 0.5]
    tau = ID(robot, [0, pi / 2], z, [-1, -1])
    print("tau", tau)
