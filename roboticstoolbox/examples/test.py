import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

import numpy.testing as nt

n = 100

coord = np.random.random((n, 6))
coord2 = np.random.random((n, 6))

for co, co2 in zip(coord, coord2):
    Te = (
        sm.SE3.Trans(co[:3]) * sm.SE3.Rx(co[3]) * sm.SE3.Ry(co[4]) * sm.SE3.Rz(co[5])
    ).A
    Tep = (
        sm.SE3.Trans(co2[:3])
        * sm.SE3.Rx(co2[3])
        * sm.SE3.Ry(co2[4])
        * sm.SE3.Rz(co2[5])
    ).A

    e1 = rtb.angle_axis(Te, Tep)
    e2 = rtb.angle_axis_python(Te, Tep)

    # print()
    # print(e1)
    # print(e2)

    nt.assert_allclose(e1, e2)


# Te = sm.SE3.Rx(0.1).A
# Tep = sm.SE3.Rx(0.2).A

# e1 = rtb.angle_axis(Te, Tep)
# e2 = rtb.angle_axis_test(Te, Tep)

# print(e1)
# print(e2)


# solver = rtb.IK_LM()

# r = rtb.models.Panda().ets()
# r2 = rtb.models.Panda()

# Tep = r.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

# print(sm.SE3(Tep, check=False))

# sol = r.ik_nr(Tep)
# sol2 = r2.ik_nr(Tep)

# print(sol[1])
# print(sol2[1])
# print()

# print(sol[0])
# print(sol2[0])
# print()

# Tq = r.eval(sol[0])
# Tq2 = r2.fkine(sol2[0]).A

# print(sol[4])
# print(sol2[4])
# print()
# print(sm.SE3(Tq, check=False))
# print(sm.SE3(Tq2, check=False))

# _, E = solver.error(Tep, Tq)
# _, E2 = solver.error(Tep, Tq2)

# print()
# print(E)
# print(E2)
