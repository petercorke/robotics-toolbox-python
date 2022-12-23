import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

solver = rtb.IK_LM()
r = rtb.models.Panda().ets()
r2 = rtb.models.Panda()

Tep = r.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

print(sm.SE3(Tep, check=False))

sol = r.ik_gn(Tep)
sol2 = r2.ik_gn(Tep)

print(sol[1])
print(sol2[1])
print()

Tq = r.eval(sol[0])
Tq2 = r2.fkine(sol2[0]).A

print(sol[4])
print(sol2[4])
print()
print(sm.SE3(Tq, check=False))
print(sm.SE3(Tq2, check=False))

_, E = solver.error(Tep, Tq)
_, E2 = solver.error(Tep, Tq2)

print()
print(E)
print(E2)
