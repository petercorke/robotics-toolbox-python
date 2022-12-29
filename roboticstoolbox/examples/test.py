import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

q0 = np.array(
    [
        -1.66441371,
        -1.20998727,
        1.04248366,
        -2.10222463,
        1.05097407,
        1.41173279,
        0.0053529,
    ]
)

tol = 1e-6

panda = rtb.models.Panda().ets()

Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

solver = rtb.IK_QP()

sol = panda.ikine_QP(Tep, tol=tol, q0=q0)

Tq = panda.eval(sol.q)

_, E = solver.error(Tep, Tq)


print(sol.success)

if sol.success and sol.searches < 2:
    print(sol.q.__repr__())
    print(sol.success)
    print(sol.residual)
    print(f"iterations: {sol.iterations}, search: {sol.searches}")

    Tq = panda.eval(sol.q)

    print(E)
