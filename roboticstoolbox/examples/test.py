import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm

tol = 1e-6

q0 = np.array(
    [
        -1.49477148,
        1.1828801,
        2.09337123,
        -2.10498264,
        -1.03160669,
        1.43577474,
        1.55373996,
    ]
)

panda = rtb.models.Panda().ets()

Tep = panda.eval([0, -0.3, 0, -2.2, 0, 2.0, np.pi / 4])

solver = rtb.IK_QP(
    joint_limits=True,
    seed=0,
    tol=tol,
    kq=2.0,
    pi=0.01,
    ps=0.001,
    kj=0.1,
    slimit=1,
)

sol = solver.solve(panda, Tep, q0=q0)

print(sol.success)

if sol.success and sol.searches < 2:
    print(sol.q.__repr__())
    print(sol.success)
    print(f"iterations: {sol.iterations}, search: {sol.searches}")

    Tq = panda.eval(sol.q)

    _, E = solver.error(Tep, Tq)

    print(tol < E)
