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

sol = panda.ik_LM(Tep, tol=tol, q0=q0, method="chan")
