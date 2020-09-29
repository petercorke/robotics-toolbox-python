import roboticstoolbox as rp
import spatialmath as sm
import numpy as np
import time

# import cProfile
panda = rp.models.Panda()
panda.q = panda.qr

# def func():
#     for i in range(1000):
#         panda.jacobe()

# cProfile.run('func()')



env = rp.backend.Swift()
env.launch()

panda = rp.models.Panda()
panda.q = panda.qr

Tep = panda.fkine() * sm.SE3.Tx(-0.2) * sm.SE3.Ty(0.6) * sm.SE3.Tz(0.4)

arrived = False
env.add(panda)

time.sleep(1)

dt = 0.05

while not arrived:


    start = time.time()
    panda.fkine_all()
    v, arrived = rp.p_servo(panda.fkine(), Tep, 1)
    panda.qd = np.r_[np.linalg.pinv(panda.jacobe()) @ v, [0, 0]]
    env.step(50)
    stop = time.time()

    print(panda.manipulability())

    if stop - start < dt:
        time.sleep(dt - (stop - start))
