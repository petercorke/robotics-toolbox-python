#!/usr/bin/env python
"""
@author Micah Huth
"""

import roboticstoolbox as rtb
# import time

from roboticstoolbox.backends.VPython import VPython

env = VPython()

env.launch()

#  PUMA560
puma = rtb.models.DH.Puma560()
env.add(puma)

qt = rtb.tools.trajectory.jtraj(puma.qz, puma.qr, 200)

for q in qt.q:
    env.step(0.1)
    puma.q = q

# Example 1
# qt = rp.tools.trajectory.jtraj(puma.qz, puma.qr, 50)
# env.record_start(5)
# for q in qt.q:
#     time.sleep(1/5)
#     env.step(puma, q=q)
# env.record_stop('vpython_jtraj_video.mp4', save_fps=25)


# Example 2
# env.record_start(10)
#
# time.sleep(1)
# env.step(puma, puma.qr, 0)
#
# time.sleep(1)
# env.step(puma, puma.qs, 0)
#
# time.sleep(1)
# env.step(puma, puma.qn, 0)
#
# time.sleep(1)
# env.step(puma, puma.qz, 0)
#
# env.record_stop('vpython_video.mp4')


#  PANDA
# panda = rp.models.DH.Panda()
# env.add(0, 'Panda', panda)
#
# time.sleep(3)
# env.step(panda, panda.qr, 0)
#
# time.sleep(3)
# env.step(panda, panda.qz, 0)


#  KR5
# kr5 = rp.models.DH.KR5()
# env.add(0, 'KR5', kr5)
#
# time.sleep(3)
# env.step(kr5, kr5.qk1, 0)
#
# time.sleep(3)
# env.step(kr5, kr5.qk2, 0)
#
# time.sleep(3)
# env.step(kr5, kr5.qk3, 0)
#
# time.sleep(3)
# env.step(kr5, kr5.qz, 0)


#  IRB140
# irb140 = rp.models.DH.IRB140()
# env.add(0, 'IRB140', irb140)
#
# time.sleep(3)
# env.step(irb140, irb140.qr, 0)
#
# time.sleep(3)
# env.step(irb140, irb140.qd, 0)
#
# time.sleep(3)
# env.step(irb140, irb140.qz, 0)
