#!/usr/bin/env python
"""
@author Micah Huth
"""

import roboticstoolbox as rp
import time

env = rp.backends.VPython()
env.launch(g_col=[0.2, 0.2, 0.2], g_opc=0.3)
# env.launch()

#  PUMA560
puma = rp.models.DH.Puma560()
env.add(0, 'Puma', puma)

env.record_start(10)

time.sleep(3)
env.step(puma, puma.qr, 0)

time.sleep(3)
env.step(puma, puma.qs, 0)

time.sleep(3)
env.step(puma, puma.qn, 0)

time.sleep(3)
env.step(puma, puma.qz, 0)

env.record_stop('vpython_video.mp4')


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
