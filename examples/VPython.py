#!/usr/bin/env python
"""
@author Micah Huth
"""

import roboticstoolbox as rp
import time

env = rp.backend.VPython()
env.launch()

puma = rp.models.DH.Puma560()
env.add(0, 'Puma', puma)

time.sleep(1)
env.step(puma, puma.qz, 0)

time.sleep(1)
env.step(puma, puma.qr, 0)

time.sleep(1)
env.step(puma, puma.qs, 0)

time.sleep(1)
env.step(puma, puma.qn, 0)