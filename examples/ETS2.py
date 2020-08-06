#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import ropy as rp
# import numpy as np

# # Contruct a robot
# l0 = rp.ET.TRz()
# l1 = rp.ET.Ttx(1)

# E = rp.ETS([l0, l1])

# E.q = 30 * np.pi/180

# print(E.fkine())

# E.teach()

env = rp.PyPlot2()

env.launch()

env.hold()
