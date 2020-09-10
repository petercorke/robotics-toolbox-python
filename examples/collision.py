#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import spatialmath as sm
import numpy as np

r = rp.models.Panda()
r.q = r.qr

r.scollision()
