import numpy as np
import roboticstoolbox as rtb
import spatialmath as sm


r = rtb.models.Fetch()

for l in r:
    print(l.name)
