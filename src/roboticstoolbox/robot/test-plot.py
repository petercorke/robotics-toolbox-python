from roboticstoolbox import ET2
import numpy as np

a1 = 1
a2 = 1
e = ET2.R() * ET2.tx(a1) * ET2.R() * ET2.tx(a2)
e.plot(np.deg2rad([30, 40]), block=True)
