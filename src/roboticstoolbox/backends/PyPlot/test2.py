from roboticstoolbox import *

# puma = models.ETS.Panda()
# print(puma)

# puma.plot(puma.qr, block=True)

a1 = 1
e = ET2.R() * ET2.tx(a1)
e.plot(0, block=True)
