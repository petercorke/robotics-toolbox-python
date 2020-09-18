import roboticstoolbox as rp
import numpy as np

r = rp.models.Panda()

for link in r._fkpath:
    print(link.name)
    print(link.ets)
    print(link.jtype)

print(np.round(r.jacob0(), 2))
