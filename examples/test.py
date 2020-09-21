import roboticstoolbox as rp
import numpy as np

r = rp.models.Panda()

print(r.fkine_graph())
print(r.fkine())

# path = r.get_path(r.base_link, r.ee_link)

# print(np.round(r.jacobe(), 2))
# print(np.round(r.jacobe_old(), 2))

# for link in r._fkpath:
#     print(link.name)
#     print(link.ets)
#     print(link.jtype)
#     print(link.collision)

# print(np.round(r.jacob0(), 2))
