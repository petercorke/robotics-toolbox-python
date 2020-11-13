#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp

panda = rp.models.Panda()
panda.plot(q=panda.qr)

# panda = rp.models.DH.Panda()
# panda.teach(q=panda.qr)
