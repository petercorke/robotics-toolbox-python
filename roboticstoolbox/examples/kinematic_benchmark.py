#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rtb
import numpy as np
import timeit

panda = rtb.models.Panda()
panda.q = panda.qr

ks = rtb.KinematicCache(panda)

reps = 100000


def fkine_sameq():
    for _ in range(reps):
        panda.fkine(panda.q, fast=True)


def fkine_cache_sameq():
    for _ in range(reps):
        ks.fkine(panda.q)


def fkine_diffq():
    for _ in range(reps):
        q = np.random.random(7)
        panda.fkine(q, fast=True)


def fkine_cache_diffq():
    for _ in range(reps):
        q = np.random.random(7)
        ks.fkine(q)


def jacob0_sameq():
    for _ in range(reps):
        panda.jacob0(panda.q, fast=True)


def jacob0_cache_sameq():
    for _ in range(reps):
        ks.jacob0(panda.q)


def jacob0_diffq():
    for _ in range(reps):
        q = np.random.random(7)
        panda.jacob0(q, fast=True)


def jacob0_cache_diffq():
    for _ in range(reps):
        q = np.random.random(7)
        ks.jacob0(q)


print("Fkine Same q:")
print("Normal: ", timeit.Timer(fkine_sameq).timeit(number=1))
print("Cached: ", timeit.Timer(fkine_cache_sameq).timeit(number=1))

print("")
print("Fkine Different q:")
print("Normal: ", timeit.Timer(fkine_diffq).timeit(number=1))
print("Cached: ", timeit.Timer(fkine_cache_diffq).timeit(number=1))

print("")
print("Jacob0 Same q:")
print("Normal: ", timeit.Timer(jacob0_sameq).timeit(number=1))
print("Cached: ", timeit.Timer(jacob0_cache_sameq).timeit(number=1))

print("")
print("Jacob0 Different q:")
print("Normal: ", timeit.Timer(jacob0_diffq).timeit(number=1))
print("Cached: ", timeit.Timer(jacob0_cache_diffq).timeit(number=1))
