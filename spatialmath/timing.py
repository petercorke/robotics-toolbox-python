#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:22:36 2020

@author: corkep
"""


import timeit

t = timeit.timeit(stmt='transforms.rotx(0.2)', setup='import transforms', number=1000000)
print('t=', t, ' us')