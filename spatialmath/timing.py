#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:22:36 2020

@author: corkep
"""


if __name__ == '__main__':

    import timeit

    N = 1000000

    quat_setup = '''import spatialmath.base.quaternion as quat
    import spatialmath.quaternion as qq
    import numpy as np
    q1 = quat.qrand()
    q2 = quat.qrand()
    v = np.r_[1,2,3]
    Q1 = qq.UnitQuaternion.Rx(0.2)
    Q2 = qq.UnitQuaternion.Ry(0.3)'''

    transforms_setup = '''import spatialmath.base.transforms as tr'''


    t = timeit.timeit(stmt='tr.rotx(0.2)', setup=transforms_setup, number=N)
    print('transforms.rotx: ', t, ' us')


    t = timeit.timeit(stmt='a = quat.qqmul(q1,q2)', setup=quat_setup, number=N)
    print('quat.qqmul:         ', t, ' us')
    t = timeit.timeit(stmt='a = quat.qvmul(q1,v)', setup=quat_setup, number=N)
    print('quat.qqmul:         ', t, ' us')
    t = timeit.timeit(stmt='a = qq.UnitQuaternion()', setup=quat_setup, number=N)
    print('UnitQuaternion() :  ', t, ' us')
    t = timeit.timeit(stmt='a = qq.UnitQuaternion.Rx(0.2)', setup=quat_setup, number=N)
    print('UnitQuaternion.Rx : ', t, ' us')
    t = timeit.timeit(stmt='a = Q1 * Q2', setup=quat_setup, number=N)
    print('UnitQuaternion *:   ', t, ' us')
