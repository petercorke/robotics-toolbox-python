def ctraj(T0, T1, N):
    """
    CTRAJ(T0, T1, N) is a Cartesian trajectory (4x4xN) from pose T0 to T1
    with N points that follow a trapezoidal velocity profile along the path.
    :param T0: Start pose
    :param T1: End pose
    :param N: number of points to be interpolated
    :return: SE3 pose
    """
    from .common import ishomog
    from .quaternion import UnitQuaternion
    from .transforms import t2r, transl
    from .pose import SE3
    assert type(N) is float or type(N) is int
    assert type(T0) is list or ishomog(T0, (4, 4))
    assert type(T1) is list or ishomog(T1, (4, 4))

    if type(T0) is list:
        for each in T0:
            assert ishomog(each, (4, 4))
    if type(T1) is list:
        for each in T1:
            assert ishomog(each, (4, 4))

    def one_to_one(T0, T1):
        rot_0 = t2r(T0)
        rot_1 = t2r(T1)
        transl_0 = T0[0:3, 3]
        transl_1 = T1[0:3, 3]
        q0 = UnitQuaternion.rot(rot_0)
        q1 = UnitQuaternion.rot(rot_1)

        time_steps = lspb(0, 1, N)
        traj = []

        for i in time_steps:
            rot_interped = q0.interp(q1, r=i).q2tr()
            transl_interped = transl_0 + (transl_1 - transl_0) * i
            transl_interped = transl([transl_interped[0, 0], transl_interped[1, 0], transl_interped[2, 0]])
            traj.append(transl_interped * rot_interped)

        return traj

    if ishomog(T0, (4, 4)):
        if ishomog(T1, (4, 4)):
            return SE3.np(one_to_one(T0, T1))
        elif type(T1) is list:
            return [SE3.np(one_to_one(T0, each)) for each in T1]
    elif type(T0) is list:
        if ishomog(T1, (4, 4)):
            return [SE3.np(one_to_one(each, T1)) for each in T0]
        elif type(T1) is list:
            assert len(T0) == len(T1), "For many to many trajectory computation, both lists should be of same length"
            return [SE3.np(one_to_one(T0[i], T1[i])) for i in range(len(T0))]


def lspb(q0, q1, t, V=None):
    import numpy as np
    if (type(t) is int) or (type(t) is float):
        t = np.arange(t)
    else:
        assert type(t) is np.ndarray

    tf = float(np.amax(t))

    if V is None:
        V = (q1 - q0) / tf * 1.5
    else:
        V = abs(V) * ((q1 - q0) / abs(q1 - q0))
        if abs(V) < abs(q1 - q0) / tf:
            raise ValueError('V too small')
        elif abs(V) > 2 * abs(q1 - q0) / tf:
            raise ValueError('V too big')

    if q0 == q1:
        s = np.ones(t.shape[0]) * q0
        sd = np.zeros(t.shape[0])
        sdd = np.zeros(t.shape[0])
        return s, sd, sdd

    tb = (q0 - q1 + V * tf) / V
    a = V / tb

    p = np.zeros(t.shape[0])
    pd = np.copy(p)
    pdd = np.copy(p)

    for i in range(t.shape[0]):
        tt = float(t[i])
        if tt <= tb:
            p[i] = q0 + a / 2 * tt ** 2
            pd[i] = V
            pdd[i] = a
        elif tt <= (tf - tb):
            p[i] = (q1 + q0 - V * tf) / 2 + V * tt
            pd[i] = V
            pdd[i] = a
        else:
            p[i] = q1 - a / 2 * tf ** 2 + a * tf * tt - a / 2 * tt ** 2
            pd[i] = a * tf - a * tt
            pdd[i] = -a

    return p
