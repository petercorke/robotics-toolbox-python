import numpy as np
from collections import OrderedDict
import roboticstoolbox as rtb
import timeit

np.set_printoptions(linewidth=np.inf)


class KinematicCache:
    """
    Kinematic cache

    Many robot kinematic (and dynamic operations) have dependencies. For
    example, computing the world-frame Jacobian requires the
    forward kinematics, computing operational space acceleration requires
    the Jacobian.  To optimize computation time it becomes difficult to keep
    track of all the dependencies.

    The ``KinematicCache`` acts as a proxy for a ``Robot`` subclass object
    and implements a subset of its methods, just those that concerned with, or
    using kinematics.

    For everycall a hash is computed for ``q`` and relevant arguments such as
    ``end`` and the value of kinematic operation is looked up in the cache.  If
    it is not in the cache it will be computed and added to the cache.

    For example::

        robot = models.ETS.Panda()
        kc = KinematicCache(robot)

        q = robot.qr
        T = kc.fkine(q)
        J = kc.jacob0(q)
        Ix = kc.inertia_x(q)
        J = kc.jacob0(q)
    
    The ``fkine`` method will be a cache miss and the forward kinematics will be
    computed.  The ``jacob0`` method will be a cache miss but the required
    forward kinematics are in the cache and will be used.  The ``inertia_x``
    method will be a cache miss but the required Jacobian is in the cache and
    will be used. The final ``jacob0`` method will be a cache hit and the
    previously computed value will be returned.

    The cost of computing the hash is small compared to the cost of the
    kinematic operations and not having to keep track of saved values makes code
    cleaner.

    """
    def __init__(self, robot, cachesize=16):
        """
        Create kinematic cache instance

        :param robot: robot to be cached
        :type robot: Robot subclass instance
        :param cachesize: maximum length of cache, defaults to 16
        :type cachesize: int, optional

        The cache is an ordered dictionary indexed by function, joint angles and
        method arguments.  If you use N different cached functions at each
        timestep then ``cachesize`` should be at least N.
        """
        self._robot = robot
        self._cachesize = cachesize
        self._dict = OrderedDict()

    def __str__(self):
        s = f"KinematicCache({self._robot.name})"
        return s

    def __repr__(self):
        return str(self)

    def __len__(self):
        """
        Length of kinematic cache

        :return: number of cache entries
        :rtype: int

        This is the length of the cache dictionary.
        """
        return len(self._dict)

    def cache(self):
        """
        Display kinematic cache

        :return: cache entries, one per line
        :rtype: str

        The cache dictionary is displayed.  Oldest entries are first.
        For example, the display::

            fkine_all   : 0x59913cdb1a5be5c0, (None,)
            fkine       : 0xb9cd1db3d2a255e0, (None,)
            fkine_all   : 0xb9cd1db3d2a255e0, (None,)
            fkine       : 0x639cf014e2baaafb, (None,)

        shows the kinematic function, the joint configuration hash, and any
        additional arguments.
        """
        s = ""
        for key in self._dict.keys():
            s += f"{key[0]:12s}: {np.uint64(key[1]):#0x}, {key[2:]}\n"
        return s

    def _qhash(self, q):
        """
        Compute joint configuration hash

        :param q: joint configuration :type q: ndarray(N) :return: hash :rtype:
        int

        Returns an integer hash of the joint configuration and trims the cache
        to length of ``cachesize``

        .. note:: Uses ``hash(q.tobytes('C'))`` as the hash, takes around 250us.

        .. note:: Hashing and cache trimming could be a separate methods, but
            since hash has to be computed for every cached kinematic function
            it's quicker to merge both functions.
        """
        # cache is an ordered dict, new entries go on the end to old entries
        # are popped from the front (last=False)
        while len(self._dict) > self._cachesize:
            self._dict.popitem(last=False)

        # compute and return the hash
        return hash(q.tobytes("C"))

    # TODO:
    # this needs to accept a common subset of arguments for DHRobot and ERobot classes.
    # that means end, tool and start.  Need to rationalise these
    # end can be ELink or str, means we have to make Elink hashable
    # if tool is allowed, then SE3 needs to be hashable, probably not a bad idea
    def fkine(self, q, end=None):
        """
        Cached forward kinematics

        :param q: Joint configuration
        :type q: ndarray(n)
        :param end: specific end-effector, defaults to None
        :type end: str or ELink instance
        :return: forward kinematics
        :rtype: SE3 instance

        :seealso: :func:`DHRobot.fkine`, :func:`ERobot.fkine`
        """
        # compute the key we use for cache lookup
        qhash = self._qhash(q)
        key = ("fkine", qhash, end)
        if key not in self._dict:
            # cache miss, check if we have fkine_all() result
            key_fkall = ("fkine_all", qhash, end)
            if key_fkall in self._dict:
                # we do, take just the EE pose and cache that
                self._dict[key] = self._dict[key_fkall][-1]
            else:
                # nothing cached, compute fkine() the hard way and cache it
                self._dict[key] = self._robot.fkine(q)
        # return the cached value
        return self._dict[key]

    def fkine_all(self, q, end=None):
        """
        Cached forward kinematics for all frames

        :param q: joint configuration
        :type q: ndarray(n)
        :param end: specific end effector, defaults to None
        :type end: str or ELink instance, optional
        :return: all link frames including base
        :rtype: multi-valuedSE3 instance
        """

        key = ("fkine_all", self._qhash(q), end)
        if key not in self._dict:
            # cache miss, compute it
            self._dict[key] = self._robot.fkine_all(q)
        return self._dict[key]

    def jacob0(self, q, end=None, analytical=None, half=None):
        """
        Cached world-frame Jacobian

        :param q: joint configuration
        :type q: ndarray(n)
        :param end: specific end effector, defaults to None
        :type end: str or ELink instance, optional
        :param half: return half Jacobian: 'trans' or 'rot'
        :type half: str
        :param analytical: return analytical Jacobian instead of geometric Jacobian (default)
        :type analytical: str
        :return: Jacobian in world frame
        :rtype: ndarray(6, n)
        """
        # NOTE: we could write
        #  def jacob0(self, q, **kwargs):
        #      key = ('jacob0', self._qhash(q), tuple(kwargs.keys()))
        # which is more elegant/compact but the order of the kw arguments would
        # then become important
        key = ("jacob0", self._qhash(q), end, analytical, half)
        if key not in self._dict:
            # cache miss, compute it

            # get fkine from the cache if possible
            # TODO: fkine() will have to compute the hash again, maybe pass it
            # down as argument _qhash
            T = self.fkine(q, end=end)
            self._dict[key] = self._robot.jacob0(
                q, T=T, end=end, half=half, analytical=analytical
            )
        return self._dict[key]

    def jacob0_inv(self, q, end=None, analytical=None):
        """
        Cached world-frame Jacobian inverse

        :param q: joint configuration
        :type q: ndarray(n)
        :param end: specific end effector, defaults to None
        :type end: str or ELink instance, optional
        :param analytical: return analytical Jacobian instead of geometric Jacobian (default)
        :type analytical: str
        :return: Inverse Jacobian in world frame
        :rtype: ndarray(6, n)

        .. note:: Robot objects don't have this method.
        """
        key = ("jacob0_inv", self._qhash(q), end, analytical)
        if key not in self._dict:
            # cache miss, compute it

            # get Jacobian from cache
            J = self.jacob0(q, end=end, analytical=analytical)
            self._dict[key] = np.linalg.inv(J)
        return self._dict[key]

    def jacob0_pinv(self, q, end=None, analytical=None):
        """
        Cached world-frame Jacobian pseudo inverse

        :param q: joint configuration
        :type q: ndarray(n)
        :param end: specific end effector, defaults to None
        :type end: str or ELink instance, optional
        :param analytical: return analytical Jacobian instead of geometric Jacobian (default)
        :type analytical: str
        :return: Pseudo inverse Jacobian in world frame
        :rtype: ndarray(6, n)

        .. note:: Robot objects don't have this method.
        """
        key = ("jacob0_pinv", self._qhash(q), end, analytical)
        if key not in self._dict:
            # cache miss, compute it

            # get Jacobian from cache
            J = self.jacob0(q, end=end, analytical=analytical)
            self._dict[key] = np.linalg.pinv(J)
        return self._dict[key]

    # TODO jacobe doesnt support end for DHRobot, should it have analytical
    def jacobe(self, q, end=None, half=None):
        """
        Cached world-frame Jacobian

        :param q: joint configuration
        :type q: ndarray(n)
        :param end: specific end effector, defaults to None
        :type end: str or ELink instance, optional
        :param half: return half Jacobian: 'trans' or 'rot'
        :type half: str
        :return: Jacobian in end-effector-frame
        :rtype: ndarray(6, n)
        """
        # NOTE: we could write
        #  def jacob0(self, q, **kwargs):
        #      key = ('jacob0', self._qhash(q), tuple(kwargs.keys()))
        # which is more elegant/compact but the order of the kw arguments would
        # then become important
        key = ("jacobe", self._qhash(q), end, half)
        if key not in self._dict:
            # cache miss, compute it

            # get fkine from the cache if possible
            if isinstance(self._robot, DHRobot):
                T = None  # actually not needed for DHRobot case
            else:
                T = self.fkine(q, end=end)
            self._dict[key] = self._robot.jacobe(q, T=T, end=end)
        return self._dict[key]

    def jacobe_inv(self, q, end=None, analytical=None):
        """
        Cached end-effector-frame Jacobian inverse

        :param q: joint configuration
        :type q: ndarray(n)
        :param end: specific end effector, defaults to None
        :type end: str or ELink instance, optional
        :return: Inverse Jacobian in end-effector-frame
        :rtype: ndarray(6, n)

        .. note:: Robot objects don't have this method.
        """
        key = ("jacobe_inv", self._qhash(q), end)
        if key not in self._dict:
            # cache miss, compute it

            # get Jacobian from cache
            J = self.jacobe(q, end=end)
            self._dict[key] = np.linalg.inv(J)
        return self._dict[key]

    def jacobe_pinv(self, q, end=None, analytical=None):
        """
        Cached end-effector-frame Jacobian pseudo inverse

        :param q: joint configuration
        :type q: ndarray(n)
        :param end: specific end effector, defaults to None
        :type end: str or ELink instance, optional
        :return: Pseudo inverse Jacobian in end-effector-frame
        :rtype: ndarray(6, n)

        .. note:: Robot objects don't have this method.
        """
        key = ("jacob0_pinv", self._qhash(q), end, analytical)
        if key not in self._dict:
            # cache miss, compute it

            # get Jacobian from cache
            J = self.jacobe(q, end=end)
            self._dict[key] = np.linalg.pinv(J)
        return self._dict[key]

    def coriolis(self, q, qd):
        key = ("coriolis", self._qhash(q), self._qhash(qd))
        if key not in self._dict:
            # cache miss, compute it
            C = self.coriolis(q, qd)
            self._dict[key] = self._robot.coriolis_x(q, qd)
        return self._dict[key]

    def inertia_x(self, q, pinv=False, analytical="rpy-xyz"):
        key = ("inertia_x", self._qhash(q), pinv, analytical)
        if key not in self._dict:
            # cache miss, compute it

            # get Jacobian inv or pinv from cache
            if pinv:
                Ji = self.jacobe0_pinv(analytical=analytical)
            else:
                Ji = self.jacobe0_inv(analytical=analytical)
            self._dict[key] = self._robot.inertia_x(q, Ji=Ji)
        return self._dict[key]

    def coriolis_x(self, q, qd, pinv=False, analytical="rpy-xyz"):
        key = ("coriolis_x", self._qhash(q), self._qhash(qd), pinv, analytical)
        if key not in self._dict:
            # cache miss, compute it

            # get Jacobian inv or pinv from cache
            if pinv:
                Ji = self.jacobe0_pinv(analytical=analytical)
            else:
                Ji = self.jacobe0_inv(analytical=analytical)
            # get inertia, Jacobian dot and Coriolis from cache
            Mx = self.inertia_x(q, pinv=pinv, analytical=analytical)
            Jd = self.jacob_dot(q, J0=self.jacob0(q, analytical=analytical))
            C = self.coriolis(q, qd)
            self._dict[key] = self._robot.coriolis_x(q, qd, pinv, analytical, 
                J, Ji, Jd, C, Mx)
        return self._dict[key]

    def gravload_x(self, q, pinv=False, analytical="rpy-xyz"):
        key = ("gravload_x", self._qhash(q), pinv, analytical)
        if key not in self._dict:
            # cache miss, compute it

            # get Jacobian inv or pinv from cache
            if pinv:
                Ji = self.jacobe0_pinv(analytical=analytical)
            else:
                Ji = self.jacobe0_inv(analytical=analytical)
            self._dict[key] = self._robot.gravload_x(q, Ji=Ji)
        return self._dict[key]


if __name__ == "__main__":
    q = np.r_[1, 2, 3, 4, 5, 6, 7]

    panda = rtb.models.DH.Panda()
    kc = KinematicCache(panda)
    print(kc)
    print(panda.fkine(panda.qr))
    print(panda.fkine(panda.qr))
    print("--")
    print(panda.jacob0(panda.qr))
    print(panda.jacob0(panda.qr))
    print("--")

    def timething(statement, N=1_000):
        t = timeit.timeit(stmt=statement, globals=globals(), number=N)
        print(f"{statement:>25s}: {t / N * 1e6:.1f}μs")

    timething("kc._qhash(q)", 1_000_000)
    timething("panda.fkine(panda.qr)", 1_000)
    timething("kc.fkine(panda.qr)", 1_000_000)

    kc.fkine_all(panda.qr)
    print(len(kc))
    print(kc.cache())

    for i in range(50):
        q = np.random.rand(7)
        kc.fkine(q)
        kc.fkine_all(q)
    print(len(kc))
    print(kc.cache())
