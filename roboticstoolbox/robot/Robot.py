import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath.base.argcheck import getvector
from roboticstoolbox.robot.Link import Link

class Robot:

    def __init__(
            self,
            links,
            name='noname',
            manufacturer='',
            base=None,
            tool=None,
            gravity=None,
            meshdir=None,
            keywords=()):

        self.name = name
        self.manufacturer = manufacturer

        self.base = base
        self.tool = tool
        if keywords is not None and not isinstance(keywords, (tuple, list)):
            raise TypeError('keywords must be a list or tuple')
        else:
            self.keywords = keywords
        if gravity is None:
            gravity = np.array([0, 0, 9.81])
        self.gravity = gravity

        if not isinstance(links, list):
            raise TypeError('The links must be stored in a list.')
        for link in links:
            if not isinstance(link, Link):
                raise TypeError('links should all be Link subclass')
            link._robot = self
        self._links = links

        self._configdict = {}

        self._dynchange = True

    def __getitem__(self, i):
        return self._links[i]

    def dynchanged(self):
        self._dynchanged = True

    @property
    def n(self):
        return len(self._links)

    def addconfiguration(self, name, q):
        v = getvector(q, self.n)
        self._configdict[name] = v
        setattr(self, name, v)

    # --------------------------------------------------------------------- #
    @property
    def name(self):
        return self._name

    @property
    def manufacturer(self):
        return self._manufacturer

    @property
    def links(self):
        return self._links

    @property
    def base(self):
        if self._base is None:
            return SE3()
        else:
            return self._base

    @property
    def tool(self):
        if self._tool is None:
            return SE3()
        else:
            return self._tool

    @property
    def gravity(self):
        return self._gravity

    @name.setter
    def name(self, name_new):
        self._name = name_new

    @manufacturer.setter
    def manufacturer(self, manufacturer_new):
        self._manufacturer = manufacturer_new

    @base.setter
    def base(self, T):
        # if not isinstance(T, SE3):
        #     T = SE3(T)
        if T is None or isinstance(T, SE3):
            self._base = T
        elif SE3.isvalid(T):
            self._tool = SE3(T, check=False)
        else:
            raise ValueError('base must be set to None (no tool) or an SE3')

    @tool.setter
    def tool(self, T):
        # if not isinstance(T, SE3):
        #     T = SE3(T)
        # this is allowed to be none, it's helpful for symbolics rather than having an identity matrix
        if T is None or isinstance(T, SE3):
            self._tool = T
        elif SE3.isvalid(T):
            self._tool = SE3(T, check=False)
        else:
            raise ValueError('tool must be set to None (no tool) or an SE3')

    @gravity.setter
    def gravity(self, gravity_new):
        self._gravity = getvector(gravity_new, 3, 'col')
        self.dynchanged()

    # --------------------------------------------------------------------- #

    @property
    def q(self):
        return self._q

    @property
    def qd(self):
        return self._qd

    @property
    def qdd(self):
        return self._qdd

    @property
    def control_type(self):
        return self._control_type

    @q.setter
    def q(self, q_new):
        self._q = getvector(q_new, self.n)

    @qd.setter
    def qd(self, qd_new):
        self._qd = getvector(qd_new, self.n)

    @qdd.setter
    def qdd(self, qdd_new):
        self._qdd = getvector(qdd_new, self.n)

    @control_type.setter
    def control_type(self, cn):
        if cn == 'p' or cn == 'v' or cn == 'a':
            self._control_type = cn
        else:
            raise ValueError(
                'Control type must be one of \'p\', \'v\', or \'a\'')
        