import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from spatialmath.base.argcheck import getvector
from roboticstoolbox.robot.Link import Link
from roboticstoolbox.backend import URDF
from roboticstoolbox.backend import xacro
from pathlib import PurePath, PurePosixPath, Path
import sys

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
            keywords=(),
            symbolic=False):

        self.name = name
        self.manufacturer = manufacturer
        self.symbolic = symbolic
        self.base = base
        self.tool = tool
        self.basemesh = None
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

        if meshdir is not None:
            classpath = sys.modules[self.__module__].__file__
            self.meshdir = PurePath(classpath).parent / PurePosixPath(meshdir)
            self.basemesh = self.meshdir / "link0.stl"
            for j, link in enumerate(self._links, start=1):
                link.mesh = self.meshdir / "link{:d}.stl".format(j)

        # URDF Parser Attempt
        # # Search mesh dir for meshes
        # if urdfdir is not None:
        #     # Parse the URDF to obtain file paths and scales
        #     data = self._get_stl_file_paths_and_scales(urdfdir)
        #     # Obtain the base mesh
        #     self.basemesh = [data[0][0], data[1][0], data[2][0]]
        #     # Save the respective meshes to each link
        #     for idx in range(1, self.n+1):
        #         self._links[idx-1].mesh = [data[0][idx], data[1][idx], data[2][idx]]
        # else:
        #     self.basemesh = None

    def __getitem__(self, i):
        return self._links[i]

    # URDF Parser Attempt
    # @staticmethod
    # def _get_stl_file_paths_and_scales(urdf_path):
    #     data = [[], [], []]  # [ [filenames] , [scales] , [origins] ]
    #
    #     name, ext = splitext(urdf_path)
    #
    #     if ext == '.xacro':
    #         urdf_string = xacro.main(urdf_path)
    #         urdf = URDF.loadstr(urdf_string, urdf_path)
    #
    #         for link in urdf.links:
    #             data[0].append(link.visuals[0].geometry.mesh.filename)
    #             data[1].append(link.visuals[0].geometry.mesh.scale)
    #             data[2].append(SE3(link.visuals[0].origin))
    #
    #     return data

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
        self._gravity = getvector(gravity_new, 3)
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
        