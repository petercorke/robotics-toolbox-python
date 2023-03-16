# PoERobot
import numpy as np
from spatialmath import Twist3, SE3
from spatialmath.base import skew, trnorm
from roboticstoolbox.robot import Link, Robot
from roboticstoolbox.robot.ET import ET
from roboticstoolbox.robot.ETS import ETS


class PoELink(Link):
    """
    Product of exponential link class

    This is a subclass of the base Link class.

    :seealso: :class:`Link`
    """

    def __init__(self, twist, name=None):
        # get ETS of the link in the world frame, given by its twist
        ets = self.ets_world(twist)

        super().__init__(ets)
        self.S = Twist3(twist)
        self.name = name

    def __repr__(self):
        s = f"PoELink({np.array2string(self.S.S, separator=',')}"
        if self.name is not None:
            s += f', name="{self.name}"'
        s += ")"
        return s

    def __str__(self):
        s = f"{self.__class__.__name__}[twist={self.S}"
        if self.name is not None:
            s += f', name="{self.name}"'
        s += "]"
        return s

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython (superclass method)

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        """
        # see https://ipython.org/ipython-doc/stable/api/generated/IPython.lib.pretty.html

        p.text(f"{i}:\n{str(x)}")

    def ets_world(self, tw):
        # initialize partial transformations between a joint from base to ee
        tf = SE3()
        # get transforms from screws, related to base frame
        # get screw axis components
        w = tw.w
        v = tw.v

        if isinstance(self, PoEPrismatic):
            # switch the directional vector components
            a_vec = v
            # get vector of the x axis
            n_vec = tf.n
            # point on screw axis
            t_vec = tf.t

        elif isinstance(self, PoERevolute):  # is revolute
            # get the nearest point of the screw axis closest to the origin
            principalpoint = np.cross(w, v)

            # get vector of the x-axis
            if round(np.linalg.norm(principalpoint), 3) == 0.0:  # the points are
                # coincident
                n_vec = tf.n
            else:
                n_vec = principalpoint / np.linalg.norm(principalpoint)

            a_vec = w
            t_vec = principalpoint


        else:  # not a joint
            n_vec = tf.n
            a_vec = tf.a
            t_vec = v

        # obtain the other vector elements of transformation matrix
        o_vec = np.cross(a_vec, n_vec)

        # construct transformation matrix from obtained vectors
        f_tf = np.eye(4)
        f_tf[0:3, 0] = n_vec
        f_tf[0:3, 1] = o_vec
        f_tf[0:3, 2] = a_vec
        f_tf[0:3, 3] = t_vec
        # normalize the matrix
        tf = SE3(trnorm(f_tf))

        et = []
        if tf.t[0] != 0.0:
            et.append(ET.tx(tf.t[0]))
        if tf.t[1] != 0.0:
            et.append(ET.ty(tf.t[1]))
        if tf.t[2] != 0.0:
            et.append(ET.tz(tf.t[2]))

        # RPY parameters, due to RPY convention the order of is reversed
        rpy = tf.rpy()
        if rpy[2] != 0.0:
            et.append(ET.Rz(rpy[2]))
        if rpy[1] != 0.0:
            et.append(ET.Ry(rpy[1]))
        if rpy[0] != 0.0:
            et.append(ET.Rx(rpy[0]))

        # assign joint variable, if the frame is not base or tool frame
        if isinstance(self, PoEPrismatic):  # test prismatic joint
            et.append(ET.tz())
        elif isinstance(self, PoERevolute):  # if revolute
            et.append(ET.Rz())
        elif et == []:
            et.append(ET.tx(0))


        ets = ETS()
        for e in et:
            ets *= e

        return ets


class PoERevolute(PoELink):
    def __init__(self, axis, point, **kwargs):
        """
        Construct revolute product of exponential link

        :param axis: axis of rotation
        :type axis: array_like(3)
        :param point: point on axis of rotation
        :type point: array_like(3)

        Construct a link and revolute joint for a PoE Robot.

        :seealso: :class:`Link` :class:`Robot`
        """

        super().__init__(Twist3.UnitRevolute(axis, point), **kwargs)


class PoEPrismatic(PoELink):
    def __init__(self, axis, **kwargs):
        """
        Construct prismatic product of exponential link

        :param axis: direction of motion
        :type axis: array_like(3)

        Construct a link and prismatic joint for a PoE Robot.

        :seealso: :class:`Link` :class:`Robot`
        """
        super().__init__(Twist3.UnitPrismatic(axis), **kwargs)


class PoERobot(Robot):
    def __init__(self, links, T0, **kwargs):
        """
        Product of exponential robot class

        :param links: robot links
        :type links: list of ``PoELink``
        :param T0: end effector pose for zero joint coordinates
        :type T0: SE3

        This is a subclass of the abstract base Robot class that provides
        only forward kinematics and world-frame Jacobian.

        :seealso: :class:`PoEPrismatic` :class:`PoERevolute`
        """

        # add base link and end-effector link
        links.insert(0, PoELink(Twist3()))
        links.append(PoELink(T0.twist()))

        super().__init__(links, **kwargs)
        self.T0 = T0
        self._update_ets()

    def __str__(self):
        """
        Pretty prints the PoE Model of the robot.
        :return: Pretty print of the robot model
        :rtype: str
        """
        s = "PoERobot:\n"
        for j, link in enumerate(self):
            s += f"  {j}: {link.S}\n"

        s += f"  T0: {self.T0.strline()}"
        return s

    def __repr__(self):
        s = "PoERobot([\n    "
        for j, link in enumerate(self):
            s += repr(link) + ","
        s += "],\n"
        s += f"    T0=SE3({np.array_repr(self.T0.A)}),\n"
        s += f"    name={self.name},\n"
        s += ")"
        return s

    def _repr_pretty_(self, p, cycle):
        """
        Pretty string for IPython (superclass method)

        :param p: pretty printer handle (ignored)
        :param cycle: pretty printer flag (ignored)

        """
        # see https://ipython.org/ipython-doc/stable/api/generated/IPython.lib.pretty.html

        p.text(f"{i}:\n{str(x)}")

    def nbranches(self):
        return 0

    def fkine(self, q):
        """
        Forward kinematics

        :param q: joint configuration
        :type q: array_like(n)
        :return: end effector pose
        :rtype: SE3
        """
        T = None
        for link, qk in zip(self, q):
            if T is None:
                T = link.S.exp(qk)
            else:
                T *= link.S.exp(qk)

        return T * self.T0

        # T = reduce(lambda x, y: x * y,
        #           [link.A(qk) for link, qk in zip(self, q)])

    def jacob0(self, q):
        """
        Jacobian in world frame

        :param q: joint configuration
        :type q: array_like(n)
        :return: Jacobian matrix
        :rtype: ndarray(6,n)
        """
        columns = []
        T = SE3()
        for link, qk in zip(self, q):
            columns.append(T.Ad() @ link.S.S)
            T *= link.S.exp(qk)
        T *= self.T0
        J = np.column_stack(columns)

        # convert Jacobian from velocity twist to spatial velocity
        Jsv = np.eye(6)
        Jsv[:3, 3:] = -skew(T.t)
        return Jsv @ J

    def jacobe(self, q):
        """
        Jacobian in end-effector frame

        :param q: joint configuration
        :type q: array_like(n)
        :return: Jacobian matrix
        :rtype: ndarray(6,n)
        """
        columns = []
        T = SE3()
        for link, qk in zip(self, q):
            columns.append(T.Ad() @ link.S.S)
            T *= link.S.exp(qk)
        T *= self.T0
        J = np.column_stack(columns)

        # convert velocity twist from world frame to EE frame
        return T.inv().Ad() @ J

    def ets2(self) -> ETS:
        """
        Generate ETS from robot screw axis and tool frame.

        This method generates ETS for a robot that is given as PoERobot class in 3D.
        It overrides the default ets() method from Robot class. It works for both
        revolute and prismatic links.

        Example
        --------
        Example description

        .. runblock:: pycon
        >>> from spatialmath import SE3
        >>> from roboticstoolbox import PoERobot, PoERevolute
        >>> link1 = PoERevolute([0, 0, 1], [0, 0, 0])
        >>> link2 = PoERevolute([0, 0, 1], [1, 0, 0])
        >>> TE0 = SE3.Tx(2)
        >>> r = PoERobot([link1, link2], TE0)
        >>> ets = r.ets()

        References
        ----------
        - D. Huczala et al. "An Automated Conversion Between Selected Robot
        Kinematic Representations," 2022 ICCMA, Luxembourg, 2022,
        doi: 10.1109/ICCMA56665.2022.10011595.

        """
        ets = ETS()
        for i in range(self.n):
            ets *= self.links[i].ets

        return ets

    def _update_ets(self):
        # initialize transformations between joints from joint 1 to ee, related to
        # the base frame
        tf_base = [SE3()] * (self.n + 1)
        for i in range(self.n + 1):
            tf_base[i] = self.links[i].ets.fkine([0])
        tf_base.append(self.T0)

        # get partial transforms
        tf_partial = [SE3()] * (self.n + 2)
        for i in reversed(range(1, self.n + 2)):
            tf_partial[i] = tf_base[i - 1].inv() * tf_base[i]

        # prepare ET sequence
        for i, tf in enumerate(tf_partial):
            et = []
            # XYZ parameters
            if np.around(tf.t[0], 4) != 0.0:
                et.append(ET.tx(tf.t[0]))
            if np.around(tf.t[1], 4) != 0.0:
                et.append(ET.ty(tf.t[1]))
            if np.around(tf.t[2], 4) != 0.0:
                et.append(ET.tz(tf.t[2]))

            # RPY parameters, due to RPY convention the order of is reversed
            rpy = tf.rpy()
            if np.around(rpy[2], 4) != 0.0:
                et.append(ET.Rz(rpy[2]))
            if np.around(rpy[1], 4) != 0.0:
                et.append(ET.Ry(rpy[1]))
            if np.around(rpy[0], 4) != 0.0:
                et.append(ET.Rx(rpy[0]))

            # assign joint variable, if the frame is not base or tool frame
            if self.links[i].isrevolute:
                et.append(ET.Rz())
            elif self.links[i].isprismatic:
                et.append(ET.tz())

            link_ets = ETS()
            for e in et:
                link_ets *= e

            self.links[i].ets = link_ets

            if self.links[i].isrevolute:
                et.append(ET.Rz())
            elif self.links[i].isprismatic:
                et.append(ET.tz())


"""
                link_ets = ETS()
                for e in et:
                    link_ets *= e

                self.links[i].ets = link_ets

            else:  # if base_link or end-effector link
                link_ets = ETS()
                if et == []:
                    et = [ET.tx(0)]
                for e in et:
                    link_ets *= e

                self.links[i].ets = link_ets
"""








if __name__ == "__main__":  # pragma nocover
    T0 = SE3.Trans(2, 0, 0)

    # rotate about z-axis, through (0,0,0)
    link1 = PoERevolute([0, 0, 1], [0, 0, 0], name="foo")
    # rotate about z-axis, through (1,0,0)
    link2 = PoERevolute([0, 0, 1], [1, 0, 0])
    # end-effector pose when q=[0,0]
    TE0 = SE3.Trans(2, 0, 0)

    print(repr(link1))
    print(link1)

    robot = PoERobot([link1, link2], T0)

    q = [0, np.pi / 2]
    # q = [0, 0]

    # robot.fkine(q).printline()
    # print(robot.jacob0(q))
    # print(robot.jacobe(q))
    print(repr(robot))
    print(robot)
