# PoERobot
import numpy as np
from spatialmath import Twist3, SE3
from spatialmath.base import skew
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
        ets = self._ets_world(twist)

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

    def _ets_world(self, twist) -> ETS:
        # initialize a transformation that will represent the twist
        twist_tf = SE3()
        # get screw axis components
        w = twist.w
        v = twist.v

        if isinstance(self, PoEPrismatic):
            # get the axis directional vector component
            a_vec = v
            # get vector of the x-axis
            n_vec = twist_tf.n
            # point on screw axis set as origin
            t_vec = twist_tf.t

        elif isinstance(self, PoERevolute):
            # get the nearest point of the screw axis closest to the origin
            principal_point = np.cross(w, v)

            # get vector of the x-axis
            if np.isclose(np.linalg.norm(principal_point), 0.0):  # the points are
                # coincident
                n_vec = twist_tf.n
            else:
                n_vec = principal_point / np.linalg.norm(principal_point)

            # get the axis directional vector component
            a_vec = w
            # point on screw axis
            t_vec = principal_point

        else:  # not a joint
            n_vec = twist_tf.n
            a_vec = twist_tf.a
            t_vec = v

        # construct transformation matrix from obtained vectors
        twist_tf = SE3.OA(np.cross(a_vec, n_vec), a_vec)
        twist_tf.t = t_vec
        # get RPY parameters
        rpy = twist_tf.rpy()

        # prepare list of ETs, due to RPY convention the RPY order is reversed
        et_list = [ET.tx(twist_tf.t[0]), ET.ty(twist_tf.t[1]), ET.tz(twist_tf.t[2]),
                   ET.Rz(rpy[2]), ET.Ry(rpy[1]), ET.Rx(rpy[0])]
        # remove ETs with empty transform
        et_list = [et for et in et_list if not np.isclose(et.eta, 0.0)]

        # assign joint variable, if the frame is not base or tool frame
        if isinstance(self, PoEPrismatic):
            et_list.append(ET.tz())
        elif isinstance(self, PoERevolute):
            et_list.append(ET.Rz())

        return ETS(et_list)


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

        # update ETS according to the given links order
        # (in PoELink their ETS is in relation to base frame, not previous joint's ETS)
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
        for i in range(self.n):
            if T is None:
                T = self.links[i+1].S.exp(q[i])
            else:
                T *= self.links[i+1].S.exp(q[i])

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
        for i in range(self.n):
            columns.append(T.Ad() @ self.links[i+1].S.S)
            T *= self.links[i+1].S.exp(q[i])
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
        for i in range(self.n):
            columns.append(T.Ad() @ self.links[i + 1].S.S)
            T *= self.links[i + 1].S.exp(q[i])
        T *= self.T0
        J = np.column_stack(columns)

        # convert velocity twist from world frame to EE frame
        return T.inv().Ad() @ J

    def _update_ets(self):
        # initialize transformations between joints from joint 1 to ee, related to
        # the world (base) frame
        twist_tf_world = [SE3(link.Ts) for link in self.links]
        # update the end-effector since its twist can provide transform
        # with different x and y -axes
        twist_tf_world[-1] = self.T0

        # get partial transforms between links
        twist_tf_partial = [SE3()] * (self.n + 2)
        for i in reversed(range(1, self.n + 2)):
            twist_tf_partial[i] = twist_tf_world[i - 1].inv() * twist_tf_world[i]

        # prepare ET sequence
        for i, tf in enumerate(twist_tf_partial):
            # get RPY parameters
            rpy = tf.rpy()

            # prepare list of ETs, due to RPY convention the RPY order is reversed
            et_list = [ET.tx(tf.t[0]), ET.ty(tf.t[1]), ET.tz(tf.t[2]),
                       ET.Rz(rpy[2]), ET.Ry(rpy[1]), ET.Rx(rpy[0])]
            # remove ETs with empty transform
            et_list = [et for et in et_list if not np.isclose(et.eta, 0.0)]

            # assign joint variable with corresponding index
            if self.links[i].isrevolute:
                et_list.append(ET.Rz(jindex=i-1))
            elif self.links[i].isprismatic:
                et_list.append(ET.tz(jindex=i-1))

            self.links[i].ets = ETS(et_list)

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
