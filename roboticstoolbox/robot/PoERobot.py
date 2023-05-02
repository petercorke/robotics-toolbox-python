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

        # initialize the link with its world frame ETS
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

    def _ets_world(self, twist: Twist3) -> ETS:
        """
        Convert twist to its corresponding ETS

        This method obtains an SE3 object that corresponds to a given twist and
        returns its ETS in the world frame.

        :param twist: given twist as Twist3 object

        References
        ----------
        - D. Huczala, T. Kot, J. Mlotek, J. Suder and M. Pfurner, *An Automated
          Conversion Between Selected Robot Kinematic Representations*, ICCMA 2022,
          Luxembourg, doi: 10.1109/ICCMA56665.2022.10011595

        """

        # set base of the robot to the origin
        base = SE3()
        # get screw axis components
        w = twist.w
        v = twist.v

        if isinstance(self, PoEPrismatic):
            # get the prismatic axis's directional vector component
            a_vec = v
            # vector of the x-axis (default to origin's n vector)
            n_vec = base.n
            # set point on screw axis to origin (prismatic joint has no moment)
            t_vec = base.t

        elif isinstance(self, PoERevolute):
            # get the nearest point of the screw axis closest to the origin
            principal_point = np.cross(w, v)

            # get vector of the x-axis
            if np.isclose(np.linalg.norm(principal_point), 0.0):  # the points are
                # coincident
                n_vec = base.n
            else:  # along the direction to principal point
                n_vec = principal_point / np.linalg.norm(principal_point)

            # get the revolute axis directional vector component
            a_vec = w
            # point on screw axis
            t_vec = principal_point

        else:  # not a joint
            n_vec = base.n
            a_vec = base.a
            t_vec = v

        o_vec = np.cross(a_vec, n_vec)

        # construct transform from obtained vectors
        twist_as_SE3 = SE3.OA(o_vec, a_vec)
        twist_as_SE3.t = t_vec
        # get RPY parameters
        rpy = twist_as_SE3.rpy()

        # prepare list of ETs, due to RPY convention the RPY order is reversed
        et_list = [ET.tx(twist_as_SE3.t[0]), ET.ty(twist_as_SE3.t[1]),
                   ET.tz(twist_as_SE3.t[2]), ET.Rz(rpy[2]), ET.Ry(rpy[1]), ET.Rx(rpy[0])]
        # remove ETs with empty transform
        et_list = [et for et in et_list if not np.isclose(et.eta, 0.0)]

        # assign joint variable at the end of list (if the frame is not base or tool
        # frame)
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

        # update ETS according to the given links order (in PoELink their ETS is
        # given WITH relation to base frame, NOT to previous joint's ETS)
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
        s = "PoERobot([\n"
        s += "\n".join(["    " + repr(link) + "," for link in self])
        s += "\n    ],\n"
        s += f"    T0=SE3({np.array_repr(self.T0.A)}),\n"
        s += f"    name=\"{self.name}\",\n"
        s += ")"
        return s

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
            columns.append(T.Ad() @ self.links[i+1].S.S)
            T *= self.links[i+1].S.exp(q[i])
        T *= self.T0
        J = np.column_stack(columns)

        # convert velocity twist from world frame to EE frame
        return T.inv().Ad() @ J

    def _update_ets(self):
        """
        Updates ETS of links when PoERobot is initialized according to joint order

        By default, PoE representation specifies twists in relation to the base frame
        of a robot. Since the PoELinks are initialized prior to PoERobot, their ETS is
        given in the base frame, not in relation between links. This method creates
        partial transforms between links and obtains new ETSs that respect the links
        order.
        """

        # initialize transformations between joints from joint 1 to ee, related to
        # the world (base) frame
        twist_as_SE3_world = [SE3(link.Ts) for link in self.links]

        # update the ee since its twist can provide transform with different x-, y-axes
        twist_as_SE3_world[-1] = self.T0

        # initialize partial transforms
        twist_as_SE3_partial = [SE3()] * (self.n + 2)

        # get partial transforms between links
        for i in reversed(range(1, self.n + 2)):
            twist_as_SE3_partial[i] = twist_as_SE3_world[i - 1].inv() * \
                                      twist_as_SE3_world[i]

        # prepare ET sequence
        for i, tf in enumerate(twist_as_SE3_partial):
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

            # update the ETS for given link
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
