from roboticstoolbox.robot.Robot2 import Robot2

class ERobot2(Robot2):

    def __init__(self,
            links,
            name='noname',
            manufacturer='',
            comment='',
            base=None,
            tool=None,
            gravity=None,
            keywords=(),
            symbolic=False):

        self.name = name
        self.manufacturer = manufacturer
        self.comment = comment
        self.symbolic = symbolic
        self.base = base
        self.tool = tool
        self._reach = None

        self._ets = links

    def __repr__(self):
        pass

# --------------------------------------------------------------------- #

    def plot(
            self, q, block=True, dt=0.05, limits=None,
            vellipse=False, fellipse=False,
            eeframe=True, name=False):
        """
        2D Graphical display and animation

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param dt: if q is a trajectory, this describes the delay in
            milliseconds between frames
        :type dt: int
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2, z1, z2]
        :type limits: ndarray(6)
        :param vellipse: (Plot Option) Plot the velocity ellipse at the
            end-effector
        :type vellipse: bool
        :param vellipse: (Plot Option) Plot the force ellipse at the
            end-effector
        :type vellipse: bool
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        :type eeframe: bool
        :param name: (Plot Option) Plot the name of the robot near its base
        :type name: bool

        :return: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        - ``robot.plot2(q)`` displays a 2D graphical view of a robot based on
          the kinematic model and the joint configuration ``q``. This is a
          stick figure polyline which joins the origins of the link coordinate
          frames. The plot will autoscale with an aspect ratio of 1.

        If ``q`` (m,n) representing a joint-space trajectory it will create an
        animation with a pause of ``dt`` seconds between each frame.

        .. note::
            - By default this method will block until the figure is dismissed.
              To avoid this set ``block=False``.
            - The polyline joins the origins of the link frames, but for
              some Denavit-Hartenberg models those frames may not actually
              be on the robot, ie. the lines to not neccessarily represent
              the links of the robot.

        :seealso: :func:`teach2`

        """

        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "2D Plotting of ERobot's not implemented yet")

        # Make an empty 2D figure
        env = self._get_graphical_backend('pyplot2')

        q = getmatrix(q, (None, 2))
        # q = getmatrix(q, (None, self.n))

        # Add the self to the figure in readonly mode
        if q.shape[0] == 1:
            env.launch(self.name + ' Plot', limits)
        else:
            env.launch(self.name + ' Trajectory Plot', limits)

        env.add(
            self, readonly=True,
            eeframe=eeframe, name=name)

        if vellipse:
            vell = self.vellipse(centre='ee')
            env.add(vell)

        if fellipse:
            fell = self.fellipse(centre='ee')
            env.add(fell)

        for qk in q:
            self.q = qk
            env.step()

        # Keep the plot open
        if block:           # pragma: no cover
            env.hold()

        return env
        
    def teach(
            self, q=None, block=True, limits=None,
            vellipse=False, fellipse=False, eeframe=True, name=False, backend='pyplot2'):
        '''
        2D Graphical teach pendant

        :param block: Block operation of the code and keep the figure open
        :type block: bool
        :param q: The joint configuration of the robot (Optional,
            if not supplied will use the stored q values).
        :type q: float ndarray(n)
        :param limits: Custom view limits for the plot. If not supplied will
            autoscale, [x1, x2, y1, y2]
        :type limits: array_like(4)
        :param vellipse: (Plot Option) Plot the velocity ellipse at the
            end-effector
        :type vellipse: bool
        :param vellipse: (Plot Option) Plot the force ellipse at the
            end-effector
        :type vellipse: bool
        :param eeframe: (Plot Option) Plot the end-effector coordinate frame
            at the location of the end-effector. Uses three arrows, red,
            green and blue to indicate the x, y, and z-axes.
        :type eeframe: bool
        :param name: (Plot Option) Plot the name of the robot near its base
        :type name: bool

        :return: A reference to the PyPlot object which controls the
            matplotlib figure
        :rtype: PyPlot

        - ``robot.teach2(q)`` creates a 2D matplotlib plot which allows the
          user to "drive" a graphical robot using a graphical slider panel.
          The robot's inital joint configuration is ``q``. The plot will
          autoscale with an aspect ratio of 1.

        - ``robot.teach2()`` as above except the robot's stored value of ``q``
          is used.

        .. note::
            - Program execution is blocked until the teach window is
              dismissed.  If ``block=False`` the method is non-blocking but
              you need to poll the window manager to ensure that the window
              remains responsive.
            - The slider limits are derived from the joint limit properties.
              If not set then:
                - For revolute joints they are assumed to be [-pi, +pi]
                - For prismatic joint they are assumed unknown and an error
                  occurs.
              If not set then
                - For revolute joints they are assumed to be [-pi, +pi]
                - For prismatic joint they are assumed unknown and an error
                  occurs.

        '''

        if isinstance(self, rtb.ERobot):  # pragma nocover
            raise NotImplementedError(
                "2D Plotting of ERobot's not implemented yet")

        if q is not None:
            self.q = q

        # Make an empty 3D figure
        env =  self._get_graphical_backend(backend)          

        # Add the robot to the figure in readonly mode
        env.launch('Teach ' + self.name, limits=limits)
        env.add(
            self, readonly=True,
            eeframe=eeframe, name=name)

        env._add_teach_panel(self)

        if limits is None:
            limits = np.r_[-1, 1, -1, 1] * self.reach * 1.5
            env.ax.set_xlim([limits[0], limits[1]])
            env.ax.set_ylim([limits[2], limits[3]])

        if vellipse:
            vell = self.vellipse(centre='ee', scale=0.5)
            env.add(vell)

        if fellipse:
            fell = self.fellipse(centre='ee')
            env.add(fell)

        # Keep the plot open
        if block:           # pragma: no cover
            env.hold()

        return env

    def jacob0(self, q):

        return self._ets.jacob0(q)

    def jacobe(self, q):

        return self._ets.jacobe(q)

    def fkine(self, q):

        return self._ets.eval(q)