#!/usr/bin/env python
"""
@author Jesse Haviland
"""

import roboticstoolbox as rp
import numpy as np
from roboticstoolbox.backends.Connector import Connector
from roboticstoolbox.backends.PyPlot.RobotPlot2 import RobotPlot2
from roboticstoolbox.backends.PyPlot.EllipsePlot import EllipsePlot
import time
import io
import sys
from spatialmath import SE2

_mpl = False
_ipy_display = None
_ipy_image = None
_ipy_svg = None
_ipy_clear_output = None

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    plt.style.use("ggplot")
    matplotlib.rcParams["font.size"] = 7
    matplotlib.rcParams["lines.linewidth"] = 0.5
    matplotlib.rcParams["xtick.major.size"] = 1.5
    matplotlib.rcParams["ytick.major.size"] = 1.5
    matplotlib.rcParams["axes.labelpad"] = 1
    plt.rc("grid", linestyle="-", color="#dbdbdb")
    _mpl = True
except ImportError:  # pragma nocover
    pass

try:
    from IPython.display import display as _ipy_display
    from IPython.display import Image as _ipy_image
    from IPython.display import SVG as _ipy_svg
    from IPython.display import clear_output as _ipy_clear_output
except ImportError:  # pragma nocover
    pass


class PyPlot2(Connector):
    def __init__(self):

        super(PyPlot2, self).__init__()
        self.robots = []
        self.ellipses = []
        self.sim_time = 0
        self.render_mode = "window"
        self.inline_every_n = 1
        self.inline_format = "svg"
        self.inline_dpi = None
        self._inline_step_count = 0
        self._inline_display_handle = None
        self._inline_is_jl = False

        if not _mpl:  # pragma nocover
            raise ImportError(
                "\n\nYou do not have matplotlib installed, do:\n"
                "pip install matplotlib\n\n"
            )

    def __repr__(self):
        s = ""
        for robot in self.robots:
            s += f"  robot: {robot.name}\n"
        for ellipse in self.ellipses:
            s += f"  ellipse: {ellipse}\n"

        if s == "":
            return f"PyPlot2D backend, t = {self.sim_time}, empty scene"
        else:
            return f"PyPlot2D backend, t = {self.sim_time}, scene:\n" + s

    def launch(self, name=None, limits=None, **kwargs):
        """
        env = launch() launchs a blank 2D matplotlib figure

        Optional keyword arguments
        --------------------------
        render_mode
            One of ``'window'``, ``'notebook-widget'``, or ``'notebook-inline'``.
        inline_every_n
            Push one inline frame every N simulation steps (inline mode only).
        inline_format
            Inline frame format: ``'svg'`` (default) or ``'png'``.
        inline_dpi
            DPI for PNG inline frames only. Ignored for SVG.

        """

        super().launch()

        self.render_mode = _resolve_render_mode(kwargs.get("render_mode"))
        self.inline_every_n = max(1, int(kwargs.get("inline_every_n", 1)))
        self.inline_format = kwargs.get("inline_format", "svg")
        if self.inline_format not in ["png", "svg"]:
            raise ValueError("inline_format must be either 'png' or 'svg'")
        inline_dpi = kwargs.get("inline_dpi", None)
        if inline_dpi is not None:
            inline_dpi = float(inline_dpi)
            if inline_dpi <= 0:
                raise ValueError("inline_dpi must be > 0")
        self.inline_dpi = inline_dpi
        self._inline_step_count = 0
        self._inline_display_handle = None
        self._inline_is_jl = sys.platform == "emscripten"

        labels = ["X", "Y"]

        if name is not None and not _isnotebook():
            # jupyter does weird stuff when figures have the same name
            self.fig = plt.figure()
        else:
            self.fig = plt.figure()

        # Create a 2D axes
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.ax.set_facecolor("white")
        plt.get_current_fig_manager().set_window_title(
            f"Robotics Toolbox for Python (Figure {self.ax.figure.number})"
        )

        self.ax.set_xbound(-0.5, 0.5)
        self.ax.set_ybound(-0.5, 0.5)

        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])

        self.ax.autoscale(enable=True, axis="both", tight=False)

        if limits is not None:
            self.ax.set_xlim([limits[0], limits[1]])
            self.ax.set_ylim([limits[2], limits[3]])

        self.ax.axis("equal")

        # In inline notebook mode (notably JupyterLite), keeping this figure
        # registered with pyplot can trigger repeated auto-display of blank
        # Figure reprs. Detach it and drive rendering via _push_inline_frame().
        if self.render_mode == "notebook-inline":
            plt.close(self.fig)

        if self.render_mode == "window":
            plt.ion()
            plt.show()
        else:
            if self.render_mode == "notebook-inline":
                plt.ioff()
            else:
                plt.ion()
            if self.render_mode != "notebook-inline":
                self.fig.canvas.draw()

        # Set the signal handler and a 0.1 second plot updater
        # signal.signal(signal.SIGALRM, self._plot_handler)
        # signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)

    def step(self, dt=0.05):
        """
        state = step(args) triggers the external program to make a time step
        of defined time updating the state of the environment as defined by
        the robot's actions.

        The will go through each robot in the list and make them act based on
        their control type (position, velocity, acceleration, or torque). Upon
        acting, the other three of the four control types will be updated in
        the internal state of the robot object. The control type is defined
        by the robot object, and not all robot objects support all control
        types.

        """

        super().step()

        self._step_robots(dt)

        # plt.ioff()
        self._draw_ellipses()
        self._draw_robots()
        # plt.ion()

        if self.render_mode == "window":
            plt.draw()
            plt.pause(dt)
        elif self.render_mode == "notebook-widget":
            plt.draw()
            self.fig.canvas.draw_idle()
            time.sleep(dt)
        else:
            if self._inline_step_count % self.inline_every_n == 0:
                self._push_inline_frame()
            self._inline_step_count += 1
            time.sleep(dt)

        self._update_robots()

    def reset(self):
        """
        state = reset() triggers the external program to reset to the
        original state defined by launch

        """

        super().reset()

    def restart(self):
        """
        state = restart() triggers the external program to close and relaunch
        to thestate defined by launch

        """

        super().restart()

    def close(self):
        """
        close() closes the plot

        """

        super().close()

        # signal.setitimer(signal.ITIMER_REAL, 0)
        plt.close(self.fig)

    #
    #  Methods to interface with the robots created in other environemnts
    #

    def add(self, ob, readonly=False, display=True, eeframe=True, name=False, **kwargs):
        """
        id = add(robot) adds the robot to the external environment. robot must
        be of an appropriate class. This adds a robot object to a list of
        robots which will act upon the step() method being called.

        """

        super().add()

        if isinstance(ob, rp.Robot2):
            self.robots.append(RobotPlot2(ob, self, readonly, display, eeframe, name))
            self.robots[len(self.robots) - 1].draw()

        elif isinstance(ob, EllipsePlot):
            ob.ax = self.ax
            self.ellipses.append(ob)
            self.ellipses[len(self.ellipses) - 1].draw2()

        if self.render_mode == "notebook-inline":
            # Push a frame immediately so static ellipse plots render without step().
            self._push_inline_frame()
        else:
            plt.draw()
            plt.show(block=False)

    def remove(self):
        """
        id = remove(robot) removes the robot to the external environment.

        """

        super().remove()  # ???

    def hold(self):  # pragma: no cover
        """
        hold() keeps the plot open i.e. stops the plot from closing once
        the main script has finished.

        """

        # signal.setitimer(signal.ITIMER_REAL, 0)
        plt.ioff()

        # keep stepping the environment while figure is open
        while True:
            if not plt.fignum_exists(self.fig.number):
                break
            self.step()

    #
    #  Private methods
    #

    def _step_robots(self, dt):

        for rpl in self.robots:
            robot = rpl.robot

            if rpl.readonly or robot.control_type == "p":
                pass  # pragma: no cover

            elif robot.control_type == "v":
                for i in range(robot.n):
                    robot.q[i] += robot.qd[i] * (dt / 1000)

            elif robot.control_type == "a":  # pragma: no cover
                pass

            else:  # pragma: no cover
                # Should be impossible to reach
                raise ValueError(
                    "Invalid robot.control_type. Must be one of 'p', 'v', or 'a'"
                )

    def _update_robots(self):
        pass

    def _draw_robots(self):

        for i in range(len(self.robots)):
            self.robots[i].draw()

    def _draw_ellipses(self):

        for i in range(len(self.ellipses)):
            self.ellipses[i].draw2()

    def _push_inline_frame(self):
        # Push a snapshot into notebook output for inline animation.
        if _ipy_display is None:
            return

        buf = io.BytesIO()
        if self.inline_format == "svg":
            if _ipy_svg is None:
                return
            self.fig.savefig(buf, format="svg")
            frame = _ipy_svg(data=buf.getvalue().decode("utf-8"))
        else:
            if _ipy_image is None:
                return
            self.fig.savefig(buf, format="png", dpi=self.inline_dpi)
            frame = _ipy_image(data=buf.getvalue())

        if self._inline_is_jl and _ipy_clear_output is not None:
            _ipy_clear_output(wait=True)
            _ipy_display(frame)
            return

        if self._inline_display_handle is None:
            self._inline_display_handle = _ipy_display(frame, display_id=True)
        elif hasattr(self._inline_display_handle, "update"):
            self._inline_display_handle.update(frame)
        else:
            _ipy_display(frame)

    # def _plot_handler(self, sig, frame):
    #     plt.pause(0.001)

    def _add_teach_panel(self, robot, q):
        """
        Add a teach panel

        :param robot: Robot being taught
        :type robot: ERobot class
        :param q: inital joint angles in radians
        :type q: array_like(n)
        """
        fig = self.fig

        # Add text to the plots
        def text_trans(text, q):  # pragma: no cover
            # update displayed robot pose value
            T = robot.fkine(q, end=robot.ee_links[0])
            t = np.round(T.t, 3)
            r = np.round(T.theta(), 3)
            text[0].set_text("x: {0}".format(t[0]))
            text[1].set_text("y: {0}".format(t[1]))
            text[2].set_text("yaw: {0}".format(r))

        # Update the self state in mpl and the text
        def update(val, text, robot):  # pragma: no cover
            for j in range(robot.n):
                if robot.isrevolute(j):
                    robot.q[j] = np.radians(self.sjoint[j].val)
                else:
                    robot.q[j] = self.sjoint[j].val

            teach_vellipse = getattr(self, "_teach_vellipse", None)
            if teach_vellipse is not None:
                teach_vellipse.q = robot.q

            teach_fellipse = getattr(self, "_teach_fellipse", None)
            if teach_fellipse is not None:
                teach_fellipse.q = robot.q

            text_trans(text, robot.q)

        fig.subplots_adjust(left=0.38)
        text = []

        x1 = 0.04
        x2 = 0.22
        yh = 0.04
        ym = 0.5 - (robot.n * yh) / 2 + 0.17 / 2

        self.axjoint = []
        self.sjoint = []

        qlim = robot.todegrees(robot.qlim)

        # Set the pose text
        # if multiple EE, display only the first one
        T = SE2(robot.fkine(q, end=robot.ee_links[0]))
        t = np.round(T.t, 3)
        r = np.round(T.theta(), 3)

        # TODO maybe put EE name in here, possible issue with DH robot
        # TODO maybe display pose of all EEs, layout hassles though

        if robot.nbranches == 0:
            header = "End-effector Pose"
        else:
            header = "End-effector #0 Pose"
        fig.text(
            0.02, 1 - ym + 0.25, header, fontsize=9, weight="bold", color="#4f4f4f"
        )
        text.append(
            fig.text(
                0.03, 1 - ym + 0.20, "x: {0}".format(t[0]), fontsize=9, color="#2b2b2b"
            )
        )
        text.append(
            fig.text(
                0.03, 1 - ym + 0.16, "y: {0}".format(t[1]), fontsize=9, color="#2b2b2b"
            )
        )
        text.append(
            fig.text(
                0.15, 1 - ym + 0.20, "yaw: {0}".format(r), fontsize=9, color="#2b2b2b"
            )
        )
        fig.text(
            0.02,
            1 - ym + 0.06,
            "Joint angles",
            fontsize=9,
            weight="bold",
            color="#4f4f4f",
        )

        for j in range(robot.n):
            # for each joint
            ymin = (1 - ym) - j * yh
            self.axjoint.append(fig.add_axes([x1, ymin, x2, 0.03], facecolor="#dbdbdb"))

            if robot.isrevolute(j):
                slider = Slider(
                    ax=self.axjoint[j],
                    label="q" + str(j),
                    valmin=qlim[0, j],
                    valmax=qlim[1, j],
                    valinit=np.degrees(q[j]),
                    valfmt="% .1f°",
                )
            else:
                slider = Slider(
                    ax=self.axjoint[j],
                    label="q" + str(j),
                    valmin=qlim[0, j],
                    valmax=qlim[1, j],
                    valinit=q[j],
                    valfmt="% .1f",
                )

            slider.on_changed(lambda x: update(x, text, robot))
            self.sjoint.append(slider)
        robot.q = q
        self.step()


def _isnotebook():
    """
    Determine if code is being run from a Jupyter notebook or JupyterLite

    ``_isnotebook`` is True if running Jupyter notebook, JupyterLite, or similar, else False

    :references:

        - https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-
        is-executed-in-the-ipython-notebook/39662359#39662359
    """
    import sys

    # Check if running in Pyodide/JupyterLite
    if sys.platform == "emscripten":
        return True

    # Fall back to checking IPython shell type
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def _resolve_render_mode(render_mode=None):
    if render_mode is not None:
        if render_mode not in ["window", "notebook-widget", "notebook-inline"]:
            raise ValueError(
                "render_mode must be one of 'window', "
                "'notebook-widget', or 'notebook-inline'"
            )
        return render_mode

    if not _isnotebook():
        return "window"

    backend = matplotlib.get_backend().lower()
    if "ipympl" in backend or "widget" in backend or "nbagg" in backend:
        return "notebook-widget"

    return "notebook-inline"
