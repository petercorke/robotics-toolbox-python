Backends
========

A *backend* is what draws a robot, and animates it.  The Toolbox talks to a backend through a small, standard interface, so the same program can show a robot in a Matplotlib window, in a notebook or in a web browser, and a new backend can be added without changing the Toolbox.  A backend could equally be a simulator or a physical robot, since all the Toolbox asks of it is to display the objects it is given, and to do something when time advances.

.. list-table::
    :header-rows: 1
    :widths: 14 18 30 20 18

    * - Name
      - Class
      - Display
      - Needs
      - Teach pendant
    * - ``"pyplot"``
      - :py:class:`~roboticstoolbox.backends.PyPlot.PyPlot`
      - 3D, in a Matplotlib window or a notebook
      - nothing extra
      - yes
    * - ``"pyplot2"``
      - :py:class:`~roboticstoolbox.backends.PyPlot.PyPlot2`
      - 2D, for planar robots, in Matplotlib
      - nothing extra
      - yes
    * - ``"swift"``
      - ``Swift``, from the `Swift <https://github.com/jhavl/swift>`_ package
      - 3D, in a web browser, with meshes
      - the ``swift`` extra, see :doc:`install`
      - yes

"Teach pendant" in this context means a set of joint configuration sliders that allow you to control the configuration of the displayed robot using the robot's :meth:`~roboticstoolbox.robot.Robot.Robot.teach` method.

Using a backend
---------------

Most of the time you don't use a backend directly: ``robot.plot(q)`` makes one, adds the robot to it and steps it.  Name the backend to choose one, or set the default for a robot or for all robots.

.. code-block:: python

    robot.plot(q, backend="swift")            # use this backend, this time
    robot.default_backend = "swift"           # the default for this robot
    rtb.Robot.set_default_backend("swift")    # the default for all robots

With no explicit choice the robot's own default is used, then the default for all robots.  If there is none, the Toolbox picks one: ``"pyplot"`` for a DH robot, ``"pyplot2"`` for a planar robot, and for any other robot ``"swift"`` if it has geometry to show and ``"pyplot"`` if not.  If Swift is not installed it falls back to ``"pyplot"``.  Swift cannot yet draw a DH robot.

``roboticstoolbox.backends.load_backend`` makes a backend from its name.  To use one directly, which is what ``robot.plot`` does for you:

.. code-block:: python

    import roboticstoolbox as rtb
    from roboticstoolbox.backends import load_backend

    robot = rtb.models.DH.Panda()
    backend = load_backend("pyplot")      # make a backend
    backend.launch()                      # start it with an empty scene
    backend.add(robot)                    # add the robot to the scene
    robot.q = robot.qr                    # set the configuration
    backend.step()                        # update the display
    backend.hold()                        # keep the window open

What a backend does
-------------------

The backend holds a *scene*: the robots and other objects that it has been asked to display.  It does not own them.  The user's robot object stays the source of truth, and the backend draws it, so that after ``robot.q = ...`` and ``step()`` the display agrees with the robot.

- **Starting and stopping.**  ``launch`` starts the display with an empty scene, which has to be done before anything is added, ``close`` shuts it down, ``reset`` returns to the state after ``launch``, and ``restart`` closes and relaunches.
- **The scene.**  ``add(obj)`` puts an object in the scene and returns an identifier, and ``remove(id)`` takes it out.  The object must be of a class that the backend knows how to draw, such as a robot, or an ellipsoid if the backend supports them.
- **Time.**  ``step(dt)`` advances the scene by ``dt`` seconds, which makes each robot act according to its control type (position, velocity, acceleration or torque) and updates the display.
- **Waiting.**  ``hold`` keeps the backend running after the script has finished, for as long as the window is open.
- **Optional features**, which a backend declares with two flags: ``supports_teach`` is true if it can show the interactive teach pendant of :meth:`~roboticstoolbox.robot.Robot.Robot.teach`, and ``supports_ellipse`` is true if it can draw the ellipsoids made by :meth:`~roboticstoolbox.robot.Robot.Robot.plot_ellipse`.

The contract
------------

This is written down as the :py:class:`~roboticstoolbox.backends.Connector.Connector` protocol.  A backend is *any* object that has these methods and flags, and does not have to inherit from it: Swift does not, since it does not depend on the Toolbox.  The PyPlot backends inherit from it, which supplies the default values of the flags and makes a missing method an error when the backend is made.  The arguments differ from one backend to another, so the methods are declared with ``*args, **kwargs``, and the details are in each backend.

.. autosummary::
    :nosignatures:

    ~roboticstoolbox.backends.Connector.Connector.launch
    ~roboticstoolbox.backends.Connector.Connector.close
    ~roboticstoolbox.backends.Connector.Connector.reset
    ~roboticstoolbox.backends.Connector.Connector.restart
    ~roboticstoolbox.backends.Connector.Connector.add
    ~roboticstoolbox.backends.Connector.Connector.remove
    ~roboticstoolbox.backends.Connector.Connector.step
    ~roboticstoolbox.backends.Connector.Connector.hold

.. autoclass:: roboticstoolbox.backends.Connector.Connector
    :members: launch, close, reset, restart, add, remove, step, hold, supports_teach, supports_ellipse
    :member-order: bysource

The backends
------------

PyPlot and PyPlot2
^^^^^^^^^^^^^^^^^^

Both use Matplotlib, which is a common and highly portable graphics library for Python but has relatively limited 3D capability.  ``PyPlot`` draws robots in 3D axes and is the default.  ``PyPlot2`` is its two-dimensional counterpart for planar robots.  In a notebook they work with the ``inline`` and ``ipympl`` backends of Matplotlib.

.. automodule:: roboticstoolbox.backends.PyPlot
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Swift
^^^^^

`Swift <https://github.com/jhavl/swift>`_ is a separate package which supports the backend contract, and is installed by the ``swift`` extra.  It shows the robot in a web browser, with its meshes, and the teach pendant as a panel of sliders.  It does not draw ellipsoids.

Swift is a web page served by the Python process, and :py:meth:`~roboticstoolbox.robot.Robot.Robot.plot` works out how to show it:

* In a script it opens a browser window.  ``browser="chrome"``, ``"firefox"`` or ``"safari"`` on macOS, ``"google-chrome"`` or ``"chromium"`` on Linux, or a command line with ``%s`` for any other browser, chooses the browser; the names are those of Python's `webbrowser <https://docs.python.org/3/library/webbrowser.html>`_ module, which depend on the operating system, and ``webbrowser.get(name)`` tells you whether one works on your machine.  The default is the system's browser.  See :py:meth:`swift.Swift.Swift.launch`.
* In a notebook it is shown in a frame of the cell, so the same ``robot.plot(q)`` works whichever you use.  ``notebook=True`` asks for this explicitly, and ``notebook=False`` for a window even in a notebook.  The browser that shows the notebook must be on the machine running the kernel, since the frame points at ``localhost``.  ``browser="notebook"``, the old way, is deprecated.
* A trajectory is animated in real time, one step to each ``dt``.  Swift draws at most 60 frames per second and skips a step that comes sooner, so ``realtime=False`` would show only the first frame of a trajectory.

.. versionchanged:: 1.5.0
    In a notebook Swift is shown in the cell and not in a browser window, and a trajectory is animated.  ``notebook=`` is new.
