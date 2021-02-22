#!/usr/bin/env python
"""
@author Jesse Haviland
"""

from roboticstoolbox.backends.Connector import Connector
import roboticstoolbox as rp
import numpy as np
import spatialmath as sm
import time
from queue import Queue
import json
from abc import ABC, abstractmethod
from functools import wraps

_sw = None
sw = None

SHAPE_ADD = 10000

def _import_swift():     # pragma nocover
    import importlib
    global sw
    try:
        sw = importlib.import_module('swift')
        # from swift import start_servers
    except ImportError:
        print(
            '\nYou must install the python package swift, see '
            'https://github.com/jhavl/swift\n')
        raise


class Swift(Connector):  # pragma nocover
    """
    Graphical backend using Swift

    Swift is a web app built on three.js. It supports many 3D graphical
    primitives including meshes, boxes, ellipsoids and lines. It can render
    Collada objects in full color.

    :param realtime: Force the simulator to display no faster than real time,
        note that it may still run slower due to complexity
    :type realtime: bool
    :param display: Do not launch the graphical front-end of the simulator.
        Will still simulate the robot. Runs faster due to not needing to
        display anything.
    :type display: bool

    Example:

    .. code-block:: python
        :linenos:

        import roboticstoolbox as rtb

        robot = rtb.models.DH.Panda()  # create a robot

        pyplot = rtb.backends.Swift()   # create a Swift backend
        pyplot.add(robot)              # add the robot to the backend
        robot.q = robot.qz             # set the robot configuration
        pyplot.step()                  # update the backend and graphical view

    :references:

        - https://github.com/jhavl/swift

    """
    def __init__(self, realtime=True, display=True):
        super(Swift, self).__init__()

        self.sim_time = 0.0
        self.robots = []
        self.shapes = []
        self.outq = Queue()
        self.inq = Queue()

        # Number of custom html elements added to page for id purposes
        self.elementid = 0

        # Element dict which holds the callback functions for form updates
        self.elements = {}

        self.realtime = realtime
        self.display = display

        self.recording = False

        if self.display and sw is None:
            _import_swift()

    #
    #  Basic methods to do with the state of the external program
    #

    def launch(self, browser=None):
        """
        Launch a graphical backend in Swift by default in the default browser
        or in the specified browser

        :param browser: browser to open in: one of
            'google-chrome', 'chrome', 'firefox', 'safari', 'opera'
            or see for full list
            https://docs.python.org/3.8/library/webbrowser.html#webbrowser.open_new
        :type browser: string

        ``env = launch(args)`` create a 3D scene in a running Swift instance as
        defined by args, and returns a reference to the backend.

        """

        super().launch()

        if self.display:
            sw.start_servers(self.outq, self.inq, browser=browser)
            self.last_time = time.time()

    def step(self, dt=0.05, render=True):
        """
        Update the graphical scene

        :param dt: time step in seconds, defaults to 0.05
        :type dt: int, optional
        :param render: render the change in Swift. If True, this updates the
            pose of the simulated robots and objects in Swift.
        :type dt: bool, optional

        ``env.step(args)`` triggers an update of the 3D scene in the Swift
        window referenced by ``env``.

        .. note::

            - Each robot in the scene is updated based on
              their control type (position, velocity, acceleration, or torque).
            - Upon acting, the other three of the four control types will be
              updated in the internal state of the robot object.
            - The control type is defined by the robot object, and not all
              robot objects support all control types.
            - Execution is blocked for the specified interval

        """

        # TODO how is the pose of shapes updated prior to step?

        super().step

        self._step_robots(dt)
        self._step_shapes(dt)

        # Adjust sim time
        self.sim_time += dt

        # Send updated sim time to Swift
        if self.display:

            # Only need to do GUI stuff if self.display is True
            # Process GUI events
            self.process_events()

            # Step through user GUI changes
            self._step_elements()

            # If realtime is set, delay progress if we are running too quickly
            if self.realtime:
                time_taken = (time.time() - self.last_time)
                diff = dt - time_taken

                if diff > 0:
                    time.sleep(diff)

                self.last_time = time.time()

            if render:
                self._draw_all()
            else:
                for i in range(len(self.robots)):
                    self.robots[i]['ob'].fkine_all(self.robots[i]['ob'].q)

            self._send_socket('sim_time', self.sim_time)

    def reset(self):
        """
        Reset the graphical scene

        ``env.reset()`` triggers a reset of the 3D scene in the Swift window
        referenced by ``env``. It is restored to the original state defined by
        ``launch()``.

        """

        super().reset

    def restart(self):
        """
        Restart the graphics display

        ``env.restart()`` triggers a restart of the Swift view referenced by
        ``env``. It is closed and relaunched to the original state defined by
        ``launch()``.

        """

        super().restart

    def close(self):
        """
        Close the graphics display

        ``env.close()`` gracefully disconnectes from the Swift visualizer
        referenced by ``env``.
        """

        super().close()

    #
    #  Methods to interface with the robots created in other environemnts
    #

    def add(
            self, ob, show_robot=True, show_collision=False,
            readonly=False):
        """
        Add a robot to the graphical scene

        :param ob: the object to add
        :type ob: Robot or Shape
        :param show_robot: Show the robot visual geometry,
            defaults to True
        :type show_robot: bool, optional
        :param show_collision: Show the collision geometry,
            defaults to False
        :type show_collision: bool, optional
        :return: object id within visualizer
        :rtype: int
        :param readonly: If true, swif twill not modify any robot attributes,
            the robot is only nbeing displayed, not simulated,
            defaults to False
        :type readonly: bool, optional

        ``id = env.add(robot)`` adds the ``robot`` to the graphical
            environment.

        .. note::

            - Adds the robot object to a list of robots which will be updated
              when the ``step()`` method is called.

        """
        # id = add(robot) adds the robot to the external environment. robot
        # must be of an appropriate class. This adds a robot object to a
        # list of robots which will act upon the step() method being called.

        super().add()

        if isinstance(ob, rp.ERobot):
            robot = ob.to_dict()
            robot['show_robot'] = show_robot
            robot['show_collision'] = show_collision

            robot_object = {
                'ob': ob,
                'readonly': readonly
            }

            if self.display:
                id = self._send_socket('robot', robot)

                loaded = 0
                while loaded == 0:
                    loaded = int(self._send_socket('is_loaded', id))
                    time.sleep(0.1)
            else:
                id = len(self.robots)

            self.robots.append(robot_object)
            return int(id)
        elif isinstance(ob, rp.Shape):
            shape = ob.to_dict()
            if self.display:
                id = int(self._send_socket('shape', shape)) + SHAPE_ADD
            else:
                id = len(self.shapes) + SHAPE_ADD
            self.shapes.append(ob)
            return int(id)
        elif isinstance(ob, rp.backends.SwiftElement):

            if ob._added_to_swift:
                raise ValueError(
                    "This element has already been added to Swift")

            ob._added_to_swift = True

            id = 'customelement' + str(self.elementid)
            self.elementid += 1
            self.elements[id] = ob
            ob._id = id

            self._send_socket(
                'add_element',
                ob.to_dict())

    def remove(self, id):
        """
        Remove a robot/shape from the graphical scene

        ``env.remove(robot)`` removes the ``robot`` from the graphical
            environment.

        :param id: the id of the object as returned by the ``add`` method,
            or the instance of the object
        :type id: Int, Robot or Shape
        """

        super().remove()

        # ob to remove
        idd = None
        code = None

        if isinstance(id, rp.ERobot):

            for i in range(len(self.robots)):
                if self.robots[i] is not None and id == self.robots[i]['ob']:
                    idd = i
                    code = 'remove_robot'
                    self.robots[idd] = None
                    break

        elif isinstance(id, rp.Shape):

            for i in range(len(self.shapes)):
                if self.shapes[i] is not None and id == self.shapes[i]:
                    idd = i
                    code = 'remove_shape'
                    self.shapes[idd] = None
                    break

        elif id >= SHAPE_ADD:
            # Number corresponding to shape ID
            idd = id - SHAPE_ADD
            code = 'remove_shape'
            self.shapes[idd] = None
        else:
            # Number corresponding to robot ID
            idd = id
            code = 'remove_robot'
            self.robots[idd] = None

        if idd is None:
            raise ValueError(
                'the id argument does not correspond with '
                'a robot or shape in Swift')

        self._send_socket(code, idd)

    def hold(self):           # pragma: no cover
        '''
        hold() keeps the browser tab open i.e. stops the browser tab from
        closing once the main script has finished.

        '''

        while True:
            time.sleep(1)

    def start_recording(self, file_name, framerate, format='webm'):
        """
        Start recording the canvas in the Swift simulator

        :param file_name: The file name for which the video will be saved as
        :type file_name: string
        :param framerate: The framerate of the video - to be timed correctly,
            this should equalt 1 / dt where dt is the time supplied to the
            step function
        :type framerate: float
        :param format: This is the format of the video, one of 'webm', 'gif',
            'png', or 'jpg'
        :type format: string

        ``env.start_recording(file_name)`` starts recording the simulation
            scene and will save it as file_name once
            ``env.start_recording(file_name)`` is called
        """

        valid_formats = ['webm', 'gif', 'png', 'jpg']

        if format not in valid_formats:
            raise ValueError(
                "Format can one of 'webm', 'gif', 'png', or 'jpg'")

        if not self.recording:
            self._send_socket(
                'start_recording', [framerate, file_name, format])
            self.recording = True
        else:
            raise ValueError(
                "You are already recording, you can only record one video"
                " at a time")

    def stop_recording(self):
        """
        Start recording the canvas in the Swift simulator. This is optional
        as the video will be automatically saved when the python script exits

        ``env.stop_recording()`` stops the recording of the simulation, can
            only be called after ``env.start_recording(file_name)``
        """

        if self.recording:
            self._send_socket('stop_recording')
        else:
            raise ValueError(
                "You must call swift.start_recording(file_name) before trying"
                " to stop the recording")

    def process_events(self):
        """
        Process the event queue from Swift, this invokes the callback functions
        from custom elements added to the page. If using custom elements
        (for example `add_slider`), use this function in your event loop to
        process updates from Swift.
        """
        events = self._send_socket('check_elements')
        events = json.loads(events)

        for event in events:
            self.elements[event].cb(events[event])

    def _step_robots(self, dt):

        for robot_object in self.robots:
            if robot_object is not None:
                robot = robot_object['ob']

                if robot_object['readonly'] or robot.control_type == 'p':
                    pass            # pragma: no cover

                elif robot.control_type == 'v':

                    for i in range(robot.n):
                        robot.q[i] += robot.qd[i] * (dt)

                        if np.any(robot.qlim[:, i] != 0) and \
                                not np.any(np.isnan(robot.qlim[:, i])):
                            robot.q[i] = np.min([robot.q[i], robot.qlim[1, i]])
                            robot.q[i] = np.max([robot.q[i], robot.qlim[0, i]])

                elif robot.control_type == 'a':
                    pass

                else:            # pragma: no cover
                    # Should be impossible to reach
                    raise ValueError(
                        'Invalid robot.control_type. '
                        'Must be one of \'p\', \'v\', or \'a\'')

    def _step_shapes(self, dt):

        for shape in self.shapes:
            if shape is not None:

                T = shape.base
                t = T.t.astype('float64')
                t += shape.v[:3] * dt

                R = sm.SO3(T.R)
                Rdelta = sm.SO3.EulerVec(shape.v[3:] * dt)
                R = Rdelta * R
                R = R.norm()  # renormalize to avoid numerical issues

                shape.base = sm.SE3.SO3(R, t=t)

    def _step_elements(self):
        """
        Check custom HTML elements to see if any have been updated, if there
        are any updates, send them through to Swift.
        """

        for element in self.elements:
            if self.elements[element]._changed:
                self.elements[element]._changed = False
                self._send_socket(
                    'update_element',
                    self.elements[element].to_dict())

    def _draw_all(self):

        for i in range(len(self.robots)):
            if self.robots[i] is not None:
                self._send_socket(
                    'robot_poses', [i, self.robots[i]['ob'].fk_dict()])

        for i in range(len(self.shapes)):
            if self.shapes[i] is not None:
                self._send_socket(
                    'shape_poses', [i, self.shapes[i].fk_dict()])

    def _send_socket(self, code, data=None):
        msg = [code, data]

        self.outq.put(msg)
        return self.inq.get()


class SwiftElement(ABC):
    """
    A basic super class for HTML elements which can be added to Swift

    """

    def __init__(self):

        self._id = None
        self._added_to_swift = False
        self._changed = False

        super().__init__()

    def _update(func):   # pragma nocover
        @wraps(func)
        def wrapper_update(*args, **kwargs):

            if args[0]._added_to_swift:
                args[0]._changed = True

            return func(*args, **kwargs)
        return wrapper_update

    @abstractmethod
    def to_dict(self):
        '''
        Outputs the element in dictionary form

        '''

        pass


class Slider(SwiftElement):
    """
    Create a range-slider html element

    :param cb: A callback function which is executed when the value of the
        slider changes. The callback should accept one argument which
        represents the new value of the slider
    :type cb: function
    :param min: the minimum value of the slider, optional
    :type min: float
    :param max: the maximum value of the slider, optional
    :type max: float
    :param step: the step size of the slider, optional
    :type step: float
    :param desc: add a description of the slider, optional
    :type desc: str
    :param unit: add a unit to the slider value, optional
    :type unit: str

    """

    def __init__(self, cb, min=0, max=100, step=1, value=0, desc='', unit=''):
        super(Slider, self).__init__()

        self._element = 'slider'
        self.cb = cb
        self.min = min
        self.max = max
        self.step = step
        self.value = value
        self.desc = desc
        self.unit = unit

    @property
    def cb(self):
        return self._cb

    @cb.setter
    @SwiftElement._update
    def cb(self, value):
        self._cb = value

    @property
    def min(self):
        return self._min

    @min.setter
    @SwiftElement._update
    def min(self, value):
        self._min = float(value)

    @property
    def max(self):
        return self._max

    @max.setter
    @SwiftElement._update
    def max(self, value):
        self._max = float(value)

    @property
    def step(self):
        return self._step

    @step.setter
    @SwiftElement._update
    def step(self, value):
        self._step = float(value)

    @property
    def value(self):
        return self._value

    @value.setter
    @SwiftElement._update
    def value(self, value):
        self._value = float(value)

    @property
    def desc(self):
        return self._desc

    @desc.setter
    @SwiftElement._update
    def desc(self, value):
        self._desc = value

    @property
    def unit(self):
        return self._unit

    @unit.setter
    @SwiftElement._update
    def unit(self, value):
        self._unit = value

    def to_dict(self):
        return {
            'element': self._element,
            'id': self._id,
            'min': self.min,
            'max': self.max,
            'step': self.step,
            'value': self.value,
            'desc': self.desc,
            'unit': self.unit
        }


class Label(SwiftElement):
    """
    Create a Label html element

    :param desc: the value of the label, optional
    :type desc: str
    """

    def __init__(self, desc=''):
        super(Label, self).__init__()

        self._element = 'label'
        self.desc = desc

    @property
    def desc(self):
        return self._desc

    @desc.setter
    @SwiftElement._update
    def desc(self, value):
        self._desc = value

    def to_dict(self):
        return {
            'element': self._element,
            'id': self._id,
            'desc': self.desc
        }


class Button(SwiftElement):
    """
    Create a Button html element

    :param cb: A callback function which is executed when the button is
        clicked. The callback should accept one argument which
        can be disregarded
    :type cb: function
    :param desc: text written on the button, optional
    :type desc: str
    """

    def __init__(self, cb, desc=''):
        super(Button, self).__init__()

        self._element = 'button'
        self.cb = cb
        self.desc = desc

    @property
    def cb(self):
        return self._cb

    @cb.setter
    @SwiftElement._update
    def cb(self, value):
        self._cb = value

    @property
    def desc(self):
        return self._desc

    @desc.setter
    @SwiftElement._update
    def desc(self, value):
        self._desc = value

    def to_dict(self):
        return {
            'element': self._element,
            'id': self._id,
            'desc': self.desc
        }


class Select(SwiftElement):
    """
    Create a Select element, used to create a drop-down list.

    :param cb: A callback function which is executed when the value select
        box changes. The callback should accept one argument which
        represents the index of the new value
    :type cb: function
    :param desc: add a description of the select box, optional
    :type desc: str
    :param options: represent the options inside the select box, optional
    :type options: List of str
    :param value: the index of the initial selection of the select
        box, optional
    :type value: int
    """

    def __init__(self, cb, desc='', options=[], value=0):
        super(Select, self).__init__()

        self._element = 'select'
        self.cb = cb
        self.desc = desc
        self.options = options
        self.value = value

    @property
    def cb(self):
        return self._cb

    @cb.setter
    @SwiftElement._update
    def cb(self, value):
        self._cb = value

    @property
    def desc(self):
        return self._desc

    @desc.setter
    @SwiftElement._update
    def desc(self, value):
        self._desc = value

    @property
    def options(self):
        return self._options

    @options.setter
    @SwiftElement._update
    def options(self, value):
        self._options = value

    @property
    def value(self):
        return self._value

    @value.setter
    @SwiftElement._update
    def value(self, nvalue):
        self._value = nvalue

    def to_dict(self):
        return {
            'element': self._element,
            'id': self._id,
            'desc': self.desc,
            'options': self.options,
            'value': self.value
        }


class Checkbox(SwiftElement):
    """
    Create a checkbox element, used to create multi-selection list.

    :param cb: A callback function which is executed when a box is checked.
        The callback should accept one argument which represents a List of
        bool representing the checked state of each box
    :type cb: function
    :param desc: add a description of the checkboxes, optional
    :type desc: str
    :param options: represents the checkboxes, optional
    :type options: List of str
    :param checked: a List represented boxes initially checked
    :type checked: List of bool
    """

    def __init__(self, cb, desc='', options=[], checked=[]):
        super(Checkbox, self).__init__()

        self._element = 'checkbox'
        self.cb = cb
        self.desc = desc
        self.options = options
        self.checked = checked

    @property
    def cb(self):
        return self._cb

    @cb.setter
    @SwiftElement._update
    def cb(self, value):
        self._cb = value

    @property
    def desc(self):
        return self._desc

    @desc.setter
    @SwiftElement._update
    def desc(self, value):
        self._desc = value

    @property
    def options(self):
        return self._options

    @options.setter
    @SwiftElement._update
    def options(self, value):
        self._options = value

    @property
    def checked(self):
        return self._checked

    @checked.setter
    @SwiftElement._update
    def checked(self, value):
        self._checked = value

    def to_dict(self):
        return {
            'element': self._element,
            'id': self._id,
            'desc': self.desc,
            'options': self.options,
            'checked': self.checked
        }


class Radio(SwiftElement):
    """
    Create a radio element, used to create single-selection list.

    :param cb: A callback function which is executed when a radio is checked.
        The callback should accept one argument which represents a index
        corresponding to the checked radio button
    :type cb: function
    :param desc: add a description of the radio buttons, optional
    :type desc: str
    :param options: represents the radio buttons, optional
    :type options: List of str
    :param checked: the initial radio button checked, optional
    :type checked: int
    """

    def __init__(self, cb, desc='', options=[], checked=[]):
        super(Radio, self).__init__()

        self._element = 'radio'
        self.cb = cb
        self.desc = desc
        self.options = options
        self.checked = checked

    @property
    def cb(self):
        return self._cb

    @cb.setter
    @SwiftElement._update
    def cb(self, value):
        self._cb = value

    @property
    def desc(self):
        return self._desc

    @desc.setter
    @SwiftElement._update
    def desc(self, value):
        self._desc = value

    @property
    def options(self):
        return self._options

    @options.setter
    @SwiftElement._update
    def options(self, value):
        self._options = value

    @property
    def checked(self):
        return self._checked

    @checked.setter
    @SwiftElement._update
    def checked(self, value):
        self._checked = value

    def to_dict(self):
        return {
            'element': self._element,
            'id': self._id,
            'desc': self.desc,
            'options': self.options,
            'checked': self.checked
        }
