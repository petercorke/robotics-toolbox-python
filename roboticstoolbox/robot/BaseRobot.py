from abc import ABC

class BaseRobot(ABC):

    # TODO probably should be a static method
    def _get_graphical_backend(self, backend):
        #
        # find the right backend, modules are imported here on an as needs basis
        if backend.lower() == 'swift':  # pragma nocover
            if isinstance(self, rtb.DHRobot):
                raise NotImplementedError(
                    'Plotting in Swift is not implemented for DHRobots yet')

            from roboticstoolbox.backends.Swift import Swift
            env = Swift()

        elif backend.lower() == 'pyplot':
            from roboticstoolbox.backends.PyPlot import PyPlot
            env = PyPlot()

        elif backend.lower() == 'pyplot2':
            from roboticstoolbox.backends.PyPlot import PyPlot2
            env = PyPlot2()

        elif backend.lower() == 'vpython':
            from roboticstoolbox.backends.VPython import VPython
            env = VPython()

        else:
            raise ValueError('unknown backend', backend)

        return env