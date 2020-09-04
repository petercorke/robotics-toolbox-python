"""
@author: Peter Corke
@author: Samuel Drew
"""

from roboticstoolbox.robot.serial_link import *
from roboticstoolbox.robot.Link import RevoluteDH
from math import pi
import numpy as np


class Panda(SerialLink):

    # %MDL_PANDA Create model of Franka-Emika PANDA robot
    # %
    # % MDL_PANDA is a script that creates the workspace variable panda which
    # % describes the kinematic characteristics of a Franka-Emika PANDA manipulator
    # % using standard DH conventions.
    # %
    # % Also define the workspace vectors:
    # %   qz         zero joint angle configuration
    # %   qr         arm along +ve x-axis configuration
    # %
    # % Reference::
    # % - http://www.diag.uniroma1.it/~deluca/rob1_en/WrittenExamsRob1/Robotics1_18.01.11.pdf
    # % - "Dynamic Identification of the Franka Emika Panda Robot With Retrieval of Feasible Parameters Using Penalty-Based Optimization"
    # %   C. Gaz, M. Cognetti, A. Oliva, P. Robuffo Giordano and A. De Luca
    # %   IEEE Robotics and Automation Letters 4(4), pp. 4147-4154, Oct. 2019, doi: 10.1109/LRA.2019.2931248
    # %
    # % Notes::
    # % - SI units of metres are used.
    # % - Unlike most other mdl_xxx scripts this one is actually a function that
    # %   behaves like a script and writes to the global workspace.
    # %
    # % See also mdl_sawyer, SerialLink.
    #
    # % MODEL: Franka-Emika, PANDA, 7DOF, standard_DH
    #
    # % Copyright (C) 1993-2018, by Peter I. Corke
    # %
    # % This file is part of The Robotics Toolbox for MATLAB (RTB).
    # %
    # % RTB is free software: you can redistribute it and/or modify
    # % it under the terms of the GNU Lesser General Public License as published by
    # % the Free Software Foundation, either version 3 of the License, or
    # % (at your option) any later version.
    # %
    # % RTB is distributed in the hope that it will be useful,
    # % but WITHOUT ANY WARRANTY; without even the implied warranty of
    # % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    # % GNU Lesser General Public License for more details.
    # %
    # % You should have received a copy of the GNU Leser General Public License
    # % along with RTB.  If not, see <http://www.gnu.org/licenses/>.
    # %
    # % http://www.petercorke.com
    def __init__(self):
        deg = pi/180

        # Define links (thanks Alex Smith for this code)
        L1 = RevoluteDH(a=0.0, d=0.333,
                        alpha=0.0,
                        qlim=[-2.8973, 2.8973],
                        m=4.970684,
                        r=[3.875e-03, 2.081e-03, 0],
                        I=[7.03370e-01, 7.06610e-01, 9.11700e-03, -1.39000e-04, 1.91690e-02, 6.77200e-03],
                        G=1,
                        mesh='FRANKA-EMIKA/Panda/link1.stl')
        L2 = RevoluteDH(a=0.0, d=0.0,
                        alpha=-pi/2,
                        qlim=[-1.7628, 1.7628],
                        m=0.646926,
                        r=[-3.141e-03, -2.872e-02, 3.495e-03],
                        I=[7.96200e-03, 2.81100e-02, 2.59950e-02, -3.92500e-03, 7.04000e-04, 1.02540e-02],
                        G=1,
                        mesh='FRANKA-EMIKA/Panda/link2.stl')
        L3 = RevoluteDH(a=0.0,
                        d=0.316,
                        alpha=pi/2,
                        qlim=[-2.8973, 2.8973],
                        m=3.228604,
                        r=[ 2.7518e-02, 3.9252e-02, -6.6502e-02],
                        I=[3.72420e-02, 3.61550e-02, 1.08300e-02, -4.76100e-03, -1.28050e-02, -1.13960e-02],
                        G=1,
                        mesh='FRANKA-EMIKA/Panda/link3.stl')
        L4 = RevoluteDH(a=0.0825,
                        d=0.0,
                        alpha=pi/2,
                        qlim=[-3.0718, -0.0698],
                        m=3.587895,
                        r=[-5.317e-02, 1.04419e-01, 2.7454e-02],
                        I=[2.58530e-02, 1.95520e-02, 2.83230e-02, 7.79600e-03, 8.64100e-03, -1.33200e-03],
                        G=1,
                        mesh='FRANKA-EMIKA/Panda/link4.stl')
        L5 = RevoluteDH(a=-0.0825,
                        d=0.384,
                        alpha=-pi/2,
                        qlim=[-2.8973, 2.8973],
                        m=1.225946,
                        r=[-1.1953e-02, 4.1065e-02, -3.8437e-02],
                        I=[3.55490e-02, 2.94740e-02, 8.62700e-03, -2.11700e-03, 2.29000e-04, -4.03700e-03],
                        G=1,
                        mesh='FRANKA-EMIKA/Panda/link5.stl')
        L6 = RevoluteDH(a=0.0,
                        d=0.0,
                        alpha=pi/2,
                        qlim=[-0.0175, 3.7525],
                        m=1.666555,
                        r=[6.0149e-02, -1.4117e-02, -1.0517e-02],
                        I=[1.96400e-03, 4.35400e-03, 5.43300e-03, 1.09000e-04, 3.41000e-04, -1.15800e-03],
                        G=1,
                        mesh='FRANKA-EMIKA/Panda/link6.stl')
        L7 = RevoluteDH(a=0.088,
                        d=0.0,
                        alpha=pi/2,
                        qlim=[-2.8973, 2.8973],
                        m=7.35522e-01,
                        r=[1.0517e-02, -4.252e-03, 6.1597e-02],
                        I=[1.25160e-02, 1.00270e-02, 4.81500e-03, -4.28000e-04, -7.41000e-04, -1.19600e-03],
                        G=1,
                        mesh='FRANKA-EMIKA/Panda/link7.stl')

        L = [L1, L2, L3, L4, L5, L6, L7]

        self._qz = np.array([0, 0, 0, 0, 0, 0, 0])

        self._qr = np.array([0, -90*deg, -90*deg, 90*deg, 0, -90*deg, 90*deg])

        # Create SerialLink object
        super(Panda, self).__init__(
            L,
            basemesh="FRANKA-EMIKA/Panda/link0.stl",
            name='Panda',
            manufacturer='FRANKA-EMIKA')

    @property
    def qz(self):
        return self._qz

    @property
    def qr(self):
        return self._qr