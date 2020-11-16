from roboticstoolbox import ETS as ET
from roboticstoolbox import ERobot


class Puma560(ERobot):
    """
    Create model of Franka-Emika Panda manipulator

    ``puma = Puma560()`` creates a robot object representing the classic
    Unimation Puma560 robot arm. This robot is represented using the elementary
    transform sequence (ETS).

    .. note::

        - The model has different joint offset conventions compared to
          ``DH.Puma560()``. For this robot:
            - Zero joint angles ``qz`` is the vertical configuration,
              corresponding to ``qr`` with ``DH.Puma560()``
            - ``qbent`` is the bent configuration, corresponding to
              ``qz`` with ``DH.Puma560()``

    :references:
        - "A Simple and Systematic Approach to Assigning Denavit–Hartenberg
          Parameters,", P. I. Corke,  in IEEE Transactions on Robotics, vol. 23,
          no. 3, pp. 590-594, June 2007, doi: 10.1109/TRO.2007.896765.
        - https://petercorke.com/robotics/a-simple-and-systematic-approach-to-assigning-denavit-hartenberg-parameters

    """  # noqa

    def __init__(self):
        # Puma dimensions (m)
        l1 = 0.672
        l2 = 0.2337
        l3 = 0.4318
        l4 = -0.0837
        l5 = 0.4318
        l6 = 0.0203

        e = ET.tz(l1) * ET.rz() * ET.ty(l2) * ET.ry() * ET.tz(l3) * \
            ET.tx(l6) * ET.ty(l4) * ET.ry() * ET.tz(l5) * ET.rz() * \
            ET.ry() * ET.rz() * ET.tx(0.2)

        super().__init__(
            e,
            name='Puma560',
            manufacturer='Unimation',
            comment='ETS-based model')

        self.addconfiguration(
            "qz", [0, 0, 0, 0, 0, 0])
        self.addconfiguration(
            "qbent", [0, -90, 90, 0, 0, 0], 'deg')


if __name__ == '__main__':   # pragma nocover

    robot = Puma560()
    print(robot)
