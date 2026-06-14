from roboticstoolbox import ETS as ET
from roboticstoolbox import ERobot, ELink, Mesh, path_to_datafile
from spatialmath import SE3
from math import pi


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
        l1 = 0.622170
        l2 = -0.12192
        l3 = 0.4318
        l4 = 0.0203
        l5 = -0.0837
        l6 = 0.4318

        e = (
            ET.tz(l1)
            * ET.rz()
            * ET.ty(l2)
            * ET.ry()
            * ET.tz(l3)
            * ET.ry()
            * ET.tx(l4)
            * ET.ty(l5)
            * ET.tz(l6)
            * ET.rz()
            * ET.ry()
            * ET.rz()
            * ET.tx(0.2)
        )
        # e = ET.tz(0.62) * ET.rz() * \
        #     ET.rz() * \
        #     ET.tx(0.4318) * ET.rz() * \
        #     ET.tz(0.15005) * ET.tx(0.0203) * ET.rx(-pi / 2) * ET.rz() * \
        #     ET.tz(0.4318) * ET.rx(pi / 2) * ET.rz() * \
        #     ET.rx(-pi / 2) * ET.rz()

        # pedestal is 0.549m high

        super().__init__(
            e, name="Puma560", manufacturer="Unimation", comment="ETS-based model"
        )

        meshes = path_to_datafile("meshes/Puma560-obj")
        base = ELink(name="base")
        self.links.insert(0, base)
        for j in range(1, 7):
            self.links[j - 1].geometry = Mesh(meshes / f"pieza{j}.obj")
        self.links[1]._parent = base

        self.addconfiguration("qz", [0, 0, 0, 0, 0, 0])
        self.addconfiguration("qbent", [0, -90, 90, 0, 0, 0], "deg")

        self.base = SE3(1, 2, 3)


if __name__ == "__main__":  # pragma nocover
    robot = Puma560()
    print(robot)

    T = robot.fkine_all(robot.qz)
    for link in robot:
        print(link._fk)
    # robot.showgraph()
    robot.plot(robot.qbent, backend="swift")

    """
Mesh Bounding Box min -0.203200 -0.203200 0.000000
Mesh Bounding Box max 0.228600 0.203200 0.548640

Mesh Bounding Box min -0.081280 -0.096520 -0.081280
Mesh Bounding Box max 0.081280 0.142240 0.081280

Mesh Bounding Box min -0.139700 -0.101600 -0.228600
Mesh Bounding Box max 0.139700 0.025400 0.495300




    """
