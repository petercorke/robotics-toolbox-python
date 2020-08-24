import graphics as gph
from roboticstoolbox.robot.serial_link import SerialLink, Link
from roboticstoolbox.models.Puma560 import Puma560

if __name__ == "__main__":
    # canvas = gph.GraphicsCanvas3D()

    # Puma560
    puma560 = Puma560()

    puma560.plot(puma560.qz)

    # g_puma560 = gph.GraphicalRobot(canvas, '', seriallink=puma560)

    # Two three-joint arms
    # L = []
    # L2 = []
    #
    # L.append(Link(a=1, jointtype='R'))
    # L.append(Link(a=1, jointtype='R'))
    # L.append(Link(a=1, jointtype='R'))
    #
    # L2.append(Link(a=1, jointtype='R'))
    # L2.append(Link(a=1, jointtype='R'))
    # L2.append(Link(a=1, jointtype='R'))
    #
    # tl = SerialLink(L, name='R1')
    # tl2 = SerialLink(L, name='R2')
    #
    # robot = gph.GraphicalRobot(canvas, '', seriallink=tl)
    # robot2 = gph.GraphicalRobot(canvas, '', seriallink=tl2)
