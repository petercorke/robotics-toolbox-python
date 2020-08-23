import graphics as gph
from roboticstoolbox.robot.serial_link import SerialLink, Link
from roboticstoolbox.models.Puma560 import Puma560

if __name__ == "__main__":

    puma = Puma560()

    puma.plot(puma.qz)

    # canvas = gph.GraphicsCanvas3D()
    #
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
