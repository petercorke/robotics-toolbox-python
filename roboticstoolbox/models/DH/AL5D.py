import numpy as np
from roboticstoolbox import DHRobot, RevoluteMDH
from spatialmath import SE3


class AL5D(DHRobot):
    """
    Class that models a Lynxmotion AL5D manipulator

    :param symbolic: use symbolic constants
    :type symbolic: bool

    ``AL5D()`` is an object which models a Lynxmotion AL5D robot and
    describes its kinematic and dynamic characteristics using modified DH
    conventions.

    .. runblock:: pycon

        >>> import roboticstoolbox as rtb
        >>> robot = rtb.models.DH.AL5D()
        >>> print(robot)

    Defined joint configurations are:

    - qz, zero joint angle configuration

    .. note::
        - SI units are used.

    :References:

        - 'Reference of the robot <http://www.lynxmotion.com/c-130-al5d.aspx>'_
        

    .. codeauthor:: Tassos Natsakis
    """  # noqa

    def __init__(self, symbolic=False):

        if symbolic:
            import spatialmath.base.symbolic as sym
            zero = sym.zero()
            pi = sym.pi()
        else:
            from math import pi
            zero = 0.0

        # robot length values (metres)
        a = [0, 0.002, 0.14679, 0.17751]
        d = [-0.06858, 0, 0, 0]

        alpha = [pi, pi/2, pi, pi]
        offset = [pi/2, pi, -0.0427, -0.0427-pi/2]

        # mass data not yet available
        links = []

        for j in range(4):
            link = RevoluteMDH(
                d=d[j],
                a=a[j],
                alpha=alpha[j],
                offset=offset[j],
                G=1
            )
            links.append(link)
            
        tool=SE3(0.07719,0,0)
    
        super().__init__(
            links,
            name="AL5D",
            manufacturer="Lynxmotion",
            keywords=('dynamics', 'symbolic'),
            symbolic=symbolic,
            tool=tool
        )
    
        # zero angles
        self.addconfiguration("home", np.array([pi/2, pi/2, pi/2, pi/2]))

if __name__ == '__main__':    # pragma nocover

    al5d = AL5D(symbolic=False)
    print(al5d)
    print(al5d.dyntable())
