import numpy as np
from math import sin, cos, pi

# import matplotlib.pyplot as plt
import time
from spatialmath import SE3
import spatialmath.base as smb

from bdsim.components import TransferBlock, FunctionBlock, SourceBlock
from bdsim.graphics import GraphicsBlock

from roboticstoolbox import quintic_func, trapezoidal_func


class Tr2Delta(FunctionBlock):
    r"""
    :blockname:`TR2DELTA`

    Transforms to delta

    :inputs: 2
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - SE3
            - :math:`\mathbf{T}_1` pose.
        *   - Input
            - 1
            - SE3
            - :math:`\mathbf{T}_2` pose.
        *   - Output
            - 0
            - ndarray(6)
            - :math:`\Delta`

    Difference between :math:`\mathbf{T}_1` and :math:`\mathbf{T}_2` as a 6-vector

    :seealso: :class:`Delta2Tr` :func:`~spatialmath.base.transforms3d.tr2delta`
    """

    nin = 2
    nout = 1
    inlabels = ("T1", "T2")
    outlabels = ("Δ",)

    def __init__(self, **blockargs):
        """
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        super().__init__(**blockargs)

        self.inport_names(("T1", "T2"))
        self.outport_names(("$\delta$",))

    def output(self, t, inports, x):
        return [smb.tr2delta(inports[0].A, inports[1].A)]


# ------------------------------------------------------------------------ #


class Delta2Tr(FunctionBlock):
    r"""
    :blockname:`DELTA2TR`

    Delta to transform

    :inputs: 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(6)
            - :math:`\Delta`
        *   - Output
            - 0
            - SE3
            - :math:`\mathbf{T}`, pose.

    6-vector spatial displacement to transform.

    :seealso: :class:`Tr2Delta` :func:`~spatialmath.base.transforms3d.delta2tr`
    """

    nin = 1
    nout = 1
    outlabels = ("T",)
    inlabels = ("Δ",)

    def __init__(self, **blockargs):
        """
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        super().__init__(**blockargs)

        self.inport_names(("$\delta$",))
        self.outport_names(("T",))

    def output(self, t, inports, x):
        return [SE3.Delta(inports[0])]


# ------------------------------------------------------------------------ #


class Point2Tr(FunctionBlock):
    r"""
    :blockname:`POINT2TR`

    Point to transform.

    :inputs: 1
    :outputs: 1
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - ndarray(3)
            - :math:`\mathit{p}`, point.
        *   - Output
            - 0
            - SE3
            - :math:`\mathbf{T}`, pose.

    The parameter ``T`` is an SE3 object whose translation part is replaced by the input
    """

    nin = 1
    nout = 1

    def __init__(self, T=None, **blockargs):
        """
        :param T: the transform
        :type T: SE3
        :param blockargs: |BlockOptions|
        :type blockargs: dict

        If ``T`` is None then it defaults to the identity matrix.
        """
        super().__init__(**blockargs)

        self.inport_names(("t",))
        self.outport_names(("T",))
        if T is None:
            T = SE3()
        self.pose = T

    def output(self, t, inports, x):
        T = SE3.Rt(self.pose.R, t=inports[0])
        return [T]


# ------------------------------------------------------------------------ #


class TR2T(FunctionBlock):
    r"""
    :blockname:`TR2T`

    Translation components of transform

    :inputs: 1
    :outputs: 3
    :states: 0

    .. list-table::
        :header-rows: 1

        *   - Port type
            - Port number
            - Types
            - Description
        *   - Input
            - 0
            - SE3
            - :math:`\mathit{T}` transform.
        *   - Output
            - 0
            - float
            - :math:`x` component of translation.
        *   - Output
            - 1
            - float
            - :math:`y` component of translation.
        *   - Output
            - 2
            - float
            - :math:`z` component of translation.

    :seealso: :func:`~spatialmath.base.transforms3d.transl`
    """

    nin = 1
    nout = 3
    inlabels = ("T",)
    outlabels = ("x", "y", "z")

    def __init__(self, **blockargs):
        """
        :param blockargs: |BlockOptions|
        :type blockargs: dict
        """
        super().__init__(**blockargs)

        self.inport_names(("T",))
        self.outport_names(("x", "y", "z"))

    def output(self, t, inports, x):
        t = inports[0].t
        return list(t)


if __name__ == "__main__":

    from pathlib import Path

    exec(
        open(
            Path(__file__).parent.parent.parent.absolute() / "tests" / "test_blocks.py"
        ).read()
    )
