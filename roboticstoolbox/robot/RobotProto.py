import numpy as np
from roboticstoolbox.tools.types import ArrayLike, NDArray

from typing import Any, Callable, Generic, List, Set, TypeVar, Union, Dict, Tuple, Type
from typing_extensions import Protocol, Self

from roboticstoolbox.robot.Link import Link, BaseLink
from roboticstoolbox.robot.Gripper import Gripper
from roboticstoolbox.robot.ETS import ETS
from spatialmath import SE3
import roboticstoolbox as rtb


class KinematicsProtocol(Protocol):
    @property
    def _T(self) -> NDArray:
        ...

    def ets(
        self,
        start: Union[Link, Gripper, str, None] = None,
        end: Union[Link, Gripper, str, None] = None,
    ) -> ETS:
        ...


class RobotProto(Protocol):
    @property
    def links(self) -> List[BaseLink]:
        ...

    @property
    def n(self) -> int:
        ...

    @property
    def q(self) -> NDArray:
        ...

    @property
    def name(self) -> str:
        ...

    @name.setter
    def name(self, new_name: str):
        ...

    @property
    def gravity(self) -> NDArray:
        ...

    def dynchanged(self):
        ...

    def jacobe(
        self,
        q: ArrayLike,
        end: Union[str, BaseLink, Gripper, None] = None,
        start: Union[str, BaseLink, Gripper, None] = None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        ...

    def jacob0(
        self,
        q: ArrayLike,
        end: Union[str, BaseLink, Gripper, None] = None,
        start: Union[str, BaseLink, Gripper, None] = None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray:
        ...

    def copy(self) -> Self:
        ...

    def accel(self, q, qd, torque, gravity=None) -> NDArray:
        ...

    def nofriction(self, coulomb: bool, viscous: bool) -> Self:
        ...

    def _fdyn(
        self,
        t: float,
        x: NDArray,
        torqfun: Callable[[Any], NDArray],
        targs: Dict,
    ) -> NDArray:
        ...

    def rne(
        self,
        q: NDArray,
        qd: NDArray,
        qdd: NDArray,
        symbolic: bool = False,
        gravity: Union[None, ArrayLike] = None,
    ) -> NDArray:
        ...

    def gravload(
        self, q: Union[None, ArrayLike] = None, gravity: Union[None, ArrayLike] = None
    ):
        ...

    def pay(
        self,
        W: ArrayLike,
        q: Union[NDArray, None] = None,
        J: Union[NDArray, None] = None,
        frame: int = 1,
    ):
        ...
