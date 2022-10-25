import numpy as np
from roboticstoolbox.tools.types import ArrayLike

from typing import Any, Callable, Generic, List, Set, TypeVar, Union, Dict, Tuple, Type
from typing_extensions import Protocol, Self

from roboticstoolbox.robot.Link import Link, BaseLink
from roboticstoolbox.robot.Gripper import Gripper
from roboticstoolbox.robot.ETS import ETS
from spatialmath import SE3
import roboticstoolbox as rtb


class KinematicsProtocol(Protocol):
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
    def q(self) -> np.ndarray:
        ...

    @property
    def name(self) -> str:
        ...

    @name.setter
    def name(self, new_name: str):
        ...

    @property
    def gravity(self) -> np.ndarray:
        ...

    def dynchanged(self):
        ...

    def jacobe(
        self,
        q: ArrayLike,
        end: Union[str, BaseLink, Gripper, None] = None,
        start: Union[str, BaseLink, Gripper, None] = None,
        tool: Union[np.ndarray, SE3, None] = None,
    ) -> np.ndarray:
        ...

    def jacob0(
        self,
        q: ArrayLike,
        end: Union[str, BaseLink, Gripper, None] = None,
        start: Union[str, BaseLink, Gripper, None] = None,
        tool: Union[np.ndarray, SE3, None] = None,
    ) -> np.ndarray:
        ...

    def copy(self) -> Self:
        ...

    def accel(self, q, qd, torque, gravity=None) -> np.ndarray:
        ...

    def nofriction(self, coulomb: bool, viscous: bool) -> Self:
        ...

    def _fdyn(
        self,
        t: float,
        x: np.ndarray,
        torqfun: Callable[[Any], np.ndarray],
        targs: Dict,
    ) -> np.ndarray:
        ...

    def rne(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        qdd: np.ndarray,
        symbolic: bool = False,
        gravity: Union[None, ArrayLike] = None,
    ) -> np.ndarray:
        ...

    def gravload(
        self, q: Union[None, ArrayLike] = None, gravity: Union[None, ArrayLike] = None
    ):
        ...

    def pay(
        self,
        W: ArrayLike,
        q: Union[np.ndarray, None] = None,
        J: Union[np.ndarray, None] = None,
        frame: int = 1,
    ):
        ...
