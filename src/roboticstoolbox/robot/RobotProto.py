from roboticstoolbox.tools.types import ArrayLike, NDArray
from typing import Any, Callable, List, Literal, Optional, Union, Dict
from typing_extensions import Protocol, Self

from roboticstoolbox.robot.Link import Link, BaseLink
from roboticstoolbox.robot.Gripper import Gripper
from roboticstoolbox.robot.ETS import ETS
from spatialmath import SE3


class KinematicsProtocol(Protocol):
    """Structural type contract for the RobotKinematics mixin.

    Declares the attributes that mixin methods access on ``self``.  This avoids
    circular imports while allowing type checkers to validate mixin usage.
    """

    @property
    def _T(self) -> NDArray: ...

    def ets(
        self,
        start: Union[Link, Gripper, str, None] = None,
        end: Union[Link, Gripper, str, None] = None,
    ) -> ETS: ...


class RobotProto(Protocol):
    """Structural type contract for the Dynamics mixin.

    Declares every attribute that Dynamics mixin methods access on ``self``,
    including methods defined within the mixin itself (which must be listed here
    so the type checker knows they exist on ``self`` when called cross-method).
    """

    # ------------------------------------------------------------------
    # Core robot properties
    # ------------------------------------------------------------------

    @property
    def links(self) -> List[BaseLink]: ...

    @property
    def n(self) -> int: ...

    @property
    def q(self) -> NDArray: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, new_name: str): ...

    @property
    def gravity(self) -> NDArray: ...

    # ------------------------------------------------------------------
    # Kinematics (called by dynamics methods)
    # ------------------------------------------------------------------

    def jacobe(
        self,
        q: ArrayLike,
        end: Union[str, BaseLink, Gripper, None] = None,
        start: Union[str, BaseLink, Gripper, None] = None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray: ...

    def jacob0(
        self,
        q: ArrayLike,
        end: Union[str, BaseLink, Gripper, None] = None,
        start: Union[str, BaseLink, Gripper, None] = None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray: ...

    def jacob0_analytical(
        self,
        q: ArrayLike,
        representation: Literal["rpy/xyz", "rpy/zyx", "eul", "exp"] = "rpy/xyz",
        end: Union[str, BaseLink, Gripper, None] = None,
        start: Union[str, BaseLink, Gripper, None] = None,
        tool: Union[NDArray, SE3, None] = None,
    ) -> NDArray: ...

    def jacob0_dot(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        J0: Optional[NDArray] = None,
        representation: Optional[Literal["rpy/xyz", "rpy/zyx", "eul", "exp"]] = None,
    ) -> NDArray: ...

    # ------------------------------------------------------------------
    # Dynamics methods (cross-called within the Dynamics mixin)
    # ------------------------------------------------------------------

    def dynchanged(self) -> None: ...

    def rne(
        self,
        q: NDArray,
        qd: NDArray,
        qdd: NDArray,
        symbolic: bool = False,
        gravity: Union[None, ArrayLike] = None,
    ) -> NDArray: ...

    def inertia(self, q: NDArray) -> NDArray: ...

    def inertia_x(
        self,
        q: Optional[ArrayLike] = None,
        pinv: bool = False,
        representation: Literal["rpy/xyz", "rpy/zyx", "eul", "exp"] = "rpy/xyz",
        Ji: Optional[NDArray] = None,
    ) -> NDArray: ...

    def coriolis(self, q: ArrayLike, qd: ArrayLike) -> NDArray: ...

    def gravload(
        self,
        q: Union[None, ArrayLike] = None,
        gravity: Union[None, ArrayLike] = None,
    ) -> NDArray: ...

    def accel(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        torque: ArrayLike,
        gravity: Optional[ArrayLike] = None,
    ) -> NDArray: ...

    def pay(
        self,
        W: ArrayLike,
        q: Union[NDArray, None] = None,
        J: Union[NDArray, None] = None,
        frame: int = 1,
    ) -> NDArray: ...

    def nofriction(self, coulomb: bool = True, viscous: bool = False) -> Self: ...

    def copy(self) -> Self: ...

    def _fdyn(
        self,
        t: float,
        x: NDArray,
        torqfun: Callable[[Any], NDArray],
        targs: Dict,
    ) -> NDArray: ...

    def delete_rne(self) -> None: ...
