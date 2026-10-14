from typing import Callable, Literal

from spatialmath import SE3
from typing_extensions import Protocol, Self

from roboticstoolbox.ets.ETS import ETS
from roboticstoolbox.robot.Gripper import Gripper
from roboticstoolbox.robot.Link import BaseLink, Link
from roboticstoolbox.tools.types import ArrayLike, NDArray


class KinematicsProtocol(Protocol):
    """Structural type contract for the RobotKinematics mixin.

    Declares the attributes that mixin methods access on ``self``.  This avoids
    circular imports while allowing type checkers to validate mixin usage.
    """

    @property
    def _T(self) -> NDArray: ...

    @property
    def tool(self) -> SE3: ...

    def _resolve_tool(self, tool: NDArray | SE3 | None) -> SE3: ...

    def ets(
        self,
        start: Link | Gripper | str | None = None,
        end: Link | Gripper | str | None = None,
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
    def links(self) -> list[BaseLink]: ...

    _links: list[BaseLink]

    @property
    def ee_links(self) -> list[BaseLink]: ...

    @property
    def n(self) -> int: ...

    @property
    def q(self) -> NDArray: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, new_name: str) -> None: ...

    @property
    def gravity(self) -> NDArray: ...

    # ------------------------------------------------------------------
    # Kinematics (called by dynamics methods)
    # ------------------------------------------------------------------

    def jacobe(
        self,
        q: ArrayLike,
        end: str | BaseLink | Gripper | None = None,
        start: str | BaseLink | Gripper | None = None,
        tool: NDArray | SE3 | None = None,
    ) -> NDArray: ...

    def jacob0(
        self,
        q: ArrayLike,
        end: str | BaseLink | Gripper | None = None,
        start: str | BaseLink | Gripper | None = None,
        tool: NDArray | SE3 | None = None,
    ) -> NDArray: ...

    def jacob0_analytical(
        self,
        q: ArrayLike,
        representation: Literal["rpy/xyz", "rpy/zyx", "eul", "exp"] = "rpy/xyz",
        end: str | BaseLink | Gripper | None = None,
        start: str | BaseLink | Gripper | None = None,
        tool: NDArray | SE3 | None = None,
    ) -> NDArray: ...

    def jacob0_dot(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        J0: NDArray | None = None,
        representation: Literal["rpy/xyz", "rpy/zyx", "eul", "exp"] | None = None,
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
        gravity: ArrayLike | None = None,
    ) -> NDArray: ...

    def inertia(self, q: NDArray) -> NDArray: ...

    def inertia_x(
        self,
        q: ArrayLike | None = None,
        pinv: bool = False,
        representation: Literal["rpy/xyz", "rpy/zyx", "eul", "exp"] = "rpy/xyz",
        Ji: NDArray | None = None,
    ) -> NDArray: ...

    def coriolis(self, q: ArrayLike, qd: ArrayLike) -> NDArray: ...

    def gravload(
        self,
        q: ArrayLike | None = None,
        gravity: ArrayLike | None = None,
    ) -> NDArray: ...

    def accel(
        self,
        q: ArrayLike,
        qd: ArrayLike,
        torque: ArrayLike,
        gravity: ArrayLike | None = None,
    ) -> NDArray: ...

    def pay(
        self,
        W: ArrayLike,
        q: NDArray | None = None,
        J: NDArray | None = None,
        frame: int = 1,
    ) -> NDArray: ...

    def nofriction(self, coulomb: bool = True, viscous: bool = False) -> Self: ...

    def copy(self) -> Self: ...

    def _fdyn(
        self,
        t: float,
        x: NDArray,
        torqfun: Callable[..., NDArray] | None,
        targs: dict,
    ) -> NDArray: ...

    def delete_rne(self) -> None: ...
