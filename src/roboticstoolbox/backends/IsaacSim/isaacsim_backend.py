from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
import numpy as np

from roboticstoolbox.tools.types import ArrayLike, NDArray

class IsaacSimRobotHandle:
    """Handle for one independently configured robot instance in Isaac Sim."""

    robot: Any
    articulation: Any
    prim_path: str
    joint_names: tuple[str, ...]
    joint_indices: NDArray
    initial_q: NDArray

    def __init__(
        self,
        robot: Any,
        articulation: Any,
        prim_path: str,
        joint_names: tuple[str, ...],
        joint_indices: NDArray,
        q: ArrayLike,
    ) -> None: ...

    @property
    def q(self) -> NDArray: ...

    @q.setter
    def q(self, q: ArrayLike) -> None: ...

    def reset(self) -> None: ...


class IsaacSim:
    """Isaac Sim 6.2 desktop backend for URDF-backed RTB robots."""
    
    _isaacsim_default_config = {
        # Disable window creation and the user interface.
        # "headless": True,

        # Viewport and generated-image dimensions.
        "width": 1280,
        "height": 720,

        # Run Isaac Sim with its desktop window and UI visible.
        "headless": False,
        
        # Application-window dimensions, independent of the viewport.
        # "window_width": 1440,
        # "window_height": 900,

        # Rendering mode: "RaytracedLighting", "PathTracing",
        # "RealTimePathTracing", or "MinimalRendering".
        # "renderer": "RealTimePathTracing",

        # GPU selection. None uses the default execution GPU.
        # "active_gpu": None,
        # "physics_gpu": 0,

        # Multi-GPU configuration.
        # "multi_gpu": True,
        # "max_gpu_count": None,

        # Open an existing USD stage automatically at startup.
        # "open_usd": None,

        # Create an empty stage when open_usd is not supplied.
        # "create_new_stage": True,

        # Pause rendering until stage assets are fully loaded.
        # "sync_loads": True,

        # Anti-aliasing mode: 0 disabled, 1 TAA, 2 FXAA, 3 DLSS, 4 RTXAA.
        # "anti_aliasing": 3,

        # Path-tracing quality settings.
        # "samples_per_pixel_per_frame": 64,
        # "denoiser": True,
        # "max_bounces": 3,

        # Exit immediately instead of shutting extensions down sequentially.
        # "fast_shutdown": True,

        # Maximum number of CPU threads used by the application.
        # "limit_cpu_threads": 32,

        # Additional command-line arguments passed to the Kit process.
        # "extra_args": [],
    }

    supports_teach  : bool = False
    supports_ellipse: bool = False
    _simulation_app : Any  = None

    def __init__(self, app_config: dict[str, Any] | None = None) -> None:
        
        """Create the Isaac Sim application from the supplied configuration.

        The constructor primarily initializes the Isaac Sim ``SimulationApp``
        instance. ``app_config`` provides application settings such as window
        dimensions and headless operation; when it is omitted, the backend's
        default desktop configuration is used.

        :param app_config: configuration passed to ``SimulationApp``
        """

        config = {**self._isaacsim_default_config, **(app_config or {})}
        self.cached: dict[str, Any] = {
            "app_config": deepcopy(config),
            "launch": None,
            "robots": [],
            "handles": [],
        }
        self._simulation_app = self._create_simulation_app(config)
        self._step_remainder = 0.0

    @property
    def robot_handles(self) -> list[IsaacSimRobotHandle]:
        """Robot handles currently active in the Isaac Sim scene."""

        return self.cached["handles"]

    def launch(
        self,
        name: str | None = None,
        limits: ArrayLike | None = None,
        fig: str | None = None,
        **kwargs: Any,
    ) -> None:
        
        self.cached["launch"] = {
            "name": name,
            "limits": deepcopy(limits),
            "fig": fig,
            "kwargs": deepcopy(kwargs),
        }
        self._create_scene()

    def add(
        self,
        robot: Any,
        readonly: bool = False,
        q: ArrayLike | None = None,
        **kwargs: Any,
    ) -> IsaacSimRobotHandle:
    
        cached_robot = {
            "robot": robot,
            "urdf_path": self._urdf_path(robot),
            "readonly": readonly,
            "q": deepcopy(q),
            "kwargs": deepcopy(kwargs),
        }
        self.cached["robots"].append(cached_robot)
        handle = self._add_cached_robot(cached_robot)
        self.cached["handles"].append(handle)
        return handle

    def remove(self, handle: IsaacSimRobotHandle) -> None: ...

    def step(self, dt: float = 0.05) -> None:
        """Advance Isaac Sim by ``dt`` seconds of simulated time."""

        if dt < 0:
            raise ValueError("dt must be non-negative")

        self._require_launched()

        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        
        physics_dt = SimulationManager.get_physics_dt()
        if physics_dt <= 0:
            raise RuntimeError("Isaac Sim physics dt must be positive")

        self._step_remainder += dt
        step_count = int((self._step_remainder + 1e-12) / physics_dt)
        if step_count == 0:
            return

        self._step_remainder -= step_count * physics_dt
        SimulationManager.step(steps=step_count)

        if not self.cached["app_config"].get("headless", False):
            RenderingManager.render()
    
    def simulation_step(self, dt: float = 0.05) -> None:
    
        if self._simulation_app.is_running():

            self._simulation_app.step()

            self._simulation_app.update()          

    def reset(self) -> None:
        """Rebuild the scene from the cached launch and URDF robot settings."""

        self._require_launched()
        self._simulation_app.reset_render_settings()
        self._step_remainder = 0.0
        self._create_scene()
        self.cached["handles"] = [
            self._add_cached_robot(cached_robot)
            for cached_robot in self.cached["robots"]
        ]

    def restart(self) -> None: 
       
        self.close() 
        
        self.__init__()

    def close(self) -> None:
        """Close the running Isaac Sim application and release its reference.

        .. seealso::
            `SimulationApp.close <https://docs.isaacsim.omniverse.nvidia.com/6.0.0/py/source/extensions/isaacsim.simulation_app/docs/index.html#isaacsim.simulation_app.SimulationApp.close>`_
        """

        if self._simulation_app is not None and not self._simulation_app.is_existing(): 
            
            self._simulation_app.close()
            self._simulation_app = None

    def hold(self) -> None: ...

    def _load_runtime(self) -> None: ...

    def _create_simulation_app(self, app_config: dict[str, Any]) -> Any:
        from isaacsim import SimulationApp

        return SimulationApp(app_config)

    def _create_scene(self) -> None: 
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.core.api.objects.ground_plane import GroundPlane
        from isaacsim.core.api.physics_context import PhysicsContext

      
        if self._simulation_app.is_running(): 

            PhysicsContext()
            GroundPlane(prim_path="/World/groundPlane", size=10, color=np.array([0.5, 0.5, 0.5]))
            
        return

    def _add_cached_robot(
        self, cached_robot: dict[str, Any]
    ) -> IsaacSimRobotHandle: ...

    def _urdf_path(self, robot: Any) -> Path: ...

    def _import_urdf(self, urdf_path: Path, prim_path: str) -> Any: ...

    def _create_joint_mapping(
        self, robot: Any, articulation: Any
    ) -> tuple[tuple[str, ...], NDArray]: ...

    def _next_prim_path(self, robot: Any) -> str: ...

    def _require_launched(self) -> None: ...
