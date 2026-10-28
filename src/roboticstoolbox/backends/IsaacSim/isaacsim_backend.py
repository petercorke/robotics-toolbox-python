from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
import numpy as np

from roboticstoolbox.tools.types import ArrayLike, NDArray


@dataclass
class IsaacSimRobotHandle:
    """Handle for one independently configured robot instance in Isaac Sim."""

    robot: Any
    articulation: Any
    prim_path: str
    joint_names: tuple[str, ...]
    joint_indices: NDArray
    q: NDArray
    initial_q: NDArray
    urdf_path: Path
    usd_path: Path


class IsaacSim:
    """Isaac Sim 6.2 desktop backend for URDF-backed RTB robots."""

    class RobotImporter:
        """Import URDF robots and cache their source paths."""

        def __init__(self) -> None:
            self.import_config: Any = None
            self.importer: Any = None
            self.imported_paths: list[Path] = []

        def create_robot(
            self, cached_robot: dict[str, Any]
        ) -> IsaacSimRobotHandle:
            """Import a cached URDF robot and return its Isaac Sim handle."""

            from isaacsim.asset.importer.urdf import (
                URDFImporter,
                URDFImporterConfig,
            )

            robot = cached_robot["robot"]
            urdf_path = cached_robot["urdf_path"]
            resolved_urdf_path = Path(urdf_path).expanduser().resolve()
            output_dir = os.path.dirname(resolved_urdf_path)
            self.import_config = URDFImporterConfig(
                urdf_path=os.fspath(resolved_urdf_path),
                usd_path=output_dir,
                collision_from_visuals=False,
                merge_mesh=False,
            )
            self.importer = URDFImporter(self.import_config)
            output_path = Path(self.importer.import_urdf())
            self.imported_paths.append(resolved_urdf_path)

            configured_q = cached_robot["q"]
            if configured_q is None:
                configured_q = getattr(robot, "q", [])
            q = np.asarray(configured_q, dtype=float).copy()

            return IsaacSimRobotHandle(
                robot=robot,
                articulation=None,
                prim_path="",
                joint_names=(),
                joint_indices=np.array([], dtype=int),
                q=q.copy(),
                initial_q=q.copy(),
                urdf_path=resolved_urdf_path,
                usd_path=output_path,
            )
    
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

    # Isaac Sim does not provide an RTB interactive teach panel in milestone one.
    supports_teach: bool = False

    # Isaac Sim does not render RTB velocity or force ellipsoids in milestone one.
    supports_ellipse: bool = False

    # Running Isaac Sim ``SimulationApp`` instance, created by the backend.
    _simulation_app: Any = None

    # Active Isaac Sim world containing the imported robot articulations.
    _world: Any = None

    # Kit extension manager used to locate Isaac Sim extensions at runtime.
    ext_manager: Any

    # Enabled extension identifier for ``isaacsim.asset.importer.urdf``.
    ext_id: str | None

    # Filesystem path containing the enabled URDF importer extension.
    extension_path: str | None

    @property
    def robot_handles(self) -> list[IsaacSimRobotHandle]:
        """Robot handles currently active in the Isaac Sim scene."""

        return self.cached["handles"]

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
        self.robot_importer = self.RobotImporter()
        self._step_remainder = 0.0

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
        urdf_path = kwargs.pop("urdf_path", None)
        
        if urdf_path is None:
            urdf_path = self._urdf_path(robot)

        cached_robot = {
            "robot": robot,
            "urdf_path": Path(urdf_path),
            "readonly": readonly,
            "q": deepcopy(q),
            "kwargs": deepcopy(kwargs),
        }
        self.cached["robots"].append(cached_robot)
        handle = self._create_robot_from_config(cached_robot)

        import omni.usd
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.utils.stage import add_reference_to_stage

        if self._world is None or omni.usd.get_context().get_stage() is None:
            raise RuntimeError("Isaac Sim world must be launched before adding a robot")

        robot_index = len(self.cached["handles"])
        robot_name = kwargs.get("name", f"robot_{robot_index}")
        robot_prim_path = kwargs.get(
            "prim_path", f"/World/Robot_{robot_index}"
        )
        add_reference_to_stage(
            usd_path=os.fspath(handle.usd_path),
            prim_path=robot_prim_path,
        )
        isaac_robot = self._world.scene.add(
            Robot(
                prim_path=robot_prim_path,
                name=robot_name,
                position=np.array([0.0, 0.0, 0.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        handle.articulation = isaac_robot
        handle.prim_path = robot_prim_path
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
            self._create_robot_from_config(cached_robot)
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


    def _load_runtime_extensions(self) -> None:
        import omni.kit

        self.ext_manager    = omni.kit.app.get_app().get_extension_manager()
        self.ext_id         = self.ext_manager.get_enabled_extension_id("isaacsim.asset.importer.urdf")
        self.extension_path = self.ext_manager.get_extension_path(self.ext_id)

    def _create_simulation_app(self, app_config: dict[str, Any]) -> Any:
        from isaacsim import SimulationApp

        return SimulationApp(app_config)

    def _create_scene(self) -> None: 
        if self._simulation_app.is_running(): 
            from isaacsim.core.api import World
            from isaacsim.core.api.objects.ground_plane import GroundPlane

            self._world = World()
            self._world.scene.add(
                GroundPlane(
                    prim_path="/World/groundPlane",
                    name="ground_plane",
                    size=10,
                    color=np.array([0.5, 0.5, 0.5]),
                )
            )

    def _create_robot_from_config(
        self, cached_robot: dict[str, Any]
    ) -> IsaacSimRobotHandle:
        return self.robot_importer.create_robot(cached_robot)

    #def hold(self) -> None: ...
    #def _urdf_path(self, robot: Any) -> Path: ...
    #def _create_joint_mapping(
        #self, robot: Any, articulation: Any
    #) -> tuple[tuple[str, ...], NDArray]: ...
    #def _next_prim_path(self, robot: Any) -> str: ...
    #def _require_launched(self) -> None: ...
