"""Manual smoke test for the optional Isaac Sim backend."""

import os
from pathlib import Path

import pytest


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("RTB_RUN_ISAACSIM_TESTS") != "1",
        reason="set RTB_RUN_ISAACSIM_TESTS=1 to launch Isaac Sim",
    ),
]


def test_isaacsim_backend_lifecycle() -> None:
    """Create the scene, add a Panda, step, and close Isaac Sim."""

    isaacsim = pytest.importorskip("isaacsim")

    import roboticstoolbox as rtb
    from roboticstoolbox.backends import load_backend
    from roboticstoolbox.backends.IsaacSim import (
        IsaacSim,
        IsaacSimRobotHandle,
    )

    isaacsim_root = Path(isaacsim.__file__).resolve().parent
    urdf_path = (
        isaacsim_root
        / "exts"
        / "isaacsim.asset.importer.urdf"
        / "data"
        / "urdf"
        / "robots"
        / "franka_description"
        / "robots"
        / "panda_arm.urdf"
    )
    assert urdf_path.is_file()

    print("Starting Isaac Sim backend")
    backend = load_backend("isaacsim")
    assert isinstance(backend, IsaacSim)

    try:
        backend.launch(name="RTB Isaac Sim smoke test")
        print("Isaac Sim scene created")

        assert backend._simulation_app is not None
        assert backend._simulation_app.is_running()
        assert backend._world is not None

        panda = rtb.models.URDF.Panda()
        handle = backend.add(
            panda,
            q=panda.qr,
            urdf_path=urdf_path,
            name="panda",
            prim_path="/World/Panda",
        )
        print(f"Panda added at {handle.prim_path}")

        assert isinstance(handle, IsaacSimRobotHandle)
        assert handle.robot is panda
        assert handle.articulation is not None
        assert handle.prim_path == "/World/Panda"
        assert handle.urdf_path == urdf_path.resolve()
        assert handle in backend.robot_handles

        backend.step(1.0 / 60.0)
        print("Simulation advanced by one step")
    finally:
        print("Closing Isaac Sim backend")
        backend.close()
