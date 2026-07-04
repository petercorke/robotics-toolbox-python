"""
Unit tests for spatialgeometry collision detection (Coal backend).

Distances are verified analytically rather than against PyBullet so the tests
serve as ground-truth validation of Coal's signed-distance results.

Note on inf_dist
----------------
closest_point(shape, inf_dist=1.0) returns (None, None, None) when the
distance exceeds inf_dist.  Tests that want the actual distance for separated
shapes must pass a suitable inf_dist.  Tests that verify the (None,None,None)
filtering use a small inf_dist intentionally.
"""

import sys
import importlib
import builtins
import pytest
import numpy as np
from unittest.mock import patch
from spatialmath import SE3

# spatialgeometry.geom.__init__ exports a class also called CollisionShape,
# which shadows the module of the same name.  Fetch the module via sys.modules
# so we can reach module-level state (_coal, _require_coal).
import spatialgeometry.geom as gm
import spatialgeometry.geom.CollisionShape   # ensure loaded
_mod = sys.modules["spatialgeometry.geom.CollisionShape"]

from spatialgeometry.geom.CollisionShape import Sphere, Cuboid, Cylinder, Box

from tests import skip_no_collision_checking


# ── helpers ───────────────────────────────────────────────────────────────────

def _reset_coal():
    """Force _require_coal() to re-run on next use."""
    _mod._coal = None


def sphere_at(radius, x=0.0, y=0.0, z=0.0) -> Sphere:
    return Sphere(radius, pose=SE3(x, y, z))


def cuboid_at(sx, sy, sz, x=0.0, y=0.0, z=0.0) -> Cuboid:
    return Cuboid([sx, sy, sz], pose=SE3(x, y, z))


def cylinder_at(radius, length, x=0.0, y=0.0, z=0.0) -> Cylinder:
    return Cylinder(radius, length, pose=SE3(x, y, z))


BIG = 1e6   # inf_dist large enough to always get a result


# ── environment guards ────────────────────────────────────────────────────────

class TestEnvironmentGuards:
    def test_pyodide_raises_runtime_error(self):
        _reset_coal()
        with patch.object(sys, "platform", "emscripten"):
            with pytest.raises(RuntimeError, match="browser"):
                _mod._require_coal()

    def test_missing_coal_raises_import_error(self):
        _reset_coal()
        with patch.dict(sys.modules, {"coal": None}):
            with pytest.raises(ImportError, match="coal"):
                _mod._require_coal()

    @skip_no_collision_checking
    def test_coal_loads_on_demand(self):
        _reset_coal()
        assert _mod._coal is None
        _mod._require_coal()
        assert _mod._coal is not None

    @skip_no_collision_checking
    def test_second_call_is_noop(self):
        """_require_coal() must not re-import on subsequent calls."""
        _mod._require_coal()
        first = _mod._coal
        _mod._require_coal()
        assert _mod._coal is first


# ── Sphere – Sphere ───────────────────────────────────────────────────────────

@skip_no_collision_checking
class TestSphereSphere:
    """Analytical ground truth: d = |c1 − c2| − r1 − r2."""

    def test_separated(self):
        # centres 5 apart, radii 1 each → gap = 3
        d, p1, p2 = sphere_at(1.0).closest_point(sphere_at(1.0, x=5.0), inf_dist=BIG)
        assert d == pytest.approx(3.0, abs=1e-6)
        assert p1 is not None and p2 is not None

    def test_touching(self):
        d, _, _ = sphere_at(1.0).closest_point(sphere_at(1.0, x=2.0), inf_dist=BIG)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_penetrating(self):
        # centres 1 apart, radii 2 each → d = 1 − 4 = −3
        d, _, _ = sphere_at(2.0).closest_point(sphere_at(2.0, x=1.0), inf_dist=BIG)
        assert d == pytest.approx(-3.0, abs=1e-6)

    def test_concentric_is_negative(self):
        d, _, _ = sphere_at(1.0).closest_point(sphere_at(2.0), inf_dist=BIG)
        assert d < 0

    def test_different_radii(self):
        # r1=1, r2=3, centres 6 apart → gap = 6 − 4 = 2
        d, _, _ = sphere_at(1.0).closest_point(sphere_at(3.0, x=6.0), inf_dist=BIG)
        assert d == pytest.approx(2.0, abs=1e-6)

    def test_contact_points_on_surface(self):
        # centres at 0 and 4, radii 1 → p1 at x=1, p2 at x=3
        d, p1, p2 = sphere_at(1.0).closest_point(sphere_at(1.0, x=4.0), inf_dist=BIG)
        assert p1[0] == pytest.approx(1.0, abs=1e-5)
        assert p2[0] == pytest.approx(3.0, abs=1e-5)

    def test_inf_dist_filters_far_shape(self):
        d, p1, p2 = sphere_at(1.0).closest_point(sphere_at(1.0, x=100.0), inf_dist=0.5)
        assert (d, p1, p2) == (None, None, None)

    def test_inf_dist_passes_close_shape(self):
        # gap = 1.0 — exactly at default inf_dist boundary
        d, _, _ = sphere_at(1.0).closest_point(sphere_at(1.0, x=3.0), inf_dist=2.0)
        assert d is not None
        assert d == pytest.approx(1.0, abs=1e-6)

    def test_symmetry(self):
        s1, s2 = sphere_at(1.0), sphere_at(2.0, x=6.0)
        d1, _, _ = s1.closest_point(s2, inf_dist=BIG)
        d2, _, _ = s2.closest_point(s1, inf_dist=BIG)
        assert d1 == pytest.approx(d2, abs=1e-10)


# ── Cuboid – Cuboid ───────────────────────────────────────────────────────────

@skip_no_collision_checking
class TestCuboidCuboid:
    """Unit cubes (1×1×1) extend ±0.5 along each axis from their centre."""

    def test_separated_along_x(self):
        # centres at 0 and 3; faces at ±0.5 and 2.5/3.5 → gap = 2.0
        d, _, _ = cuboid_at(1,1,1).closest_point(cuboid_at(1,1,1, x=3.0), inf_dist=BIG)
        assert d == pytest.approx(2.0, abs=1e-5)

    def test_face_to_face_touching(self):
        d, _, _ = cuboid_at(1,1,1).closest_point(cuboid_at(1,1,1, x=1.0), inf_dist=BIG)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_penetrating(self):
        d, _, _ = cuboid_at(1,1,1).closest_point(cuboid_at(1,1,1, x=0.5), inf_dist=BIG)
        assert d < 0

    def test_separated_along_y(self):
        # gap = 2.0 − 0.5 − 0.5 = 1.0
        d, _, _ = cuboid_at(1,1,1).closest_point(cuboid_at(1,1,1, y=2.0), inf_dist=BIG)
        assert d == pytest.approx(1.0, abs=1e-5)

    def test_different_sizes(self):
        # 2×2×2 centred at 0, 1×1×1 centred at 3: faces at 1.0 and 2.5 → gap = 1.5
        d, _, _ = cuboid_at(2,2,2).closest_point(cuboid_at(1,1,1, x=3.0), inf_dist=BIG)
        assert d == pytest.approx(1.5, abs=1e-5)


# ── Cylinder ──────────────────────────────────────────────────────────────────

@skip_no_collision_checking
class TestCylinder:
    def test_separated_radially(self):
        # two Z-axis cylinders, centres 3 apart along X, radius 0.5 each → gap 2.0
        d, _, _ = cylinder_at(0.5, 2.0).closest_point(
            cylinder_at(0.5, 2.0, x=3.0), inf_dist=BIG)
        assert d == pytest.approx(2.0, abs=1e-5)

    def test_touching_radially(self):
        d, _, _ = cylinder_at(1.0, 2.0).closest_point(
            cylinder_at(1.0, 2.0, x=2.0), inf_dist=BIG)
        assert d == pytest.approx(0.0, abs=1e-5)

    def test_penetrating(self):
        d, _, _ = cylinder_at(1.0, 2.0).closest_point(
            cylinder_at(1.0, 2.0, x=1.0), inf_dist=BIG)
        assert d < 0


# ── Mixed shape pairs ─────────────────────────────────────────────────────────

@skip_no_collision_checking
class TestMixedPairs:
    def test_sphere_cuboid_separated(self):
        # sphere r=1 at origin; cuboid 1×1×1 at (3,0,0)
        # sphere surface at 1.0; cuboid face at 2.5 → gap = 1.5
        d, _, _ = sphere_at(1.0).closest_point(cuboid_at(1,1,1, x=3.0), inf_dist=BIG)
        assert d == pytest.approx(1.5, abs=1e-5)

    def test_sphere_cuboid_penetrating(self):
        d, _, _ = sphere_at(1.0).closest_point(cuboid_at(1,1,1), inf_dist=BIG)
        assert d < 0

    def test_sphere_cylinder(self):
        # sphere r=1 at origin; cylinder r=1 at (4,0,0) → gap = 4 − 1 − 1 = 2
        d, _, _ = sphere_at(1.0).closest_point(
            cylinder_at(1.0, 4.0, x=4.0), inf_dist=BIG)
        assert d == pytest.approx(2.0, abs=1e-5)

    def test_mixed_symmetry(self):
        s = sphere_at(1.0)
        c = cuboid_at(2, 2, 2, x=5.0)
        d1, _, _ = s.closest_point(c, inf_dist=BIG)
        d2, _, _ = c.closest_point(s, inf_dist=BIG)
        assert d1 == pytest.approx(d2, abs=1e-10)


# ── iscollided ────────────────────────────────────────────────────────────────

@skip_no_collision_checking
class TestIsCollided:
    def test_separated_not_collided(self):
        assert not sphere_at(1.0).iscollided(sphere_at(1.0, x=5.0))

    def test_touching_is_collided(self):
        assert sphere_at(1.0).iscollided(sphere_at(1.0, x=2.0))

    def test_penetrating_is_collided(self):
        assert sphere_at(2.0).iscollided(sphere_at(2.0, x=1.0))

    def test_cuboids_not_collided(self):
        assert not cuboid_at(1,1,1).iscollided(cuboid_at(1,1,1, x=3.0))

    def test_cuboids_collided(self):
        assert cuboid_at(1,1,1).iscollided(cuboid_at(1,1,1, x=0.5))

    def test_cylinder_sphere_not_collided(self):
        assert not cylinder_at(0.5, 1.0).iscollided(sphere_at(0.5, x=5.0))

    def test_cylinder_sphere_collided(self):
        assert cylinder_at(1.0, 2.0).iscollided(sphere_at(1.0))


# ── deprecation warnings ──────────────────────────────────────────────────────

class TestDeprecation:
    @skip_no_collision_checking
    def test_collided_warns(self):
        with pytest.warns(FutureWarning, match="iscollided"):
            result = sphere_at(1.0).collided(sphere_at(1.0, x=5.0))
        assert result is False

    @skip_no_collision_checking
    def test_collided_result_matches_iscollided(self):
        s1, s2 = sphere_at(2.0), sphere_at(2.0, x=1.0)
        with pytest.warns(FutureWarning):
            deprecated = s1.collided(s2)
        assert deprecated == s1.iscollided(s2)

    def test_box_warns(self):
        with pytest.warns(FutureWarning, match="Cuboid"):
            b = Box([1, 1, 1])
        assert isinstance(b, Cuboid)


# ── pose / world-frame transforms ─────────────────────────────────────────────

@skip_no_collision_checking
class TestPoseTransforms:
    def test_sphere_pose_at_construction(self):
        s1 = Sphere(1.0, pose=SE3(0, 0, 0))
        s2 = Sphere(1.0, pose=SE3(4, 0, 0))
        d, _, _ = s1.closest_point(s2, inf_dist=BIG)
        assert d == pytest.approx(2.0, abs=1e-6)

    def test_cuboid_pose_at_construction(self):
        c1 = Cuboid([1, 1, 1], pose=SE3(10, 0, 0))
        c2 = Cuboid([1, 1, 1], pose=SE3(13, 0, 0))
        d, _, _ = c1.closest_point(c2, inf_dist=BIG)
        assert d == pytest.approx(2.0, abs=1e-5)

    def test_contact_points_in_world_frame(self):
        # spheres r=1, centres at (10,0,0) and (14,0,0)
        # p1 at (11,0,0), p2 at (13,0,0)
        s1 = Sphere(1.0, pose=SE3(10, 0, 0))
        s2 = Sphere(1.0, pose=SE3(14, 0, 0))
        d, p1, p2 = s1.closest_point(s2, inf_dist=BIG)
        assert p1[0] == pytest.approx(11.0, abs=1e-4)
        assert p2[0] == pytest.approx(13.0, abs=1e-4)
        # Y and Z components should be near zero
        assert p1[1] == pytest.approx(0.0, abs=1e-4)
        assert p1[2] == pytest.approx(0.0, abs=1e-4)


# ── collision=False guard ─────────────────────────────────────────────────────

@skip_no_collision_checking
class TestCollisionFalseGuard:
    def test_sphere_collision_false_raises(self):
        with pytest.raises(ValueError, match="collision=False"):
            sphere_at(1.0, x=0).closest_point(Sphere(1.0, collision=False))

    def test_cuboid_collision_false_raises(self):
        with pytest.raises(ValueError, match="collision=False"):
            Cuboid([1, 1, 1], collision=False).closest_point(cuboid_at(1, 1, 1))

    def test_cylinder_collision_false_raises(self):
        with pytest.raises(ValueError, match="collision=False"):
            Cylinder(1.0, 1.0, collision=False).closest_point(cylinder_at(1.0, 1.0))


# ── return type / structure ───────────────────────────────────────────────────

@skip_no_collision_checking
class TestResultStructure:
    def test_returns_three_tuple(self):
        result = sphere_at(1.0).closest_point(sphere_at(1.0, x=5.0), inf_dist=BIG)
        assert len(result) == 3

    def test_p1_p2_are_ndarrays_shape_3(self):
        d, p1, p2 = sphere_at(1.0).closest_point(sphere_at(1.0, x=5.0), inf_dist=BIG)
        assert isinstance(p1, np.ndarray) and p1.shape == (3,)
        assert isinstance(p2, np.ndarray) and p2.shape == (3,)

    def test_none_triple_beyond_inf_dist(self):
        d, p1, p2 = sphere_at(1.0).closest_point(sphere_at(1.0, x=100.0), inf_dist=0.5)
        assert d is None and p1 is None and p2 is None

    def test_distance_is_float(self):
        d, _, _ = sphere_at(1.0).closest_point(sphere_at(1.0, x=5.0), inf_dist=BIG)
        assert isinstance(d, float)


# ── original test_closest chain (ported from spatialgeometry test suite) ──────

@skip_no_collision_checking
class TestClosestChain:
    """
    Mixed-type chain matching the analytical scenario from the original
    spatialgeometry test suite, now validated against Coal ground truth.

    Layout (all on X axis):
        s0 = Cuboid 1×1×1  at (0, 0, 0)  — faces at ±0.5
        s1 = Cylinder r=1, l=1  at (2, 0, 0)  — radial surface at x=1
        s2 = Sphere r=1         at (4, 0, 0)  — surface at x=3
    """

    def setup_method(self):
        self.s0 = gm.Cuboid([1, 1, 1], pose=SE3(0, 0, 0))
        self.s1 = gm.Cylinder(1, 1, pose=SE3(2, 0, 0))
        self.s2 = gm.Sphere(1, pose=SE3(4, 0, 0))

    def test_cuboid_to_cylinder(self):
        # cuboid face at 0.5; cylinder surface at 1.0 → gap = 0.5
        d, _, _ = self.s0.closest_point(self.s1, 10)
        assert d == pytest.approx(0.5, abs=1e-6)

    def test_cylinder_to_sphere(self):
        # cylinder surface at 3.0; sphere surface at 3.0 → touching (0)
        d, _, _ = self.s1.closest_point(self.s2, 10)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_sphere_to_cuboid(self):
        # sphere surface at 3.0; cuboid face at 0.5 → gap = 2.5
        d, _, _ = self.s2.closest_point(self.s0, 10)
        assert d == pytest.approx(2.5, abs=1e-6)

    def test_sphere_to_cuboid_default_inf_dist(self):
        # gap 2.5 > default inf_dist 1.0 → None triple
        d, p1, p2 = self.s2.closest_point(self.s0)
        assert (d, p1, p2) == (None, None, None)


# ── to_dict ───────────────────────────────────────────────────────────────────

class TestToDict:
    def test_sphere_stype(self):
        assert gm.Sphere(1).to_dict()["stype"] == "sphere"

    def test_sphere_radius(self):
        assert gm.Sphere(2.5).to_dict()["radius"] == 2.5

    def test_sphere_origin(self):
        d = gm.Sphere(1).to_dict()
        assert d["t"] == [0.0, 0.0, 0.0]
        assert d["q"] == pytest.approx([0.0, 0.0, 0.0, 1.0], abs=1e-9)

    def test_cylinder_stype(self):
        assert gm.Cylinder(1, 2).to_dict()["stype"] == "cylinder"

    def test_cylinder_dimensions(self):
        d = gm.Cylinder(0.5, 3.0).to_dict()
        assert d["radius"] == 0.5
        assert d["length"] == 3.0

    def test_cuboid_stype(self):
        assert gm.Cuboid([1, 2, 3]).to_dict()["stype"] == "cuboid"

    def test_cuboid_scale(self):
        assert gm.Cuboid([1.0, 2.0, 3.0]).to_dict()["scale"] == [1.0, 2.0, 3.0]

    def test_cuboid_none_scale_defaults(self):
        d = gm.Cuboid(None).to_dict()
        assert d["scale"] == [1.0, 1.0, 1.0]

    def test_mesh_stype(self):
        assert gm.Mesh("robot.stl").to_dict()["stype"] == "mesh"

    def test_mesh_filename(self):
        assert gm.Mesh("robot.stl").to_dict()["filename"] == "robot.stl"

    def test_mesh_scale(self):
        assert gm.Mesh("robot.stl", scale=[2, 2, 2]).to_dict()["scale"] == [2.0, 2.0, 2.0]


# ── _init_coal collision=False direct call ────────────────────────────────────

@skip_no_collision_checking
class TestInitCoalDirect:
    """Call _init_coal() directly, matching the pattern in the original suite."""

    def test_mesh_collision_false(self):
        s = gm.Mesh("test.stl", collision=False)
        _mod._require_coal()   # ensure coal loaded so _init_coal can proceed
        with pytest.raises(ValueError, match="collision=False"):
            s._init_coal()

    def test_cylinder_collision_false(self):
        s = gm.Cylinder(1, 1, collision=False)
        _mod._require_coal()
        with pytest.raises(ValueError, match="collision=False"):
            s._init_coal()

    def test_sphere_collision_false(self):
        s = gm.Sphere(1, collision=False)
        _mod._require_coal()
        with pytest.raises(ValueError, match="collision=False"):
            s._init_coal()

    def test_cuboid_collision_false(self):
        s = gm.Cuboid([1, 1, 1], collision=False)
        _mod._require_coal()
        with pytest.raises(ValueError, match="collision=False"):
            s._init_coal()
