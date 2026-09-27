#!/usr/bin/env python3
"""
Created on Fri May 1 14:04:04 2020
@author: Jesse Haviland
"""

import numpy.testing as nt
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import ERobot, ET, ETS, Link

# from spatialmath import SE2, SE3
import unittest
import spatialmath as sm
import spatialgeometry as gm
from math import pi, sin, cos
from tests import skip_no_collision_checking

try:
    from sympy import symbols

    _sympy = True
except ModuleNotFoundError:
    _sympy = False


class TestERobot(unittest.TestCase):
    def test_jacobm(self):
        panda = rtb.models.ETS.Panda()
        q1 = np.array([1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9])
        q2 = [1.4, 0.2, 1.8, 0.7, 0.1, 3.1, 2.9]
        q3 = np.expand_dims(q1, 0)
        q4 = np.expand_dims(q1, 1)

        ans = np.array(
            [
                [1.27080875e-17],
                [2.38242538e-02],
                [6.61029519e-03],
                [8.18202121e-03],
                [7.74546204e-04],
                [-1.10885380e-02],
                [0.00000000e00],
            ]
        )

        panda.q = q1
        nt.assert_array_almost_equal(panda.jacobm(), ans)
        nt.assert_array_almost_equal(panda.jacobm(q2), ans)
        nt.assert_array_almost_equal(panda.jacobm(q3), ans)
        nt.assert_array_almost_equal(panda.jacobm(q4), ans)
        nt.assert_array_almost_equal(panda.jacobm(J=panda.jacob0(q1)), ans)
        # self.assertRaises(ValueError, panda.jacobm)
        self.assertRaises(TypeError, panda.jacobm, "Wfgsrth")
        self.assertRaises(ValueError, panda.jacobm, [1, 3], np.array([1, 5]))
        self.assertRaises(TypeError, panda.jacobm, [1, 3], "qwe")
        self.assertRaises(TypeError, panda.jacobm, [1, 3], panda.jacob0(q1), [1, 2, 3])
        self.assertRaises(
            ValueError, panda.jacobm, [1, 3], panda.jacob0(q1), np.array([1, 2, 3])
        )

    def test_dict(self):
        panda = rtb.models.Panda()
        panda.grippers[0].links[0].collision.append(gm.Cuboid([1, 1, 1]))
        panda._to_dict()

        wx = rtb.models.wx250s()
        wx._to_dict()

    def test_fkdict(self):
        panda = rtb.models.Panda()
        panda.grippers[0].links[0].collision.append(gm.Cuboid([1, 1, 1]))
        panda._fk_dict()

    @skip_no_collision_checking
    def test_dist(self):
        s0 = gm.Cuboid([1, 1, 1], pose=sm.SE3(0, 0, 0))
        s1 = gm.Cuboid([1, 1, 1], pose=sm.SE3(3, 0, 0))
        p = rtb.models.Panda()

        d0, _, _ = p.closest_point(p.q, s0)
        d1, _, _ = p.closest_point(p.q, s1, 5)
        d2, _, _ = p.closest_point(p.q, s1)

        self.assertAlmostEqual(d0, -0.5599999999995913)  # type: ignore
        self.assertAlmostEqual(d1, 2.3621, places=4)  # type: ignore
        self.assertAlmostEqual(d2, None)  # type: ignore

    @skip_no_collision_checking
    def test_collided(self):
        s0 = gm.Cuboid([1, 1, 1], pose=sm.SE3(0, 0, 0))
        s1 = gm.Cuboid([1, 1, 1], pose=sm.SE3(3, 0, 0))
        p = rtb.models.Panda()

        c0 = p.iscollided(p.q, s0)
        c1 = p.iscollided(p.q, s1)

        self.assertTrue(c0)
        self.assertFalse(c1)

    def test_invdyn(self):
        # create a 2 link robot
        # Example from Spong etal. 2nd edition, p. 260
        l1 = Link(ets=ETS(ET.Ry()), m=1, r=[0.5, 0, 0], name="l1")
        l2 = Link(ets=ETS(ET.tx(1)) * ET.Ry(), m=1, r=[0.5, 0, 0], parent=l1, name="l2")
        robot = ERobot([l1, l2], name="simple 2 link")
        z = np.zeros(robot.n)

        # check gravity load
        tau = robot.rne(z, z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-2, -0.5])

        tau = robot.rne(np.array([0.0, -pi / 2.0]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-1.5, 0])

        tau = robot.rne(np.array([-pi / 2, pi / 2]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-0.5, -0.5])

        tau = robot.rne(np.array([-pi / 2, 0]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        # check velocity terms
        robot.gravity = [0, 0, 0]
        q = np.array([0, -pi / 2])
        h = -0.5 * sin(q[1])

        tau = robot.rne(q, np.array([0, 0]), z)
        nt.assert_array_almost_equal(tau, np.r_[0, 0] * h)

        tau = robot.rne(q, np.array([1, 0]), z)
        nt.assert_array_almost_equal(tau, np.r_[0, -1] * h)

        tau = robot.rne(q, np.array([0, 1]), z)
        nt.assert_array_almost_equal(tau, np.r_[1, 0] * h)

        tau = robot.rne(q, np.array([1, 1]), z)
        nt.assert_array_almost_equal(tau, np.r_[3, -1] * h)

        # check inertial terms

        d11 = 1.5 + cos(q[1])
        d12 = 0.25 + 0.5 * cos(q[1])
        d21 = d12
        d22 = 0.25

        tau = robot.rne(q, z, np.array([0, 0]))
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        tau = robot.rne(q, z, np.array([1, 0]))
        nt.assert_array_almost_equal(tau, np.r_[d11, d21])

        tau = robot.rne(q, z, np.array([0, 1]))
        nt.assert_array_almost_equal(tau, np.r_[d12, d22])

        tau = robot.rne(q, z, np.array([1, 1]))
        nt.assert_array_almost_equal(tau, np.r_[d11 + d12, d21 + d22])

    def test_invdyn_static(self):
        # create a 2 link robot
        # Example from Spong etal. 2nd edition, p. 260
        l1 = Link(ets=ETS(ET.Ry()), m=1, r=[0.5, 0, 0], name="l1")
        l2 = Link(ets=ETS(), m=0, r=[0, 0, 0], parent=l1, name="l2")
        l3 = Link(ets=ETS(ET.tx(1)) * ET.Ry(), m=1, r=[0.5, 0, 0], parent=l2, name="l3")
        robot = ERobot([l1, l2, l3], name="simple 3 link")
        z = np.zeros(robot.n)

        # check gravity load
        tau = robot.rne(z, z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-2, -0.5])

        tau = robot.rne(np.array([0.0, -pi / 2.0]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-1.5, 0])

        tau = robot.rne(np.array([-pi / 2, pi / 2]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-0.5, -0.5])

        tau = robot.rne(np.array([-pi / 2, 0]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        # check velocity terms
        robot.gravity = [0, 0, 0]
        q = np.array([0, -pi / 2])
        h = -0.5 * sin(q[1])

        tau = robot.rne(q, np.array([0, 0]), z)
        nt.assert_array_almost_equal(tau, np.r_[0, 0] * h)

        tau = robot.rne(q, np.array([1, 0]), z)
        nt.assert_array_almost_equal(tau, np.r_[0, -1] * h)

        tau = robot.rne(q, np.array([0, 1]), z)
        nt.assert_array_almost_equal(tau, np.r_[1, 0] * h)

        tau = robot.rne(q, np.array([1, 1]), z)
        nt.assert_array_almost_equal(tau, np.r_[3, -1] * h)

        # check inertial terms

        d11 = 1.5 + cos(q[1])
        d12 = 0.25 + 0.5 * cos(q[1])
        d21 = d12
        d22 = 0.25

        tau = robot.rne(q, z, np.array([0, 0]))
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        tau = robot.rne(q, z, np.array([1, 0]))
        nt.assert_array_almost_equal(tau, np.r_[d11, d21])

        tau = robot.rne(q, z, np.array([0, 1]))
        nt.assert_array_almost_equal(tau, np.r_[d12, d22])

        tau = robot.rne(q, z, np.array([1, 1]))
        nt.assert_array_almost_equal(tau, np.r_[d11 + d12, d21 + d22])

    def test_invdyn_static2(self):
        # create a 3 link robot: joint1 (Ry) -> l2 (*fixed*, tx(1)) ->
        # joint2 (Ry, massless -- l3 carries no mass, and l2 is rigidly
        # welded to joint1's output with no rotational freedom of its own).
        #
        # Unlike test_invdyn (whose second link rotates with joint2), l2's
        # position here depends only on q0, never on q1: joint2 has nothing
        # downstream with mass, so its manipulator-inertia row/column and
        # every velocity (Coriolis) term must be exactly zero, and M11 is a
        # constant rather than a function of q1. Verified against an
        # independent virtual-work (finite-difference of CoM height vs. q)
        # ground truth and by hand (M11 = m1*r1^2 + m2*(offset+r2)^2 =
        # 1*0.5^2 + 1*1.5^2 = 2.5), not just against rne() itself -- see
        # #483, where a sandwiched static link's mass was misattributed to
        # the *following* joint's group instead of the *preceding* one it's
        # rigidly attached to, and this test's previous expected values
        # (copied from test_invdyn's different topology, where l2 *does*
        # rotate with joint2) matched that bug rather than catching it.
        l1 = Link(ets=ETS(ET.Ry()), m=1, r=[0.5, 0, 0], name="l1")
        l2 = Link(ets=ETS(ET.tx(1)), m=1, r=[0.5, 0, 0], parent=l1, name="l2")
        l3 = Link(ets=ETS(ET.Ry()), m=0, r=[0, 0, 0], parent=l2, name="l3")
        robot = ERobot([l1, l2, l3], name="simple 3 link")
        z = np.zeros(robot.n)

        # check gravity load -- tau1 is exactly zero at every configuration
        tau = robot.rne(z, z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-2, 0])

        tau = robot.rne(np.array([0.0, -pi / 2.0]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[-2, 0])

        tau = robot.rne(np.array([-pi / 2, pi / 2]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        tau = robot.rne(np.array([-pi / 2, 0]), z, z) / 9.81
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        # check velocity terms -- M(q) doesn't depend on q at all (l2 is
        # rigidly fixed, l3 is massless), so every Coriolis/centrifugal
        # term is zero regardless of qd
        robot.gravity = [0, 0, 0]
        q = np.array([0, -pi / 2])

        for qd in (np.r_[0, 0], np.r_[1, 0], np.r_[0, 1], np.r_[1, 1]):
            tau = robot.rne(q, qd, z)
            nt.assert_array_almost_equal(tau, np.r_[0, 0])

        # check inertial terms -- M11=2.5 (constant), M12=M21=M22=0
        tau = robot.rne(q, z, np.array([0, 0]))
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        tau = robot.rne(q, z, np.array([1, 0]))
        nt.assert_array_almost_equal(tau, np.r_[2.5, 0])

        tau = robot.rne(q, z, np.array([0, 1]))
        nt.assert_array_almost_equal(tau, np.r_[0, 0])

        tau = robot.rne(q, z, np.array([1, 1]))
        nt.assert_array_almost_equal(tau, np.r_[2.5, 0])

    def test_invdyn_sandwiched_static_link(self):
        # Regression test for #483: a static (fixed) link *sandwiched*
        # between two joints -- rigidly welded to the *preceding* joint's
        # output, not the *following* one -- had its mass misattributed to
        # the wrong joint's torque. The grouping logic fixed by #636
        # attached a static link to whichever joint came next scanning
        # forward through the link list, which is correct for a trailing
        # run but wrong here.
        #
        # joint1 (Ry, massless) -> l2 (fixed, tx(1), m=1, r=[0.5,0,0]) ->
        # joint2 (Ry, massless downstream -- nothing after it has mass).
        # l2 has no rotational freedom of its own, so joint2's torque must
        # be exactly zero at *every* configuration, for *any* q1 -- not
        # just at a few sampled points.
        joint1 = Link(ets=ETS(ET.Ry()), m=0, r=[0, 0, 0], name="joint1")
        l2 = Link(ets=ETS(ET.tx(1)), m=1, r=[0.5, 0, 0], parent=joint1, name="l2")
        joint2 = Link(ets=ETS(ET.Ry()), m=0, r=[0, 0, 0], parent=l2, name="joint2")
        robot = ERobot([joint1, l2, joint2], name="joint, sandwiched static link, joint")
        self.assertEqual(robot.n, 2)

        z = np.zeros(robot.n)
        for q0 in (0.0, 0.5, -1.2, pi / 2):
            for q1 in (0.0, 0.7, -2.1, pi):
                tau = robot.rne(np.r_[q0, q1], z, z, gravity=[0, 0, -9.81])
                self.assertAlmostEqual(tau[1], 0.0, places=9)
                # joint1 carries l2's full weight at its fixed 1.5 m offset,
                # independent of q1 -- the standard single-point-mass
                # pendulum formula
                expected_tau0 = -1.0 * 9.81 * 1.5 * cos(q0)
                self.assertAlmostEqual(tau[0], expected_tau0, places=9)

    def test_invdyn_sandwiched_static_link_with_inertia(self):
        # Same as above, but the sandwiched static link also carries a
        # nonzero inertia tensor and an off-axis r, exercising the general
        # CoM/inertia transform into the joint's frame rather than just a
        # simple point mass on the rotation axis.
        joint1 = Link(ets=ETS(ET.Ry()), m=0, r=[0, 0, 0], name="joint1")
        l2 = Link(
            ets=ETS(ET.tx(0.30)),
            m=2,
            r=[0.05, 0, 0],
            I=np.diag([0.001, 0.002, 0.003]),
            parent=joint1,
            name="l2",
        )
        joint2 = Link(ets=ETS(ET.Ry()), m=0, r=[0, 0, 0], parent=l2, name="joint2")
        robot = ERobot([joint1, l2, joint2], name="joint, sandwiched static link 2, joint")
        self.assertEqual(robot.n, 2)

        z = np.zeros(robot.n)
        for q0 in (0.0, 0.5, -1.2):
            for q1 in (0.0, 0.7, -2.1):
                tau = robot.rne(np.r_[q0, q1], z, z, gravity=[0, 0, -9.81])
                self.assertAlmostEqual(tau[1], 0.0, places=9)
                expected_tau0 = -2.0 * 9.81 * 0.35 * cos(q0)
                self.assertAlmostEqual(tau[0], expected_tau0, places=9)

    def test_invdyn_base_mounted_static_link(self):
        # Regression test: a static (fixed) link *before* the first joint,
        # rigidly mounted directly on the immovable base -- e.g. URDF
        # Panda's panda_link0 -- has no joint ancestor at all. Grouping it
        # correctly (attaching it to nothing, since it contributes no joint
        # torque) previously broke the *kinematic* parent lookup for the
        # first joint itself: that joint's own .parent is this static base
        # link, and resolving its upstream group via raw list-membership
        # search found the link in no group at all (since it was
        # deliberately dropped, not kinematically missing) and crashed with
        # IndexError, rather than being treated as having no upstream joint
        # (the same as .parent being None outright).
        base = Link(ets=ETS(), m=5, r=[0.1, 0, 0], name="base")
        joint1 = Link(ets=ETS(ET.Ry()), m=1, r=[0.5, 0, 0], parent=base, name="joint1")
        robot = ERobot([base, joint1], name="base-mounted static link then joint")
        self.assertEqual(robot.n, 1)

        z = np.zeros(robot.n)
        for q in (0.0, 0.5, -1.2, pi / 2):
            tau = robot.rne(np.r_[q], z, z, gravity=[0, 0, -9.81])
            # base is rigidly fixed to the world -- it never moves, so it
            # contributes nothing to any joint torque regardless of its
            # own mass/r
            expected = -1.0 * 9.81 * 0.5 * cos(q)
            self.assertAlmostEqual(tau[0], expected, places=9)

    def test_invdyn_trailing_static_link(self):
        # Regression test for #636: a run of static (fixed) links *after*
        # the last joint -- e.g. a tool-mount flange with no further joint
        # after it, such as URDF Panda's panda_link8 -- was silently
        # dropped from link_groups entirely, excluding its mass/inertia
        # from every torque in the chain rather than raising or warning.
        #
        # joint1 (Ry, massless) -> l2 (fixed, tx(1), m=1, r=[0.5,0,0],
        # trailing -- no joint after it). l2's own CoM sits 1.5 total from
        # the joint (1.0 from l2's own fixed offset + 0.5 from its own r),
        # rotating rigidly with joint1, so gravity torque is the standard
        # single-point-mass pendulum formula -m*g*r*cos(q) at r=1.5.
        joint1 = Link(ets=ETS(ET.Ry()), m=0, r=[0, 0, 0], name="joint1")
        l2 = Link(ets=ETS(ET.tx(1)), m=1, r=[0.5, 0, 0], parent=joint1, name="l2")
        robot = ERobot([joint1, l2], name="joint then trailing static link")
        self.assertEqual(robot.n, 1)

        z = np.zeros(robot.n)
        for q in (0.0, 0.5, -1.2, pi / 2):
            tau = robot.rne(np.r_[q], z, z, gravity=[0, 0, -9.81])
            expected = -1.0 * 9.81 * 1.5 * cos(q)
            self.assertAlmostEqual(tau[0], expected, places=9)

    def test_invdyn_trailing_static_link_with_inertia(self):
        # Same as above, but the trailing static link also carries a
        # nonzero inertia tensor and an off-axis r, exercising the general
        # CoM/inertia transform into the joint's frame rather than just a
        # simple point mass on the rotation axis.
        joint1 = Link(ets=ETS(ET.Ry()), m=0, r=[0, 0, 0], name="joint1")
        l2 = Link(
            ets=ETS(ET.tx(0.30)),
            m=2,
            r=[0.05, 0, 0],
            I=np.diag([0.001, 0.002, 0.003]),
            parent=joint1,
            name="l2",
        )
        robot = ERobot([joint1, l2], name="joint then trailing static link 2")
        self.assertEqual(robot.n, 1)

        z = np.zeros(robot.n)
        for q in (0.0, 0.5, -1.2):
            tau = robot.rne(np.r_[q], z, z, gravity=[0, 0, -9.81])
            expected = -2.0 * 9.81 * 0.35 * cos(q)
            self.assertAlmostEqual(tau[0], expected, places=9)

    def test_payload_end_effector_link(self):
        # The payload belongs on the end-effector link.  When static links
        # follow the last joint, as in URDF models, links[n - 1] is not the
        # end-effector and used to receive the payload instead (#638).
        #
        # joint1 (Ry, massless) -> l2 (fixed, tx(1), m=1, r=[0.5,0,0]), the
        # end-effector.  A 2 kg payload at [0.25,0,0] in l2's frame sits
        # 1.25 from the joint, next to l2's own centre of mass at 1.5.
        joint1 = Link(ets=ETS(ET.Ry()), m=0, r=[0, 0, 0], name="joint1")
        l2 = Link(ets=ETS(ET.tx(1)), m=1, r=[0.5, 0, 0], parent=joint1, name="l2")
        robot = ERobot([joint1, l2], name="joint then trailing static link")
        self.assertEqual(robot.ee_links, [l2])
        self.assertIs(robot.links[robot.n - 1], joint1)

        robot.payload(2, [0.25, 0, 0])

        self.assertEqual(joint1.m, 0)
        z = np.zeros(robot.n)
        for q in (0.0, 0.5, -1.2, pi / 2):
            tau = robot.rne(np.r_[q], z, z, gravity=[0, 0, -9.81])
            expected = -9.81 * (1.0 * 1.5 + 2.0 * 1.25) * cos(q)
            self.assertAlmostEqual(tau[0], expected, places=9)

        robot.payload(0)
        for q in (0.0, 0.5, -1.2):
            tau = robot.rne(np.r_[q], z, z, gravity=[0, 0, -9.81])
            self.assertAlmostEqual(tau[0], -9.81 * 1.5 * cos(q), places=9)

    def test_payload_several_end_effectors(self):
        # With more than one end-effector the choice is ambiguous, and the
        # payload stays on links[n - 1] as it always has
        joint1 = Link(ets=ETS(ET.Ry()), m=0, r=[0, 0, 0], name="joint1")
        a = Link(ets=ETS(ET.tx(1)), m=1, r=[0.5, 0, 0], parent=joint1, name="a")
        b = Link(ets=ETS(ET.tx(-1)), m=1, r=[-0.5, 0, 0], parent=joint1, name="b")
        robot = ERobot([joint1, a, b], name="two end-effectors")
        self.assertEqual(len(robot.ee_links), 2)

        robot.payload(2, [0.25, 0, 0])

        self.assertEqual(joint1.m, 2)
        nt.assert_array_almost_equal(joint1.r, [0.25, 0, 0])
        self.assertEqual(a.m, 1)
        self.assertEqual(b.m, 1)


class TestERobot2(unittest.TestCase):
    def test_plot(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.plot(robot.qz, block=False, name=True)
        e.close()

    def test_teach(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.teach(robot.qz, block=False)
        e.close()

        e = robot.teach(robot.qz, block=False)
        e.close()

    def test_plot_with_vellipse(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.plot(
            robot.qb, block=False, name=True, vellipse=True, limits=[1, 2, 1, 2]
        )
        e.step()
        e.close()

    def test_plot_with_fellipse(self):
        robot = rtb.models.ETS.Planar2()
        e = robot.plot(
            robot.qz, block=False, name=True, dellipse=True, limits=[1, 2, 1, 2]
        )
        e.step()
        e.close()

    def test_base(self):
        robot = rtb.models.ETS.Planar2()
        nt.assert_almost_equal(robot.base.A, sm.SE2().A)

    def test_jacobe(self):
        robot = rtb.models.ETS.Planar2()
        J = robot.jacobe(robot.qz)

        a1 = np.array([[0.0, 0.0], [2.0, 1.0], [1.0, 1.0]])

        nt.assert_almost_equal(J, a1)

    @unittest.skipUnless(_sympy, "sympy not installed")
    def test_symdyn(self):

        a1, a2, r1, r2, m1, m2, g = symbols("a1 a2 r1 r2 m1 m2 g")
        link1 = Link(ET.Ry(flip=True), m=m1, r=[r1, 0, 0], name="link0")
        link2 = Link(ET.tx(a1) * ET.Ry(flip=True), m=m2, r=[r2, 0, 0], name="link1")
        robot = ERobot([link1, link2])

        q = symbols("q:2")
        qd = symbols("qd:2")
        qdd = symbols("qdd:2")
        Q = robot.rne(q, qd, qdd, gravity=[0, 0, g], symbolic=True)

        self.assertEqual(
            str(Q[0]),
            "a1**2*m2*qd0**2*sin(q1)*cos(q1) + a1*qd0*(-a1*m2*qd0*cos(q1) - m2*r2*(qd0 + qd1))*sin(q1) + g*m1*r1*cos(q0) + m1*qdd0*r1**2 + m2*r2**2*(qdd0 + qdd1) - m2*r2*(-a1*qd0*qd1*sin(q1) - a1*qdd0*cos(q1) + g*sin(q0)*sin(q1) - g*cos(q0)*cos(q1)) - (a1*sin(q1)**2 + a1*cos(q1)**2)*(m2*(a1*qd0*qd1*cos(q1) - a1*qdd0*sin(q1) - g*sin(q0)*cos(q1) - g*sin(q1)*cos(q0)) + (qd0 + qd1)*(-a1*m2*qd0*cos(q1) - m2*r2*(qd0 + qd1)))*sin(q1) - (a1*sin(q1)**2 + a1*cos(q1)**2)*(-a1*m2*qd0*(-qd0 - qd1)*sin(q1) - m2*r2*(qdd0 + qdd1) + m2*(-a1*qd0*qd1*sin(q1) - a1*qdd0*cos(q1) + g*sin(q0)*sin(q1) - g*cos(q0)*cos(q1)))*cos(q1)",
        )
        self.assertEqual(
            str(Q[1]),
            "a1**2*m2*qd0**2*sin(q1)*cos(q1) + a1*qd0*(-a1*m2*qd0*cos(q1) - m2*r2*(qd0 + qd1))*sin(q1) + m2*r2**2*(qdd0 + qdd1) - m2*r2*(-a1*qd0*qd1*sin(q1) - a1*qdd0*cos(q1) + g*sin(q0)*sin(q1) - g*cos(q0)*cos(q1))",
        )


if __name__ == "__main__":  # pragma nocover
    unittest.main()
