import numpy as np
import numpy.testing as nt
import pytest

import roboticstoolbox as rtb

pytest.importorskip("bdsim")

from roboticstoolbox.blocks.arm import JTraj  # noqa: E402


class _State:
    """The simulation state that a block's ``start`` method is given"""

    tf = 2.0


def _block(*args, **kwargs):
    block = JTraj(*args, T=2.0, **kwargs)
    block.start(_State())
    return block


def _output(block, t):
    return [np.asarray(x) for x in block.output(t, None, None)]


Q0 = np.array([0.0, 0.5, -1.0])
QF = np.array([1.0, 2.0, 0.5])


@pytest.mark.parametrize("t", [0.0, 0.3, 1.0, 1.7, 2.0])
def test_default_block_matches_jtraj(t):
    # the final velocity was set to qf, not to zero, so the block ended moving
    block = _block(Q0, QF)
    ref = rtb.jtraj(Q0, QF, np.array([0.0, 0.3, 1.0, 1.7, 2.0]) / 2.0)
    i = [0.0, 0.3, 1.0, 1.7, 2.0].index(t)
    q, qd, qdd = _output(block, t)
    nt.assert_allclose(q, ref.q[i], atol=1e-12)
    nt.assert_allclose(qd, ref.qd[i] / 2.0, atol=1e-12)
    nt.assert_allclose(qdd, ref.qdd[i] / 4.0, atol=1e-12)


def test_default_block_comes_to_rest_at_the_goal():
    block = _block(Q0, QF)
    nt.assert_allclose(block.qdf, 0)
    q, qd, qdd = _output(block, 2.0)
    nt.assert_allclose(q, QF, atol=1e-12)
    nt.assert_allclose(qd, 0, atol=1e-12)


def test_boundary_velocities_are_used():
    qd0 = np.array([0.1, -0.2, 0.3])
    qdf = np.array([-0.4, 0.5, 0.6])
    block = _block(Q0, QF, qd0=qd0, qdf=qdf)
    nt.assert_allclose(block.qd0, qd0)
    nt.assert_allclose(block.qdf, qdf)
    q, qd, _ = _output(block, 0.0)
    nt.assert_allclose(qd, qd0, atol=1e-12)
    q, qd, _ = _output(block, 2.0)
    nt.assert_allclose(q, QF, atol=1e-12)
    nt.assert_allclose(qd, qdf, atol=1e-12)


def test_boundary_velocities_accept_a_list():
    # getvector was not imported, so any qd0 or qdf raised NameError
    block = _block(Q0, QF, qd0=[0, 0, 0], qdf=[0, 0, 0])
    nt.assert_allclose(block.qd0, 0)


@pytest.mark.parametrize("name", ["qd0", "qdf"])
def test_wrong_size_boundary_velocity(name):
    with pytest.raises(ValueError, match=name):
        JTraj(Q0, QF, **{name: [0.0, 0.0]})
