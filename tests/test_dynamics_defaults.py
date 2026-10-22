"""
Default arguments of the dynamics methods are None, not a mutable object that is
shared by every call.
"""

import inspect

import numpy as np
import numpy.testing as nt
import pytest

import roboticstoolbox as rtb
from roboticstoolbox.robot.Dynamics import DynamicsMixin


@pytest.mark.parametrize(
    "method, name",
    [("fdyn", "Q_args"), ("fdyn", "solver_args"), ("payload", "p")],
)
def test_default_is_none(method, name):
    default = inspect.signature(getattr(DynamicsMixin, method)).parameters[name].default
    assert default is None


def test_payload_without_a_position_is_at_the_origin():
    puma = rtb.models.DH.Puma560()
    q = puma.qn
    puma.payload(2.0)
    default = puma.gravload(q)
    puma.payload(0)
    puma.payload(2.0, [0, 0, 0])
    nt.assert_allclose(default, puma.gravload(q))
    puma.payload(0)


def test_payload_zero_mass_is_a_no_op():
    puma = rtb.models.DH.Puma560()
    before = puma.gravload(puma.qn)
    puma.payload(0)
    nt.assert_allclose(puma.gravload(puma.qn), before)


def test_fdyn_without_optional_arguments():
    puma = rtb.models.DH.Puma560()
    for _ in range(2):
        t, q, qd = puma.fdyn(0.1, puma.qn, dt=0.05)
        assert q.shape[1] == 6
        assert np.all(np.isfinite(q))
