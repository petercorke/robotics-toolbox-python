"""String joint descriptors for ET and ET2, such as "q2", "-theta3" and "A1_"

A string for ``param`` that is not a number names a variable joint:

- a leading ``-`` (or the Unicode minus) sets ``flip``, a leading ``+`` is ignored
- the first run of digits is the joint index, counted from zero
- a trailing underscore says the name counts from one, so the index is one less
- the name is shown when the ET is displayed, without that underscore
"""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import ET, ET2, ERobot, ETS

MINUS = "−"  # Unicode minus sign

# descriptor, jindex, flip, str(ET.Rz(descriptor))
CASES = [
    ("q0", 0, False, "Rz(q0)"),
    ("q12", 12, False, "Rz(q12)"),
    ("-q1", 1, True, "Rz(-q1)"),
    ("+q1", 1, False, "Rz(+q1)"),
    ("theta2", 2, False, "Rz(theta2)"),
    ("θ_3", 3, False, "Rz(θ_3)"),
    ("q(3)", 3, False, "Rz(q(3))"),
    ("-q(3)", 3, True, "Rz(-q(3))"),
    (f"{MINUS}q1", 1, True, f"Rz({MINUS}q1)"),
    # one-based names, trailing underscore
    ("A1_", 0, False, "Rz(A1)"),
    ("A6_", 5, False, "Rz(A6)"),
    ("A10_", 9, False, "Rz(A10)"),
    ("A01_", 0, False, "Rz(A01)"),
    ("-A1_", 0, True, "Rz(-A1)"),
    (f"{MINUS}A2_", 1, True, f"Rz({MINUS}A2)"),
    ("+A2_", 1, False, "Rz(+A2)"),
    ("θ2_", 1, False, "Rz(θ2)"),
    ("θ_3_", 2, False, "Rz(θ_3)"),
    ("q(3)_", 2, False, "Rz(q(3))"),
    # without the underscore the digits are the index, as written
    ("A1", 1, False, "Rz(A1)"),
    ("A6", 6, False, "Rz(A6)"),
    # no digits: auto-numbered, and the underscore means nothing
    ("joint", None, False, "Rz(joint)"),
    ("joint_", None, False, "Rz(joint_)"),
    ("-joint", None, True, "Rz(-joint)"),
]


@pytest.mark.parametrize("descriptor, jindex, flip, text", CASES)
def test_descriptor_sets_index_flip_and_name(descriptor, jindex, flip, text) -> None:
    et = ET.Rz(descriptor)
    assert et.isjoint
    assert et.jindex == jindex
    assert et.isflip is flip
    assert str(et) == text


@pytest.mark.parametrize("make", [ET.Rx, ET.Ry, ET.Rz, ET.tx, ET.ty, ET.tz])
def test_every_elementary_transform_takes_a_descriptor(make) -> None:
    et = make("-A3_")
    assert et.isjoint and et.jindex == 2 and et.isflip


def test_planar_et2_takes_a_descriptor() -> None:
    et = ET2.R("-A2_")
    assert et.isjoint and et.jindex == 1 and et.isflip


@pytest.mark.parametrize("descriptor", ["A0_", "-A0_", "q0_", "θ0_"])
def test_one_based_name_cannot_be_zero(descriptor: str) -> None:
    with pytest.raises(ValueError, match="count from one"):
        ET.Rz(descriptor)


@pytest.mark.parametrize("keyword", [{"jindex": 0}, {"flip": True}])
def test_descriptor_and_keyword_are_exclusive(keyword) -> None:
    with pytest.raises(ValueError, match="cannot specify"):
        ET.Rz("A1_", **keyword)


@pytest.mark.parametrize("value, expected", [("1.5", 1.5), ("-2", -2.0), ("1_0", 10.0)])
def test_a_numeric_string_is_a_constant(value: str, expected: float) -> None:
    et = ET.tz(value)
    assert not et.isjoint
    nt.assert_allclose(et.A()[2, 3], expected)


def test_flipped_descriptor_negates_the_joint_variable() -> None:
    q = 0.4
    nt.assert_allclose(ET.Rz("-A1_").A(q), ET.Rz("A1_").A(-q), atol=1e-12)
    nt.assert_allclose(ET.Rz("A1_").A(q), ET.Rz(jindex=0).A(q), atol=1e-12)


def _kuka(names: list[str]) -> ETS:
    # six revolute joints, alternating z and x axes, with link offsets
    ets = ETS()
    for k, name in enumerate(names):
        axis = ET.Rz if k % 2 == 0 else ET.Rx
        ets = ets * axis(name) * ET.tz(0.2 + 0.05 * k) * ET.tx(0.1)
    return ets


def test_one_based_names_build_the_same_robot_as_zero_based() -> None:
    one = ETS(_kuka([f"A{i}_" for i in range(1, 7)]))
    zero = ETS(_kuka([f"q{i}" for i in range(6)]))

    assert [j.jindex for j in one.joints()] == [0, 1, 2, 3, 4, 5]
    assert [str(j) for j in one.joints()][:2] == ["Rz(A1)", "Rx(A2)"]
    assert "A1" in str(one) and "A6" in str(one) and "_" not in str(one)

    r1, r0 = ERobot(one), ERobot(zero)
    assert r1.n == r0.n == 6
    q = np.array([0.1, -0.4, 0.7, 0.2, -0.9, 0.5])
    nt.assert_allclose(r1.fkine(q).A, r0.fkine(q).A, atol=1e-12)
    nt.assert_allclose(r1.jacob0(q), r0.jacob0(q), atol=1e-12)


def test_flipped_one_based_joint_in_a_robot() -> None:
    names = [f"A{i}_" for i in range(1, 7)]
    plain = ERobot(ETS(_kuka(names)))
    flipped = ERobot(ETS(_kuka(["-A1_"] + names[1:])))
    q = np.array([0.3, 0.1, -0.2, 0.4, 0.5, -0.6])
    q_flipped = q.copy()
    q_flipped[0] = -q[0]
    nt.assert_allclose(flipped.fkine(q).A, plain.fkine(q_flipped).A, atol=1e-12)


def test_one_based_names_without_underscore_are_rejected_with_a_hint() -> None:
    ets = ETS(_kuka([f"A{i}" for i in range(1, 7)]))
    assert [j.jindex for j in ets.joints()] == [1, 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="trailing underscore") as info:
        ERobot(ets)
    assert "out of range" in str(info.value)


def test_other_out_of_range_indices_get_no_hint() -> None:
    ets = ETS(_kuka(["q0", "q1", "q2", "q3", "q4", "q9"]))
    with pytest.raises(ValueError, match="out of range") as info:
        ERobot(ets)
    assert "underscore" not in str(info.value)
