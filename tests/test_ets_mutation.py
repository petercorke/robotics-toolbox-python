"""Kinematics must follow joint indices after list-like ETS edits."""

import numpy as np
import numpy.testing as nt
import pytest

from roboticstoolbox import ET, ET2, ETS, ETS2


@pytest.mark.parametrize("et, ets_type, rotation", [(ET, ETS, "Rz"), (ET2, ETS2, "R")])
@pytest.mark.parametrize(
    "operation",
    ["replace", "slice", "delete", "insert", "sequence", "pop", "reverse", "data"],
)
def test_mutation_refreshes_joint_indices(et, ets_type, rotation, operation) -> None:
    rotate = getattr(et, rotation)
    sequence = (
        rotate(jindex=0) * et.tx(1) * rotate(jindex=2) * et.tx(2) * rotate(jindex=4)
    )
    # Populate the cached index array just as the first FK call does.
    sequence.fkine([0.1, 0.2, 0.3])

    if operation == "replace":
        sequence[-1] = rotate(jindex=6)
    elif operation == "slice":
        sequence[2:] = [et.tx(2), rotate(jindex=6)]
    elif operation == "delete":
        del sequence[0]
    elif operation == "insert":
        sequence.insert(2, rotate(jindex=6))
    elif operation == "sequence":
        sequence.insert(2, ets_type([rotate(jindex=6), et.tx(0.5)]))
    elif operation == "pop":
        sequence.pop()
    elif operation == "reverse":
        sequence.reverse()
    else:
        sequence.data = [rotate(jindex=6), et.tx(0.5)]

    # A freshly constructed sequence has the same transforms and no stale cache.
    expected = ets_type(list(sequence))
    q = np.linspace(0.1, 0.3, sequence.n)
    nt.assert_array_equal(sequence.jindices, expected.jindices)
    nt.assert_allclose(sequence.fkine(q).A, expected.fkine(q).A)
    nt.assert_allclose(sequence.jacob0(q), expected.jacob0(q))
