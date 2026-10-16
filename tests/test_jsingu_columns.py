"""Singularity diagnostics must refer to the caller's Jacobian columns."""

import re

import numpy as np
import numpy.testing as nt
import pytest

import roboticstoolbox as rtb


@pytest.mark.parametrize(
    "jacobian, expected",
    [
        ([[1, 0, 0, 0], [0, 0, 1, 2]], "column 3 = + 2 column_2"),
        ([[0, 1, 3], [0, 0, 0]], "column 2 =3 column_1"),
        ([[1, 0, 0, 2], [0, 0, 1, -3]], "column 3 =2 column_0 - 3 column_2"),
    ],
)
def test_dependent_columns_keep_original_indices(jacobian, expected, capsys):
    matrix = np.array(jacobian, dtype=float)
    rtb.jsingu(matrix)
    lines = capsys.readouterr().out.splitlines()

    nt.assert_equal(expected in lines, True)
    for line in lines:
        header, expression = line.split("=", 1)
        dependent = int(header.split()[1])
        reconstruction = np.zeros(matrix.shape[0])
        for sign, magnitude, index in re.findall(
            r"([+-]?)\s*(\d+(?:\.\d*)?(?:e[+-]?\d+)?) column_(\d+)", expression
        ):
            coefficient = float(magnitude) * (-1 if sign == "-" else 1)
            reconstruction += coefficient * matrix[:, int(index)]
        nt.assert_allclose(reconstruction, matrix[:, dependent], atol=1e-14)


def test_full_rank_matrix_has_no_dependent_columns(capsys):
    rtb.jsingu(np.eye(3))
    nt.assert_equal(capsys.readouterr().out, "")


@pytest.mark.parametrize("spherical_chain", [False, True])
def test_robot_jacobian_reports_original_joint_columns(capsys, spherical_chain):
    if spherical_chain:
        robot = rtb.Robot(rtb.ET.Rz() * rtb.ET.Rz() * rtb.ET.Rx() * rtb.ET.Rx())
    else:
        robot = rtb.models.DH.Puma560()
    jacobian = robot.jacob0(np.zeros(robot.n))
    rtb.jsingu(jacobian)
    lines = capsys.readouterr().out.splitlines()

    nt.assert_equal(bool(lines), True)
    for line in lines:
        header, expression = line.split("=", 1)
        dependent = int(header.split()[1])
        reconstruction = np.zeros(6)
        for sign, magnitude, index in re.findall(
            r"([+-]?)\s*(\d+(?:\.\d*)?(?:e[+-]?\d+)?) column_(\d+)", expression
        ):
            coefficient = float(magnitude) * (-1 if sign == "-" else 1)
            reconstruction += coefficient * jacobian[:, int(index)]
        nt.assert_allclose(reconstruction, jacobian[:, dependent], atol=1e-3, rtol=1e-3)
