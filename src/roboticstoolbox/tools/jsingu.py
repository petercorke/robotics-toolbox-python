from __future__ import annotations

from typing import Any

import numpy as np

# def jsingu(J):

#     # convert to row-echelon form
#     P, L, U = lu(J)
#     U = np.where(abs(U) < 100 * np.finfo(np.float64).eps, False, True)

#     for j in range(J.shape[0]):
#         if not U[j, j]:
#             print(
#                 f'joint {j} is dependent on joint ' +
#                 ', '.join(
#                     [str(i) for i in range(j) if all(U[:, j] == U[:, i])]))


def jsingu(J: Any) -> None:
    """Print dependencies using the original Jacobian column indices.

    :param J: Jacobian matrix to analyse
    """

    indep_columns = np.empty((J.shape[0], 0))
    indep_indices: list[int] = []
    rank = 0
    for j in range(J.shape[1]):
        temp = np.column_stack((indep_columns, J[:, j]))
        temp_rank = np.linalg.matrix_rank(temp)
        if temp_rank > rank:
            # this column is independent
            rank = temp_rank
            indep_columns = temp
            indep_indices.append(j)

        else:
            s = f"column {j} ="
            c = np.linalg.pinv(indep_columns) @ J[:, j]
            for i, ci in enumerate(c):
                if abs(ci) > 10 * np.finfo(np.float64).eps:
                    if ci < 0:
                        s += " - "
                    elif i > 0:
                        s += " + "
                    s += f"{abs(ci):.3g} column_{indep_indices[i]}"
            print(s)


if __name__ == "__main__":  # pragma nocover
    import roboticstoolbox as rtb

    # robot = rtb.models.DH.Puma560()
    robot = rtb.models.URDF.UR5()  # type: ignore[operator]  # class, not module
    J = robot.jacob0(robot.qr)

    jsingu(J)
