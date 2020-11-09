import numpy as np
from scipy.linalg import lu


def jsingu(J):

    # convert to row-echelon form
    P, L, U = lu(J)
    U = np.where(abs(U) < 100 * np.finfo(np.float64).eps, False, True)

    for j in range(J.shape[0]):
        if not U[j, j]:
            print(
                f'joint {j} is dependent on joint ' +
                ', '.join(
                    [str(i) for i in range(j) if all(U[:, j] == U[:, i])]))


if __name__ == "__main__":   # pragma nocover
    import roboticstoolbox as rtb

    puma = rtb.models.DH.Puma560()
    J = puma.jacob0(puma.qr)

    jsingu(J)
