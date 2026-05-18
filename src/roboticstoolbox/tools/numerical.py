import numpy as np
from spatialmath import base

def jacobian_numerical(f, x, dx=1e-8, N=0):
    r"""
    Numerically compute Jacobian of function

    :param f: the function, returns an m-vector
    :type f: callable
    :param x: function argument
    :type x: ndarray(n)
    :param dx: the numerical perturbation, defaults to 1e-8
    :type dx: float, optional
    :param N: function returns SE(N) matrix, defaults to 0
    :type N: int, optional
    :return: Jacobian matrix
    :rtype: ndarray(m,n)

    Computes a numerical approximation to the Jacobian for ``f(x)`` where 
    :math:`f: \mathbb{R}^n \mapsto \mathbb{R}^m`.

    Uses first-order difference :math:`J[:,i] = (f(x + dx) - f(x)) / dx`.

    If ``N`` is 2 or 3, then it is assumed that the function returns
    an SE(N) matrix which is converted into a Jacobian column comprising the
    translational Jacobian followed by the rotational Jacobian.
    """

    Jcol = []
    J0 = f(x)
    I = np.eye(len(x))
    f0 = f(x)
    for i in range(len(x)):
        fi = f(x + I[:,i] * dx)
        Ji = (fi - f0) / dx

        if N > 0:
            t = Ji[:N,N]
            r = base.vex(Ji[:N,:N] @ J0[:N,:N].T)
            Ji = np.r_[t, r]
        
        Jcol.append(Ji)

    return np.c_[Jcol].T


def hessian_numerical(J, x, dx=1e-8):
    r"""
    Numerically compute Hessian of Jacobian function

    :param J: the Jacobian function, returns an ndarray(m,n)
    :type J: callable
    :param x: function argument
    :type x: ndarray(n)
    :param dx: the numerical perturbation, defaults to 1e-8
    :type dx: float, optional
    :return: Hessian matrix
    :rtype: ndarray(m,n,n)

    Computes a numerical approximation to the Hessian for ``J(x)`` where 
    :math:`f: \mathbb{R}^n  \mapsto \mathbb{R}^{m \times n}`

    Uses first-order difference :math:`H[:,:,i] = (J(x + dx) - J(x)) / dx`.
    """

    I = np.eye(len(x))
    Hcol = []
    J0 = J(x)
    for i in range(len(x)):

        Ji = J(x + I[:,i] * dx)
        Hi = (Ji - J0) / dx

        Hcol.append(Hi)
        
    return np.stack(Hcol, axis=2)


if __name__ == "__main__":

    import roboticstoolbox as rtb
    np.set_printoptions(linewidth=120, formatter={'float': lambda x: f"{x:8.4g}" if abs(x) > 1e-10 else f"{0:8.4g}"})

    robot = rtb.models.DH.Puma560()
    q = robot.qn

    J = jacobian_numerical(lambda q: robot.fkine(q).A, q, N=3)
    print(J)
    print(robot.jacob0(q))

    H = hessian_numerical(robot.jacob0, q)
    print(H)
    print(robot.ets().hessian0(q))