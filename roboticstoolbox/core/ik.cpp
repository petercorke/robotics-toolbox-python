/* ik.cpp */

#include "linalg.h"
#include "methods.h"
#include "ik.h"
#include "structs.h"

#include <Python.h>
#include <math.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>

extern "C"
{

    void _IK_LM_Chan(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        double lambda, MapVectorX we)
    {
        int iter = 0;

        double *np_Te = (double *)PyMem_RawCalloc(16, sizeof(double));
        MapMatrix4dc Te(np_Te);

        double *np_J = (double *)PyMem_RawCalloc(6 * ets->n, sizeof(double));
        MapMatrixJc J(np_J, 6, ets->n);

        double *np_e = (double *)PyMem_RawCalloc(6, sizeof(double));
        MapVectorX e(np_e, 6);

        Matrix6dc We;

        Eigen::MatrixXd Wn(ets->n, ets->n);
        Eigen::MatrixXd EyeN = Eigen::MatrixXd::Identity(ets->n, ets->n);

        VectorX g(ets->n);

        // Set we
        if (we.size() == 6)
        {
            We = we.asDiagonal();
        }
        else
        {
            We = Matrix6dc::Identity();
        }

        // Set the first q0
        if (q0.size() == ets->n)
        {
            q = q0;
        }
        else
        {
            q = _rand_q(ets);
        }

        // Global search up to slimit
        while (*search < slimit)
        {

            while (iter < ilimit)
            {
                // Current pose Te
                _ETS_fkine(ets, q.data(), (double *)NULL, NULL, Te);

                // Angle axis error e
                _angle_axis(Te, Tep, e);

                // Squared error E
                *E = 0.5 * e.transpose() * We * e;

                if (*E < tol)
                {
                    // We have arrived

                    // Check for joint limit violation
                    if (reject_jl)
                    {
                        *solution = _check_lim(ets, q);
                    }
                    else
                    {
                        *solution = 1;
                    }

                    break;
                }

                // Jacobian Matric J
                _ETS_jacob0(ets, q.data(), (double *)NULL, J);

                // Weighting matrix Wn
                Wn = lambda * *E * EyeN;

                // The vector g
                g = J.transpose() * We * e;

                // Work out the joint velocity qd
                q += (J.transpose() * We * J + Wn).inverse() * g;

                iter += 1;
            }

            if (*solution)
            {
                *it += iter;
                break;
            }

            *it += iter;
            iter = 0;
            *search += 1;
            q = _rand_q(ets);
        }

        free(np_e);
        free(np_Te);
        free(np_J);
    }

    int _check_lim(ETS *ets, MapVectorX q)
    {
        for (int i = 0; i < ets->n; i++)
        {
            if (q(i) < ets->qlim_l[i] || q(i) > ets->qlim_h[i])
            {
                // std::cout << "Joint limit: " << q.transpose() << "  :  " << q(i) << "\n";
                return 0;
            }
        }

        return 1;
    }

    void _angle_axis(MapMatrix4dc Te, Matrix4dc Tep, MapVectorX e)
    {
        double num, li_norm, R_tr, ang;
        Matrix3dc R;
        Vector3 li;

        // e[:3] = Tep[:3, 3] - Te[:3, 3]
        e.block<3, 1>(0, 0) = Tep.block<3, 1>(0, 3) - Te.block<3, 1>(0, 3);

        // R = Tep.R @ T.R.T
        // R = Tep[:3, :3] @ T[:3, :3].T
        R = Tep.block<3, 3>(0, 0) * Te.block<3, 3>(0, 0).transpose();

        // li = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]);
        li << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);

        // if base.iszerovec(li)
        li_norm = li.norm();

        R_tr = R.trace();
        if (li_norm < 1e-6)
        {
            // diagonal matrix case
            // if np.trace(R) > 0
            if (R_tr > 0)
            {
                // (1,1,1) case
                // a = np.zeros((3, ));
                e.block<3, 1>(3, 0) << 0, 0, 0;
            }
            else
            {
                // a = np.pi / 2 * (np.diag(R) + 1);
                e(3) = PI_2 * (R(0, 0) + 1);
                e(4) = PI_2 * (R(1, 1) + 1);
                e(5) = PI_2 * (R(2, 2) + 1);
            }
        }
        else
        {
            // non-diagonal matrix case
            // a = math.atan2(li_norm, np.trace(R) - 1) * li / li_norm
            ang = atan2(li_norm, R_tr - 1);
            e.block<3, 1>(3, 0) = ang * li / li_norm;
        }
    }

    VectorX _rand_q(ETS *ets)
    {
        Eigen::Map<Eigen::ArrayXd> qlim_l(ets->qlim_l, ets->n);
        Eigen::Map<Eigen::ArrayXd> q_range2(ets->q_range2, ets->n);

        VectorX q = VectorX::Random(ets->n);

        q = (q.array() + 1) * q_range2;
        q = q.array() + qlim_l;

        return q;
    }

} /* extern "C" */