/* ik.cpp */

#include "linalg.h"
#include "methods.h"
#include "ik.h"
#include "structs.h"

#include <Python.h>
#include <math.h>
#include <iostream>
#include <Eigen/Dense>
// #include <Eigen/QR>
// #include <Eigen/Core>
// #include <Eigen/LU>
// #include <Eigen/SVD>

extern "C"
{

    void _IK_GN(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        MapVectorX we, int use_pinv, double pinv_damping)
    {
        int iter = 1;

        double *np_Te = (double *)PyMem_RawCalloc(16, sizeof(double));
        MapMatrix4dc Te(np_Te);

        double *np_J = (double *)PyMem_RawCalloc(6 * ets->n, sizeof(double));
        MapMatrixJc J(np_J, 6, ets->n);

        double *np_Jw = (double *)PyMem_RawCalloc(ets->n * ets->n, sizeof(double));
        Eigen::Map<Eigen::MatrixXd> Jw(np_Jw, ets->n, ets->n);

        double *np_e = (double *)PyMem_RawCalloc(6, sizeof(double));
        MapVectorX e(np_e, 6);

        Matrix6dc We;
        double *np_pinv;
        Eigen::Map<Eigen::MatrixXd> pinv(NULL, 0, 0);

        if (use_pinv)
        {
            np_pinv = (double *)PyMem_RawCalloc(ets->n * ets->n, sizeof(double));
            new (&pinv) Eigen::Map<Eigen::MatrixXd>(np_pinv, ets->n, ets->n);
        }

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
            _rand_q(ets, q);
        }

        // Global search up to slimit
        while (*search <= slimit)
        {

            while (iter <= ilimit)
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

                    // wrap q to +- pi
                    for (int i = 0; i < ets->n; i++)
                    {
                        q(i) = std::fmod(q(i) + PI, PI_x2) - PI;
                    }

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

                g = J.transpose() * We * e;
                Jw = J.transpose() * We * J;

                if (use_pinv)
                {
                    Eigen::BDCSVD<Eigen::MatrixXd> svd(Jw, Eigen::ComputeFullU | Eigen::ComputeFullV);
                    q += svd.solve(g);
                }
                else
                {
                    q += Jw.colPivHouseholderQr().solve(g);
                }

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
            _rand_q(ets, q);
        }

        free(np_e);
        free(np_Te);
        free(np_J);

        if (use_pinv)
        {
            free(np_pinv);
        }
    }

    void _IK_NR(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        MapVectorX we, int use_pinv, double pinv_damping)
    {
        int iter = 1;

        double *np_Te = (double *)PyMem_RawCalloc(16, sizeof(double));
        MapMatrix4dc Te(np_Te);

        double *np_J = (double *)PyMem_RawCalloc(6 * ets->n, sizeof(double));
        Eigen::Map<Eigen::MatrixXd> J(np_J, 6, ets->n);

        double *np_e = (double *)PyMem_RawCalloc(6, sizeof(double));
        MapVectorX e(np_e, 6);

        Matrix6dc We;
        double *np_J_pinv;
        Eigen::Map<Eigen::MatrixXd> J_pinv(NULL, 0, 0);

        if (ets->n != 6)
        {
            use_pinv = 1;
        }

        if (use_pinv)
        {
            np_J_pinv = (double *)PyMem_RawCalloc(ets->n * 6, sizeof(double));
            new (&J_pinv) Eigen::Map<Eigen::MatrixXd>(np_J_pinv, ets->n, 6);
        }

        Eigen::MatrixXd Wn(ets->n, ets->n);
        Eigen::MatrixXd EyeN = Eigen::MatrixXd::Identity(ets->n, ets->n);

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
            _rand_q(ets, q);
        }

        // Global search up to slimit
        while (*search <= slimit)
        {

            while (iter <= ilimit)
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

                    // wrap q to +- pi
                    for (int i = 0; i < ets->n; i++)
                    {
                        q(i) = std::fmod(q(i) + PI, PI_x2) - PI;
                    }

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

                // robot.q += np.linalg.inv(J) @ e

                if (use_pinv)
                {
                    // Work out the joint velocity qd
                    _pseudo_inverse(J, J_pinv, pinv_damping);
                    q += J_pinv * e;
                }
                else
                {
                    q += J.inverse() * e;
                }

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
            _rand_q(ets, q);
        }

        free(np_e);
        free(np_Te);
        free(np_J);

        if (use_pinv)
        {
            free(np_J_pinv);
        }
    }

    void _IK_LM_Chan(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        double lambda, MapVectorX we)
    {
        int iter = 1;

        // std::cout << Tep << "\n";
        // std::cout << std::endl;

        double *np_Te = (double *)PyMem_RawCalloc(16, sizeof(double));
        MapMatrix4dc Te(np_Te);

        double *np_J = (double *)PyMem_RawCalloc(6 * ets->n, sizeof(double));
        Eigen::Map<Eigen::MatrixXd> J(np_J, 6, ets->n);

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
            _rand_q(ets, q);
        }

        // Global search up to slimit
        while (*search <= slimit)
        {

            while (iter <= ilimit)
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

                    // wrap q to +- pi
                    for (int i = 0; i < ets->n; i++)
                    {
                        q(i) = std::fmod(q(i) + PI, PI_x2) - PI;
                    }

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
                // q += (J.transpose() * We * J + Wn).colPivHouseholderQr().solve(g);

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
            _rand_q(ets, q);
        }

        free(np_e);
        free(np_Te);
        free(np_J);
    }

    void _IK_LM_Wampler(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        double lambda, MapVectorX we)
    {
        int iter = 1;

        double *np_Te = (double *)PyMem_RawCalloc(16, sizeof(double));
        MapMatrix4dc Te(np_Te);

        double *np_J = (double *)PyMem_RawCalloc(6 * ets->n, sizeof(double));
        Eigen::Map<Eigen::MatrixXd> J(np_J, 6, ets->n);

        double *np_e = (double *)PyMem_RawCalloc(6, sizeof(double));
        MapVectorX e(np_e, 6);

        Matrix6dc We;

        Eigen::MatrixXd Wn = lambda * Eigen::MatrixXd::Identity(ets->n, ets->n);

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
            _rand_q(ets, q);
        }

        // Global search up to slimit
        while (*search <= slimit)
        {

            while (iter <= ilimit)
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

                    // wrap q to +- pi
                    for (int i = 0; i < ets->n; i++)
                    {
                        q(i) = std::fmod(q(i) + PI, PI_x2) - PI;
                    }

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

                // The vector g
                g = J.transpose() * We * e;

                // Work out the joint velocity qd
                q += (J.transpose() * We * J + Wn).inverse() * g;
                // q += (J.transpose() * We * J + Wn).colPivHouseholderQr().solve(g);

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
            _rand_q(ets, q);
        }

        free(np_e);
        free(np_Te);
        free(np_J);
    }

    void _IK_LM_Sugihara(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        double lambda, MapVectorX we)
    {
        int iter = 1;

        double *np_Te = (double *)PyMem_RawCalloc(16, sizeof(double));
        MapMatrix4dc Te(np_Te);

        double *np_J = (double *)PyMem_RawCalloc(6 * ets->n, sizeof(double));
        Eigen::Map<Eigen::MatrixXd> J(np_J, 6, ets->n);

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
            _rand_q(ets, q);
        }

        // Global search up to slimit
        while (*search <= slimit)
        {

            while (iter <= ilimit)
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

                    // wrap q to +- pi
                    for (int i = 0; i < ets->n; i++)
                    {
                        q(i) = std::fmod(q(i) + PI, PI_x2) - PI;
                    }

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
                Wn = *E * EyeN + lambda * EyeN;

                // The vector g
                g = J.transpose() * We * e;

                // Work out the joint velocity qd
                q += (J.transpose() * We * J + Wn).inverse() * g;
                // q += (J.transpose() * We * J + Wn).colPivHouseholderQr().solve(g);

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
            _rand_q(ets, q);
        }

        free(np_e);
        free(np_Te);
        free(np_J);
    }

    void _pseudo_inverse(Eigen::Map<Eigen::MatrixXd> J, Eigen::Map<Eigen::MatrixXd> J_pinv, double damping)
    {
        Eigen::JacobiSVD<Eigen::MatrixXd>
            svd(J, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals = svd.singularValues();

        Eigen::MatrixXd S = J; // copying the dimensions of J, its content is not needed.
        S.setZero();

        for (int i = 0; i < sing_vals.size(); i++)
            S(i, i) = (sing_vals(i)) / (sing_vals(i) * sing_vals(i) + damping * damping);

        J_pinv = svd.matrixV() * S.transpose() * svd.matrixU().transpose();
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
        double li_norm, R_tr, ang;
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

    void _rand_q(ETS *ets, MapVectorX q)
    {
        Eigen::Map<Eigen::ArrayXd> qlim_l(ets->qlim_l, ets->n);
        Eigen::Map<Eigen::ArrayXd> q_range2(ets->q_range2, ets->n);

        q = VectorX::Random(ets->n);

        q = (q.array() + 1) * q_range2;
        q = q.array() + qlim_l;

        // return q;
    }

} /* extern "C" */