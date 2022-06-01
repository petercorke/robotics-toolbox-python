/* methods.cpp */

#include "linalg.h"
#include "methods.h"
#include "structs.h"

#include <Python.h>
#include <math.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>

extern "C"
{

    void _IK_LM_Chan(ETS *ets, Matrix4dc Tep, MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl, MapVectorX q, int *it, int *search, int *solution, double *E)
    {
        int iter = 0;
        double lambda = 1.0;

        double *np_Te = (double *)PyMem_RawCalloc(16, sizeof(double));
        MapMatrix4dc Te(np_Te);

        double *np_J = (double *)PyMem_RawCalloc(6 * ets->n, sizeof(double));
        MapMatrixJc J(np_J, 6, ets->n);

        double *np_e = (double *)PyMem_RawCalloc(6, sizeof(double));
        MapVectorX e(np_e, 6);

        Matrix6dc We = Matrix6dc::Identity();

        Eigen::MatrixXd Wn(ets->n, ets->n);
        Eigen::MatrixXd EyeN = Eigen::MatrixXd::Identity(ets->n, ets->n);

        VectorX g(ets->n);

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
                // std::cout << "Search: " << *search << " Iter: " << *it << " Solution: " << *solution << " E: " << *E << " e: " << e.transpose() << '\n';
                break;
            }

            // std::cout << "Search: " << *search << " Iter: " << *it << " Solution: " << *solution << " E: " << *E << " e: " << e.transpose() << '\n';
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
        int i, j, k;
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

    void _ETS_hessian(int n, MapMatrixJc &J, MapMatrixHr &H)
    {
        for (int j = 0; j < n; j++)
        {
            for (int i = j; i < n; i++)
            {
                H.block<3, 1>(j * 6, i) = J.block<3, 1>(3, j).cross(J.block<3, 1>(0, i));
                H.block<3, 1>(j * 6 + 3, i) = J.block<3, 1>(3, j).cross(J.block<3, 1>(3, i));

                if (i != j)
                {
                    H.block<3, 1>(i * 6, j) = H.block<3, 1>(j * 6, i);
                    H.block<3, 1>(i * 6 + 3, j) = Eigen::Vector3d::Zero();
                }
            }
        }
    }

    void _ETS_jacob0(ETS *ets, double *q, double *tool, MapMatrixJc &eJ)
    {
        // ET *et;
        // double T[16];
        // MapMatrix4dc eT(T);
        // Matrix4dc U;
        // Matrix4dc invU;
        // Matrix4dc temp;
        // Matrix4dc ret;

        // int j = 0;

        // U = Eigen::Matrix4d::Identity();

        // // Get the forward  kinematics into T
        // _ETS_fkine(ets, q, (double *)NULL, tool, eT);

        // for (int i = 0; i < ets->m; i++)
        // {
        //     et = ets->ets[i];

        //     if (et->isjoint)
        //     {
        //         _ET_T(et, &ret(0), q[et->jindex]);
        //         temp = U * ret;
        //         U = temp;

        //         if (i == ets->m - 1 && tool != NULL)
        //         {
        //             MapMatrix4dc e_tool(tool);
        //             temp = U * e_tool;
        //             U = temp;
        //         }

        //         _inv(&U(0), &invU(0));
        //         temp = invU * eT;

        //         if (et->axis == 0)
        //         {
        //             eJ(Eigen::seq(0, 2), j) = U(Eigen::seq(0, 2), 2) * temp(1, 3) - U(Eigen::seq(0, 2), 1) * temp(2, 3);

        //             eJ(Eigen::seq(3, 5), j) = U(Eigen::seq(0, 2), 0);
        //         }
        //         else if (et->axis == 1)
        //         {
        //             eJ(Eigen::seq(0, 2), j) = U(Eigen::seq(0, 2), 0) * temp(2, 3) - U(Eigen::seq(0, 2), 2) * temp(0, 3);
        //             eJ(Eigen::seq(3, 5), j) = U(Eigen::seq(0, 2), 1);
        //         }
        //         else if (et->axis == 2)
        //         {
        //             eJ(Eigen::seq(0, 2), j) = U(Eigen::seq(0, 2), 1) * temp(0, 3) - U(Eigen::seq(0, 2), 0) * temp(1, 3);
        //             eJ(Eigen::seq(3, 5), j) = U(Eigen::seq(0, 2), 2);
        //         }
        //         else if (et->axis == 3)
        //         {
        //             eJ(Eigen::seq(0, 2), j) = U(Eigen::seq(0, 2), 0);
        //             eJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
        //         }
        //         else if (et->axis == 4)
        //         {
        //             eJ(Eigen::seq(0, 2), j) = U(Eigen::seq(0, 2), 1);
        //             eJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
        //         }
        //         else if (et->axis == 5)
        //         {
        //             eJ(Eigen::seq(0, 2), j) = U(Eigen::seq(0, 2), 2);
        //             eJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
        //         }
        //         j++;
        //     }
        //     else
        //     {
        //         _ET_T(et, &ret(0), q[et->jindex]);
        //         temp = U * ret;
        //         U = temp;
        //     }
        // }

        ET *et;
        Eigen::Matrix<double, 6, Eigen::Dynamic> tJ(6, ets->n);
        double T[16];
        MapMatrix4dc eT(T);
        Matrix4dc U = Eigen::Matrix4d::Identity();
        Matrix4dc invU;
        Matrix4dc temp;
        Matrix4dc ret;
        int j = ets->n - 1;

        if (tool != NULL)
        {
            Matrix4dc e_tool(tool);
            temp = e_tool * U;
            U = temp;
        }

        for (int i = ets->m - 1; i >= 0; i--)
        {
            et = ets->ets[i];

            if (et->isjoint)
            {
                if (et->axis == 0)
                {
                    tJ(Eigen::seq(0, 2), j) = U(2, Eigen::seq(0, 2)) * U(1, 3) - U(1, Eigen::seq(0, 2)) * U(2, 3);
                    tJ(Eigen::seq(3, 5), j) = U(0, Eigen::seq(0, 2));
                }
                else if (et->axis == 1)
                {
                    tJ(Eigen::seq(0, 2), j) = U(0, Eigen::seq(0, 2)) * U(2, 3) - U(2, Eigen::seq(0, 2)) * U(0, 3);
                    tJ(Eigen::seq(3, 5), j) = U(1, Eigen::seq(0, 2));
                }
                else if (et->axis == 2)
                {
                    tJ(Eigen::seq(0, 2), j) = U(1, Eigen::seq(0, 2)) * U(0, 3) - U(0, Eigen::seq(0, 2)) * U(1, 3);
                    tJ(Eigen::seq(3, 5), j) = U(2, Eigen::seq(0, 2));
                }
                else if (et->axis == 3)
                {
                    tJ(Eigen::seq(0, 2), j) = U(0, Eigen::seq(0, 2));
                    tJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (et->axis == 4)
                {
                    tJ(Eigen::seq(0, 2), j) = U(1, Eigen::seq(0, 2));
                    tJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (et->axis == 5)
                {
                    tJ(Eigen::seq(0, 2), j) = U(2, Eigen::seq(0, 2));
                    tJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }

                _ET_T(et, &ret(0), q[et->jindex]);
                temp = ret * U;
                U = temp;
                j--;
            }
            else
            {
                _ET_T(et, &ret(0), q[et->jindex]);
                temp = ret * U;
                U = temp;
            }
        }

        Eigen::Matrix<double, 6, 6> ev;
        ev.topLeftCorner<3, 3>() = U.topLeftCorner<3, 3>();
        ev.topRightCorner<3, 3>() = Eigen::Matrix3d::Zero();
        ev.bottomLeftCorner<3, 3>() = Eigen::Matrix3d::Zero();
        ev.bottomRightCorner<3, 3>() = U.topLeftCorner<3, 3>();
        eJ = ev * tJ;
    }

    void _ETS_jacobe(ETS *ets, double *q, double *tool, MapMatrixJc &eJ)
    {
        ET *et;
        double T[16];
        MapMatrix4dc eT(T);
        Matrix4dc U = Eigen::Matrix4d::Identity();
        Matrix4dc invU;
        Matrix4dc temp;
        Matrix4dc ret;
        int j = ets->n - 1;

        if (tool != NULL)
        {
            Matrix4dc e_tool(tool);
            temp = e_tool * U;
            U = temp;
        }

        for (int i = ets->m - 1; i >= 0; i--)
        {
            et = ets->ets[i];

            if (et->isjoint)
            {
                if (et->axis == 0)
                {
                    eJ(Eigen::seq(0, 2), j) = U(2, Eigen::seq(0, 2)) * U(1, 3) - U(1, Eigen::seq(0, 2)) * U(2, 3);
                    eJ(Eigen::seq(3, 5), j) = U(0, Eigen::seq(0, 2));
                }
                else if (et->axis == 1)
                {
                    eJ(Eigen::seq(0, 2), j) = U(0, Eigen::seq(0, 2)) * U(2, 3) - U(2, Eigen::seq(0, 2)) * U(0, 3);
                    eJ(Eigen::seq(3, 5), j) = U(1, Eigen::seq(0, 2));
                }
                else if (et->axis == 2)
                {
                    eJ(Eigen::seq(0, 2), j) = U(1, Eigen::seq(0, 2)) * U(0, 3) - U(0, Eigen::seq(0, 2)) * U(1, 3);
                    eJ(Eigen::seq(3, 5), j) = U(2, Eigen::seq(0, 2));
                }
                else if (et->axis == 3)
                {
                    eJ(Eigen::seq(0, 2), j) = U(0, Eigen::seq(0, 2));
                    eJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (et->axis == 4)
                {
                    eJ(Eigen::seq(0, 2), j) = U(1, Eigen::seq(0, 2));
                    eJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }
                else if (et->axis == 5)
                {
                    eJ(Eigen::seq(0, 2), j) = U(2, Eigen::seq(0, 2));
                    eJ(Eigen::seq(3, 5), j) = Eigen::Vector3d::Zero();
                }

                _ET_T(et, &ret(0), q[et->jindex]);
                temp = ret * U;
                U = temp;
                j--;
            }
            else
            {
                _ET_T(et, &ret(0), q[et->jindex]);
                temp = ret * U;
                U = temp;
            }
        }
    }

    void _ETS_fkine(ETS *ets, double *q, double *base, double *tool, MapMatrix4dc &e_ret)
    {
        ET *et;
        Matrix4dc temp;
        Matrix4dc current;

        if (base != NULL)
        {
            MapMatrix4dc e_base(base);
            current = e_base;
        }
        else
        {
            current = Eigen::Matrix4d::Identity();
        }

        for (int i = 0; i < ets->m; i++)
        {
            et = ets->ets[i];

            _ET_T(et, &e_ret(0), q[et->jindex]);
            temp = current * e_ret;
            current = temp;
        }

        if (tool != NULL)
        {
            MapMatrix4dc e_tool(tool);
            e_ret = current * e_tool;
        }
        else
        {
            e_ret = current;
        }
    }

    void _ET_T(ET *et, double *ret, double eta)
    {
        // Check if static and return static transform
        if (!et->isjoint)
        {
            _copy(et->T, ret);
            return;
        }

        if (et->isflip)
        {
            eta = -eta;
        }

        // Calculate ET trasform based on eta
        et->op(ret, eta);
    }

} /* extern "C" */