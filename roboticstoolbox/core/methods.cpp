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
    void _ETS_IK(PyObject *ets, int n, double *q, double *Tep, double *ret)
    {
        // double E;
        double *Te = (double *)PyMem_RawCalloc(16, sizeof(double));
        double *e = (double *)PyMem_RawCalloc(6, sizeof(double));

        double *a = (double *)PyMem_RawCalloc(6, sizeof(double));
        // a[0] = 1.0;
        // a[1] = 4.0;
        // a[2] = 2.0;
        // a[3] = 5.0;
        // a[4] = 3.0;
        // a[5] = 6.0;
        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        a[3] = 4.0;
        a[4] = 5.0;
        a[5] = 6.0;
        double *b = (double *)PyMem_RawCalloc(12, sizeof(double));
        b[0] = 11.0;
        b[1] = 15.0;
        b[2] = 19.0;
        b[3] = 12.0;
        b[4] = 16.0;
        b[5] = 20.0;
        b[6] = 13.0;
        b[7] = 17.0;
        b[8] = 21.0;
        b[9] = 14.0;
        b[10] = 18.0;
        b[11] = 22.0;
        // b[0] = 11.0;
        // b[1] = 12.0;
        // b[2] = 13.0;
        // b[3] = 14.0;
        // b[4] = 15.0;
        // b[5] = 16.0;
        // b[6] = 17.0;
        // b[7] = 18.0;
        // b[8] = 19.0;
        // b[9] = 20.0;
        // b[10] = 21.0;
        // b[11] = 22.0;

        // double *U = (double *)PyMem_RawCalloc(16, sizeof(double));
        // double *invU = (double *)PyMem_RawCalloc(16, sizeof(double));
        // double *temp = (double *)PyMem_RawCalloc(16, sizeof(double));
        // double *ret = (double *)PyMem_RawCalloc(16, sizeof(double));
        // Py_ssize_t m;
        int arrived = 0, iter = 0;

        while (arrived == 0 && iter < 500)
        {
            // Current pose Te
            // _ETS_fkine(ets, q, (double *)NULL, NULL, Te);

            // Angle axis error e
            _angle_axis(Te, Tep, e);

            // Squared error E
            // E = 0.5 * e @ We @ e
            // E = 0.5 * (e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3] + e[4] * e[4] + e[5] * e[5]);
        }

        // _ETS_fkine(ets, q, (double *)NULL, NULL, Te);

        // for (int i = 0; i < 2; i++)
        // {
        //     for (int j = 0; j < 4; j++)
        //     {
        //         ret[i * 4 + j] = 0.0;
        //     }
        // }

        // _mult_T(2, 3, 0, a, 3, 4, 0, b, ret);
        // _mult_T(3, 2, 1, a, 3, 4, 0, b, ret);
        // _mult_T(3, 2, 1, a, 4, 3, 1, b, ret);
        // _mult_T(2, 3, 0, a, 4, 3, 1, b, ret);

        // int j = 0;
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