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

    void _ETS_hessian(int n, double *J, double *H)
    {
        int a, b;
        int n2 = 2 * n, n3 = 3 * n, n4 = 4 * n, n5 = 5 * n;

        for (int j = 0; j < n; j++)
        {
            a = j * 6 * n;
            for (int i = j; i < n; i++)
            {
                b = i * 6 * n;
                _cross(J + j + n3, J + i, H + a + i, n);
                _cross(J + j + n3, J + i + n3, H + a + i + n3, n);

                if (i != j)
                {
                    H[b + j] = H[a + i];
                    H[b + j + n] = H[a + i + n];
                    H[b + j + n2] = H[a + i + n2];
                    H[b + j + n3] = 0;
                    H[b + j + n4] = 0;
                    H[b + j + n5] = 0;
                }
            }
        }
    }

    void _ETS_jacob0(PyObject *ets, int n, double *q, double *tool, double *J)
    {
        ET *et;
        Py_ssize_t m;

        MapMatrixJ eJ(J, 6, n);
        double T[16];
        MapMatrix4dr eT(T);
        Matrix4dr U;
        Matrix4dr invU;
        Matrix4dr temp;
        Matrix4dr ret;

        int j = 0;

        // eye4(U);
        U = Eigen::Matrix4d::Identity();

        // Get the forward  kinematics into T
        _ETS_fkine(ets, q, (double *)NULL, tool, eT);

        PyObject *iter_et = PyObject_GetIter(ets);

        m = PyList_GET_SIZE(ets);
        for (int i = 0; i < m; i++)
        {
            if (!(et = (ET *)PyCapsule_GetPointer(PyIter_Next(iter_et), "ET")))
                return;

            if (et->isjoint)
            {
                _ET_T(et, &ret(0), q[et->jindex]);
                temp = U * ret;
                U = temp;

                if (i == m - 1 && tool != NULL)
                {
                    MapMatrix4dr e_tool(tool);
                    temp = U * e_tool;
                    U = temp;
                }

                _inv(&U(0), &invU(0));
                temp = invU * eT;

                if (et->axis == 0)
                {
                    eJ(0, j) = U(0, 2) * temp(1, 3) - U(0, 1) * temp(2, 3);
                    eJ(1, j) = U(1, 2) * temp(1, 3) - U(1, 1) * temp(2, 3);
                    eJ(2, j) = U(2, 2) * temp(1, 3) - U(2, 1) * temp(2, 3);
                    eJ(3, j) = U(0, 0);
                    eJ(4, j) = U(1, 0);
                    eJ(5, j) = U(2, 0);
                }
                else if (et->axis == 1)
                {
                    eJ(0, j) = U(0, 0) * temp(2, 3) - U(0, 2) * temp(0, 3);
                    eJ(1, j) = U(1, 0) * temp(2, 3) - U(1, 2) * temp(0, 3);
                    eJ(2, j) = U(2, 0) * temp(2, 3) - U(2, 2) * temp(0, 3);
                    eJ(3, j) = U(0, 1);
                    eJ(4, j) = U(1, 1);
                    eJ(5, j) = U(2, 1);
                }
                else if (et->axis == 2)
                {
                    eJ(0, j) = U(0, 1) * temp(0, 3) - U(0, 0) * temp(1, 3);
                    eJ(1, j) = U(1, 1) * temp(0, 3) - U(1, 0) * temp(1, 3);
                    eJ(2, j) = U(2, 1) * temp(0, 3) - U(2, 0) * temp(1, 3);
                    eJ(3, j) = U(0, 2);
                    eJ(4, j) = U(1, 2);
                    eJ(5, j) = U(2, 2);
                }
                else if (et->axis == 3)
                {
                    eJ(0, j) = U(0, 0);
                    eJ(1, j) = U(1, 0);
                    eJ(2, j) = U(2, 0);
                    eJ(3, j) = 0.0;
                    eJ(4, j) = 0.0;
                    eJ(5, j) = 0.0;
                }
                else if (et->axis == 4)
                {
                    eJ(0, j) = U(0, 1);
                    eJ(1, j) = U(1, 1);
                    eJ(2, j) = U(2, 1);
                    eJ(3, j) = 0.0;
                    eJ(4, j) = 0.0;
                    eJ(5, j) = 0.0;
                }
                else if (et->axis == 5)
                {
                    eJ(0, j) = U(0, 2);
                    eJ(1, j) = U(1, 2);
                    eJ(2, j) = U(2, 2);
                    eJ(3, j) = 0.0;
                    eJ(4, j) = 0.0;
                    eJ(5, j) = 0.0;
                }
                j++;
            }
            else
            {
                _ET_T(et, &ret(0), q[et->jindex]);
                temp = U * ret;
                U = temp;
            }
        }

        Py_DECREF(iter_et);
    }

    void _ETS_jacobe(PyObject *ets, int n, double *q, double *tool, double *J)
    {
        ET *et;
        double *T = (double *)PyMem_RawCalloc(16, sizeof(double));
        double *U = (double *)PyMem_RawCalloc(16, sizeof(double));
        double *temp = (double *)PyMem_RawCalloc(16, sizeof(double));
        double *ret = (double *)PyMem_RawCalloc(16, sizeof(double));
        Py_ssize_t m;

        int j = n - 1;

        _eye4(U);

        // Get the forward  kinematics into T
        // _ETS_fkine(ets, q, (double *)NULL, tool, T);

        PyList_Reverse(ets);
        PyObject *iter_et = PyObject_GetIter(ets);

        if (tool != NULL)
        {
            _mult4(tool, U, temp);
            _copy(temp, U);
        }

        m = PyList_GET_SIZE(ets);
        for (int i = 0; i < m; i++)
        {
            if (!(et = (ET *)PyCapsule_GetPointer(PyIter_Next(iter_et), "ET")))
                return;

            if (et->isjoint)
            {
                if (et->axis == 0)
                {
                    J[0 * n + j] = U[2 * 4 + 0] * U[1 * 4 + 3] - U[1 * 4 + 0] * U[2 * 4 + 3];
                    J[1 * n + j] = U[2 * 4 + 1] * U[1 * 4 + 3] - U[1 * 4 + 1] * U[2 * 4 + 3];
                    J[2 * n + j] = U[2 * 4 + 2] * U[1 * 4 + 3] - U[1 * 4 + 2] * U[2 * 4 + 3];
                    J[3 * n + j] = U[0 * 4 + 0];
                    J[4 * n + j] = U[0 * 4 + 1];
                    J[5 * n + j] = U[0 * 4 + 2];
                }
                else if (et->axis == 1)
                {
                    J[0 * n + j] = U[0 * 4 + 0] * U[2 * 4 + 3] - U[2 * 4 + 0] * U[0 * 4 + 3];
                    J[1 * n + j] = U[0 * 4 + 1] * U[2 * 4 + 3] - U[2 * 4 + 1] * U[0 * 4 + 3];
                    J[2 * n + j] = U[0 * 4 + 2] * U[2 * 4 + 3] - U[2 * 4 + 2] * U[0 * 4 + 3];
                    J[3 * n + j] = U[1 * 4 + 0];
                    J[4 * n + j] = U[1 * 4 + 1];
                    J[5 * n + j] = U[1 * 4 + 2];
                }
                else if (et->axis == 2)
                {
                    J[0 * n + j] = U[1 * 4 + 0] * U[0 * 4 + 3] - U[0 * 4 + 0] * U[1 * 4 + 3];
                    J[1 * n + j] = U[1 * 4 + 1] * U[0 * 4 + 3] - U[0 * 4 + 1] * U[1 * 4 + 3];
                    J[2 * n + j] = U[1 * 4 + 2] * U[0 * 4 + 3] - U[0 * 4 + 2] * U[1 * 4 + 3];
                    J[3 * n + j] = U[2 * 4 + 0];
                    J[4 * n + j] = U[2 * 4 + 1];
                    J[5 * n + j] = U[2 * 4 + 2];
                }
                else if (et->axis == 3)
                {
                    J[0 * n + j] = U[0 * 4 + 0];
                    J[1 * n + j] = U[0 * 4 + 1];
                    J[2 * n + j] = U[0 * 4 + 2];
                    J[3 * n + j] = 0.0;
                    J[4 * n + j] = 0.0;
                    J[5 * n + j] = 0.0;
                }
                else if (et->axis == 4)
                {
                    J[0 * n + j] = U[1 * 4 + 0];
                    J[1 * n + j] = U[1 * 4 + 1];
                    J[2 * n + j] = U[1 * 4 + 2];
                    J[3 * n + j] = 0.0;
                    J[4 * n + j] = 0.0;
                    J[5 * n + j] = 0.0;
                }
                else if (et->axis == 5)
                {
                    J[0 * n + j] = U[2 * 4 + 0];
                    J[1 * n + j] = U[2 * 4 + 1];
                    J[2 * n + j] = U[2 * 4 + 2];
                    J[3 * n + j] = 0.0;
                    J[4 * n + j] = 0.0;
                    J[5 * n + j] = 0.0;
                }

                _ET_T(et, ret, q[et->jindex]);
                _mult4(ret, U, temp);
                _copy(temp, U);
                j--;
            }
            else
            {
                _ET_T(et, ret, q[et->jindex]);
                _mult4(ret, U, temp);
                _copy(temp, U);
            }
        }

        PyList_Reverse(ets);

        Py_DECREF(iter_et);

        free(T);
        free(U);
        free(temp);
        free(ret);
    }

    void _ETS_fkine(PyObject *ets, double *q, double *base, double *tool, MapMatrix4dr &e_ret)
    {
        // double *temp, *current;
        ET *et;
        Py_ssize_t m;
        // MapMatrix4dr e_ret(ret);

        // temp = (double *)PyMem_RawCalloc(16, sizeof(double));
        // current = (double *)PyMem_RawCalloc(16, sizeof(double));
        Matrix4dr temp;
        Matrix4dr current;

        // MapMatrix4dr e_temp(temp);
        // MapMatrix4dr e_current(current);

        PyObject *iter_et = PyObject_GetIter(ets);

        // std::cout << std::endl;
        // std::cout << &current << std::endl;

        if (base != NULL)
        {
            MapMatrix4dr e_base(base);
            current = e_base;
            // _copy(base, current);
        }
        else
        {
            eye4(current);
        }

        m = PyList_GET_SIZE(ets);
        for (int i = 0; i < m; i++)
        {
            if (!(et = (ET *)PyCapsule_GetPointer(PyIter_Next(iter_et), "ET")))
                return;

            _ET_T(et, &e_ret(0, 0), q[et->jindex]);
            temp = current * e_ret;
            current = temp;
            // _mult4(current, ret, temp);
            // _copy(temp, current);
        }

        if (tool != NULL)
        {
            MapMatrix4dr e_tool(tool);
            e_ret = current * e_tool;
            // _mult4(current, tool, ret);
        }
        else
        {
            e_ret = current;
            // _copy(current, ret);
        }

        Py_DECREF(iter_et);

        // free(temp);
        // free(current);
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

    // void _ET_Alloc(ET *et)
    // {
    //     new (&et->Tm) Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(et->T);
    // }

} /* extern "C" */