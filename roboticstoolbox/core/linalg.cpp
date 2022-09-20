/**
 * \file linalg.cpp
 * \author Jesse Haviland
 *
 *
 */
/* linalg.cpp */

// #define EIGEN_USE_BLAS

#include "linalg.h"

#include <Python.h>
#include <math.h>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/QR>

extern "C"
{
    // --------------------------------------------------------------------- //
    // SE3 specific
    // --------------------------------------------------------------------- //

    void _inv(double *m, double *inv)
    {
        inv[0] = m[0];
        inv[4] = m[1];
        inv[8] = m[2];

        inv[1] = m[4];
        inv[5] = m[5];
        inv[9] = m[6];

        inv[2] = m[8];
        inv[6] = m[9];
        inv[10] = m[10];

        inv[12] = -(inv[0] * m[12] + inv[4] * m[13] + inv[8] * m[14]);
        inv[13] = -(inv[1] * m[12] + inv[5] * m[13] + inv[9] * m[14]);
        inv[14] = -(inv[2] * m[12] + inv[6] * m[13] + inv[10] * m[14]);

        inv[3] = 0;
        inv[7] = 0;
        inv[11] = 0;
        inv[15] = 1;

        // inv[0] = m[0];
        // inv[1] = m[4];
        // inv[2] = m[8];

        // inv[4] = m[1];
        // inv[5] = m[5];
        // inv[6] = m[9];

        // inv[8] = m[2];
        // inv[9] = m[6];
        // inv[10] = m[10];

        // inv[3] = -(inv[0] * m[3] + inv[1] * m[7] + inv[2] * m[11]);
        // inv[7] = -(inv[4] * m[3] + inv[5] * m[7] + inv[6] * m[11]);
        // inv[11] = -(inv[8] * m[3] + inv[9] * m[7] + inv[10] * m[11]);

        // inv[12] = 0;
        // inv[13] = 0;
        // inv[14] = 0;
        // inv[15] = 1;
    }

    void _r2q(double *r, double *q)
    {
        double t12p, t13p, t23p;
        double t12m, t13m, t23m;
        double d1, d2, d3, d4;

        t12p = pow((r[0 * 4 + 1] + r[1 * 4 + 0]), 2);
        t13p = pow((r[0 * 4 + 2] + r[2 * 4 + 0]), 2);
        t23p = pow((r[1 * 4 + 2] + r[2 * 4 + 1]), 2);

        t12m = pow((r[0 * 4 + 1] - r[1 * 4 + 0]), 2);
        t13m = pow((r[0 * 4 + 2] - r[2 * 4 + 0]), 2);
        t23m = pow((r[1 * 4 + 2] - r[2 * 4 + 1]), 2);

        d1 = pow((r[0 * 4 + 0] + r[1 * 4 + 1] + r[2 * 4 + 2] + 1), 2);
        d2 = pow((r[0 * 4 + 0] - r[1 * 4 + 1] - r[2 * 4 + 2] + 1), 2);
        d3 = pow((-r[0 * 4 + 0] + r[1 * 4 + 1] - r[2 * 4 + 2] + 1), 2);
        d4 = pow((-r[0 * 4 + 0] - r[1 * 4 + 1] + r[2 * 4 + 2] + 1), 2);

        q[3] = sqrt(d1 + t23m + t13m + t12m) / 4.0;
        q[0] = sqrt(t23m + d2 + t12p + t13p) / 4.0;
        q[1] = sqrt(t13m + t12p + d3 + t23p) / 4.0;
        q[2] = sqrt(t12m + t13p + t23p + d4) / 4.0;

        // transfer sign from rotation element differences
        if (r[2 * 4 + 1] < r[1 * 4 + 2])
            q[0] = -q[0];
        if (r[0 * 4 + 2] < r[2 * 4 + 0])
            q[1] = -q[1];
        if (r[1 * 4 + 0] < r[0 * 4 + 1])
            q[2] = -q[2];
    }

    // --------------------------------------------------------------------- //
    // 4x4 matrix linalg
    // --------------------------------------------------------------------- //

    void _eye4(double *data)
    {
        data[0] = 1;
        data[1] = 0;
        data[2] = 0;
        data[3] = 0;
        data[4] = 0;
        data[5] = 1;
        data[6] = 0;
        data[7] = 0;
        data[8] = 0;
        data[9] = 0;
        data[10] = 1;
        data[11] = 0;
        data[12] = 0;
        data[13] = 0;
        data[14] = 0;
        data[15] = 1;
    }

    void eye4(Matrix4dc &data)
    {
        data(0, 0) = 1;
        data(0, 1) = 0;
        data(0, 2) = 0;
        data(0, 3) = 0;
        data(1, 0) = 0;
        data(1, 1) = 1;
        data(1, 2) = 0;
        data(1, 3) = 0;
        data(2, 0) = 0;
        data(2, 1) = 0;
        data(2, 2) = 1;
        data(2, 3) = 0;
        data(3, 0) = 0;
        data(3, 1) = 0;
        data(3, 2) = 0;
        data(3, 3) = 1;
    }

    void _copy(double *A, double *B)
    {
        // copy A into B
        B[0] = A[0];
        B[1] = A[1];
        B[2] = A[2];
        B[3] = A[3];
        B[4] = A[4];
        B[5] = A[5];
        B[6] = A[6];
        B[7] = A[7];
        B[8] = A[8];
        B[9] = A[9];
        B[10] = A[10];
        B[11] = A[11];
        B[12] = A[12];
        B[13] = A[13];
        B[14] = A[14];
        B[15] = A[15];
    }

    void _mult4(double *A, double *B, double *C)
    {
        // mult4(A, B, C);

        const int N = 4;
        int i, j;

        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {
                C[i * N + j] = A[i * N + 0] * B[0 + j] + A[i * N + 1] * B[4 + j] + A[i * N + 2] * B[8 + j] + A[i * N + 3] * B[12 + j];
            }
        }

        // C[0] = A[0] * B[0] + A[1] * B[4 + 0] + A[2] * B[8 + 0] + A[3] * B[12 + 0];
        // C[1] = A[0] * B[1] + A[1] * B[4 + 1] + A[2] * B[8 + 1] + A[3] * B[12 + 1];
        // C[2] = A[0] * B[2] + A[1] * B[4 + 2] + A[2] * B[8 + 2] + A[3] * B[12 + 2];
        // C[3] = A[0] * B[3] + A[1] * B[4 + 3] + A[2] * B[8 + 3] + A[3] * B[12 + 3];
        // C[4] = A[4 + 0] * B[0] + A[4 + 1] * B[4 + 0] + A[4 + 2] * B[8 + 0] + A[4 + 3] * B[12 + 0];
        // C[4 + 1] = A[4 + 0] * B[1] + A[4 + 1] * B[4 + 1] + A[4 + 2] * B[8 + 1] + A[4 + 3] * B[12 + 1];
        // C[4 + 2] = A[4 + 0] * B[2] + A[4 + 1] * B[4 + 2] + A[4 + 2] * B[8 + 2] + A[4 + 3] * B[12 + 2];
        // C[4 + 3] = A[4 + 0] * B[3] + A[4 + 1] * B[4 + 3] + A[4 + 2] * B[8 + 3] + A[4 + 3] * B[12 + 3];
        // C[8 + 0] = A[8 + 0] * B[0] + A[8 + 1] * B[4 + 0] + A[8 + 2] * B[8 + 0] + A[8 + 3] * B[12 + 0];
        // C[8 + 1] = A[8 + 0] * B[1] + A[8 + 1] * B[4 + 1] + A[8 + 2] * B[8 + 1] + A[8 + 3] * B[12 + 1];
        // C[8 + 2] = A[8 + 0] * B[2] + A[8 + 1] * B[4 + 2] + A[8 + 2] * B[8 + 2] + A[8 + 3] * B[12 + 2];
        // C[8 + 3] = A[8 + 0] * B[3] + A[8 + 1] * B[4 + 3] + A[8 + 2] * B[8 + 3] + A[8 + 3] * B[12 + 3];
        // C[12 + 0] = A[12 + 0] * B[0] + A[12 + 1] * B[4 + 0] + A[12 + 2] * B[8 + 0] + A[12 + 3] * B[12 + 0];
        // C[12 + 1] = A[12 + 0] * B[1] + A[12 + 1] * B[4 + 1] + A[12 + 2] * B[8 + 1] + A[12 + 3] * B[12 + 1];
        // C[12 + 2] = A[12 + 0] * B[2] + A[12 + 1] * B[4 + 2] + A[12 + 2] * B[8 + 2] + A[12 + 3] * B[12 + 2];
        // C[12 + 3] = A[12 + 0] * B[3] + A[12 + 1] * B[4 + 3] + A[12 + 2] * B[8 + 3] + A[12 + 3] * B[12 + 3];
    }

    // --------------------------------------------------------------------- //
    // General matrix linalg
    // --------------------------------------------------------------------- //

    double _trace(double *a, int n)
    {
        // Assumes square nxn matrix
        int i;
        double sum = 0;

        for (i = 0; i < n; i++)
        {
            sum += a[i * n + i];
        }

        return sum;
    }

    void _mult(int n, int m, double *A, int p, int q, double *B, double *C)
    {
        int i, j, k;
        double num;

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < q; j++)
            {
                num = 0;
                for (k = 0; k < p; k++)
                {
                    num += A[i * m + k] * B[k * q + j];
                }
                C[i * q + j] = num;
            }
        }
    }

    void _mult_T(int n, int m, int AT, double *A, int p, int q, int BT, double *B, double *C)
    {
        int i, j, k, temp;
        double num, a, b;

        if (AT)
        {
            temp = n;
            n = m;
            m = temp;
        }

        if (BT)
        {
            temp = p;
            p = q;
            q = temp;
        }

        for (i = 0; i < n; i++)
        {
            for (j = 0; j < q; j++)
            {
                num = 0;
                for (k = 0; k < p; k++)
                {
                    if (AT)
                    {
                        a = A[k * n + i];
                    }
                    else
                    {
                        a = A[i * m + k];
                    }

                    if (BT)
                    {
                        b = B[j * p + k];
                    }
                    else
                    {
                        b = B[k * q + j];
                    }

                    num += a * b;
                }
                C[i * q + j] = num;
            }
        }
    }

    // --------------------------------------------------------------------- //
    // Vector linalg
    // --------------------------------------------------------------------- //

    void _cross(double *a, double *b, double *ret, int n)
    {
        ret[0] = a[1 * n] * b[2 * n] - a[2 * n] * b[1 * n];
        ret[1 * n] = a[2 * n] * b[0] - a[0] * b[2 * n];
        ret[2 * n] = a[0] * b[1 * n] - a[1 * n] * b[0];
        // ret[0] = b[0 * n];
        // ret[1 * n] = b[1 * n];
        // ret[2 * n] = b[2 * n];
    }

    double _norm(double *a, int n)
    {
        int i;
        double sum = 0;

        for (i = 0; i < n; i++)
        {
            sum += pow(a[i], 2);
        }

        return sqrt(sum);
    }

} /* extern "C" */