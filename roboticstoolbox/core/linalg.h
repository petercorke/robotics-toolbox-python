/**
 * \file linalg.h
 * \author Jesse Haviland
 *
 */
/* linalg.h */

#ifndef _LINALG_H_
#define _LINALG_H_

#include <Eigen/Dense>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

// #include <Eigen/Dense>
#define Matrix4dr Eigen::Matrix<double, 4, 4, Eigen::RowMajor>
#define MapMatrix4dr Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>

    void _inv(double *m, double *invOut);
    void _r2q(double *r, double *q);
    void _angle_axis(double *Te, double *Tep, double *e);
    void _eye4(double *data);
    void eye4(Matrix4dr &data);
    void _copy(double *A, double *B);
    void _mult4(double *A, double *B, double *C);
    double _trace(double *a, int n);
    void _mult(int n, int m, double *A, int p, int q, double *B, double *C);
    void _mult_T(int n, int m, int AT, double *A, int p, int q, int BT, double *B, double *C);
    void _cross(double *a, double *b, double *ret, int n);
    double _norm(double *a, int n);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif