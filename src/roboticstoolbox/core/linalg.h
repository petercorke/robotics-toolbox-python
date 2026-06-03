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

#define PI_2 1.57079632679489661923132169163975144
#define PI 3.14159265358979323846264338327950288
#define PI_x2 6.283185307179586

#define Matrix3dc Eigen::Matrix3d

#define Matrix4dc Eigen::Matrix4d
#define Matrix4dr Eigen::Matrix<double, 4, 4, Eigen::RowMajor>

#define MapMatrix4dc Eigen::Map<Matrix4dc>
#define MapMatrix4dr Eigen::Map<Matrix4dr>

#define Matrix6dc Eigen::Matrix<double, 6, 6, Eigen::ColMajor>

#define MatrixJc Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
#define MatrixJr Eigen::Matrix<double, 6, Eigen::Dynamic, Eigen::RowMajor>
#define MapMatrixJc Eigen::Map<MatrixJc>
#define MapMatrixJr Eigen::Map<MatrixJr>

#define MatrixHc Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>
#define MatrixHr Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
#define MapMatrixHc Eigen::Map<MatrixHc>
#define MapMatrixHr Eigen::Map<MatrixHr>

#define Vector3 Eigen::Vector3d
#define MapVector3 Eigen::Map<Vector3>

#define VectorX Eigen::VectorXd
#define MapVectorX Eigen::Map<VectorX>

    void _inv(double *m, double *invOut);
    void _r2q(double *r, double *q);
    void _eye4(double *data);
    void eye4(Matrix4dc &data);
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