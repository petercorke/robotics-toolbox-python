/**
 * \file ik.h
 * \author Jesse Haviland
 *
 */
/* ik.h */

#ifndef _IK_H_
#define _IK_H_

#include <Python.h>
#include "structs.h"
#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    void _IK_GN(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        MapVectorX we, int use_pinv, double pinv_damping);

    void _IK_NR(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        MapVectorX we, int use_pinv, double pinv_damping);

    void _IK_LM_Chan(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        double lambda, MapVectorX we);

    void _IK_LM_Wampler(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        double lambda, MapVectorX we);

    void _IK_LM_Sugihara(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        double lambda, MapVectorX we);

    void _pseudo_inverse(Eigen::Map<Eigen::MatrixXd> J, Eigen::Map<Eigen::MatrixXd> J_pinv, double damping);
    void _rand_q(ETS *ets, MapVectorX q);
    int _check_lim(ETS *ets, MapVectorX q);
    void _angle_axis(MapMatrix4dc Te, Matrix4dc Tep, MapVectorX e);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif