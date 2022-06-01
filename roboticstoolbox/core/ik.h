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

    void _IK_LM_Chan(
        ETS *ets, Matrix4dc Tep,
        MapVectorX q0, int ilimit, int slimit, double tol, int reject_jl,
        MapVectorX q, int *it, int *search, int *solution, double *E,
        double lambda, MapVectorX we
    );
    
    VectorX _rand_q(ETS *ets);
    int _check_lim(ETS *ets, MapVectorX q);
    void _angle_axis(MapMatrix4dc Te, Matrix4dc Tep, MapVectorX e);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif