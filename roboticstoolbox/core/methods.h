/**
 * \file methods.h
 * \author Jesse Haviland
 *
 */
/* methods.h */

#ifndef _METHODS_H_
#define _METHODS_H_

#include <Python.h>
#include "structs.h"
#include "linalg.h"

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    int _check_lim(ETS *ets, MapVectorX q);
    void _angle_axis(MapMatrix4dc Te, Matrix4dc Tep, MapVectorX e);
    void _ETS_hessian(int n, MapMatrixJc &J, MapMatrixHr &H);
    void _ETS_jacob0(ETS *ets, double *q, double *tool, MapMatrixJc &eJ);
    void _ETS_jacobe(ETS *ets, double *q, double *tool, MapMatrixJc &eJ);
    void _ETS_fkine(ETS *ets, double *q, double *base, double *tool, MapMatrix4dc &e_ret);
    void _ET_T(ET *et, double *ret, double eta);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif