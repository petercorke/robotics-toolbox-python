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

    void _ETS_IK(PyObject *ets, int n, double *q, double *Tep, double *ret);
    void _ETS_hessian(double *J, double *H);
    void _ETS_jacob0(ETS *ets, double *q, double *tool, MapMatrixJc &eJ);
    void _ETS_jacobe(ETS *ets, double *q, double *tool, MapMatrixJc &eJ);
    void _ETS_fkine(ETS *ets, double *q, double *base, double *tool, MapMatrix4dc &e_ret);
    void _ET_T(ET *et, double *ret, double eta);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif