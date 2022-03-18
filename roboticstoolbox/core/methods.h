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
    void _ETS_hessian(int n, double *J, double *H);
    void _ETS_jacob0(PyObject *ets, int n, double *q, double *tool, MapMatrixJc &eJ);
    void _ETS_jacobe(PyObject *ets, int n, double *q, double *tool, MapMatrixJc &eJ);
    void _ETS_fkine(PyObject *ets, double *q, double *base, double *tool, MapMatrix4dc &e_ret);
    void _ET_T(ET *et, double *ret, double eta);
    // void _ET_Alloc(ET *et);

    // #include "structs.h"

    // typedef struct ET ET;
    // typedef struct Link Link;

    // struct ET
    // {
    //     /**********************************************************
    //      *************** kinematic parameters *********************
    //      **********************************************************/
    //     int isjoint;
    //     int isflip;
    //     int jindex;
    //     int axis;
    //     double *T; /* link static transform */
    //     // Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> Tm;
    //     double *qlim; /* joint limits */
    //     void (*op)(double *data, double eta);
    // };

    // struct Link
    // {
    //     /**********************************************************
    //      *************** kinematic parameters *********************
    //      **********************************************************/
    //     int isjoint;
    //     int isflip;
    //     int jindex;
    //     int axis;
    //     int n_shapes;
    //     double *A;  /* link static transform */
    //     double *fk; /* link world transform */
    //     void (*op)(double *data, double eta);
    //     Link *parent;
    //     double **shape_base; /* link visual and collision geometries */
    //     double **shape_wT;   /* link visual and collision geometries */
    //     double **shape_sT;   /* link visual and collision geometries */
    //     double **shape_sq;   /* link visual and collision geometries */
    // };

    // int _check_array_type(PyObject *toCheck);
    // void _ETS_IK(PyObject *ets, int n, double *q, double *Tep, double *ret);
    // void _ETS_hessian(int n, double *J, double *H);
    // void _ETS_jacob0(PyObject *ets, int n, double *q, double *tool, double *J);
    // void _ETS_jacobe(PyObject *ets, int n, double *q, double *tool, double *J);
    // void _ETS_fkine(PyObject *ets, double *q, double *base, double *tool, double *ret);
    // void _ET_T(ET *et, double *ret, double eta);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif