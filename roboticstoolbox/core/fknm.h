/**
 * \file fknm.h
 * \author Jesse Haviland
 *
 */

#ifndef _FKNM_H_
#define _FKNM_H_

#include <Python.h>
#include <numpy/arrayobject.h>

#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */

    // forward defines
    static PyObject *Angle_Axis(PyObject *self, PyObject *args);

    static PyObject *IK_GN_c(PyObject *self, PyObject *args);
    static PyObject *IK_NR_c(PyObject *self, PyObject *args);
    static PyObject *IK_LM_c(PyObject *self, PyObject *args);
    // static PyObject *IK_LM_Chan_c(PyObject *self, PyObject *args);
    // static PyObject *IK_LM_Wampler_c(PyObject *self, PyObject *args);
    // static PyObject *IK_LM_Sugihara_c(PyObject *self, PyObject *args);

    static PyObject *Robot_link_T(PyObject *self, PyObject *args);

    static PyObject *ETS_hessian0(PyObject *self, PyObject *args);
    static PyObject *ETS_hessiane(PyObject *self, PyObject *args);
    static PyObject *ETS_jacob0(PyObject *self, PyObject *args);
    static PyObject *ETS_jacobe(PyObject *self, PyObject *args);
    static PyObject *ETS_fkine(PyObject *self, PyObject *args);
    static PyObject *ETS_init(PyObject *self, PyObject *args);

    static PyObject *ET_init(PyObject *self, PyObject *args);
    static PyObject *ET_update(PyObject *self, PyObject *args);
    static PyObject *ET_T(PyObject *self, PyObject *args);

    static PyObject *r2q(PyObject *self, PyObject *args);
    int _check_array_type(PyObject *toCheck);

    void rx(npy_float64 *data, double eta);
    void ry(npy_float64 *data, double eta);
    void rz(npy_float64 *data, double eta);
    void tx(npy_float64 *data, double eta);
    void ty(npy_float64 *data, double eta);
    void tz(npy_float64 *data, double eta);

#ifdef __cplusplus
} /* extern "C" */
#endif /* __cplusplus */

#endif