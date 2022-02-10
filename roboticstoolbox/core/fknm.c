/**
 * \file fknm.c
 * \author Jesse Haviland
 *
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "fknm.h"
#include <stdio.h>

// forward defines
static PyObject *Robot_link_T(PyObject *self, PyObject *args);
static PyObject *ETS_hessian0(PyObject *self, PyObject *args);
static PyObject *ETS_hessiane(PyObject *self, PyObject *args);
static PyObject *ETS_jacob0(PyObject *self, PyObject *args);
static PyObject *ETS_jacobe(PyObject *self, PyObject *args);
static PyObject *ETS_fkine(PyObject *self, PyObject *args);
static PyObject *ET_init(PyObject *self, PyObject *args);
static PyObject *ET_update(PyObject *self, PyObject *args);
static PyObject *ET_T(PyObject *self, PyObject *args);
static PyObject *fkine_all(PyObject *self, PyObject *args);
static PyObject *jacob0(PyObject *self, PyObject *args);
static PyObject *jacobe(PyObject *self, PyObject *args);
static PyObject *fkine(PyObject *self, PyObject *args);
static PyObject *link_init(PyObject *self, PyObject *args);
static PyObject *link_A(PyObject *self, PyObject *args);
static PyObject *link_update(PyObject *self, PyObject *args);
static PyObject *compose(PyObject *self, PyObject *args);
static PyObject *r2q(PyObject *self, PyObject *args);

int _check_array_type(PyObject *toCheck);
void _ETS_hessian(int n, npy_float64 *J, npy_float64 *H);
void _ETS_jacob0(PyObject *ets, int n, npy_float64 *q, npy_float64 *tool, npy_float64 *J);
void _ETS_jacobe(PyObject *ets, int n, npy_float64 *q, npy_float64 *tool, npy_float64 *J);
void _ETS_fkine(PyObject *ets, npy_float64 *q, npy_float64 *base, npy_float64 *tool, npy_float64 *ret);
void _ET_T(ET *et, npy_float64 *ret, double eta);
void _jacob0(PyObject *links, int m, int n, npy_float64 *q, npy_float64 *etool, npy_float64 *tool, npy_float64 *J);
void _jacobe(PyObject *links, int m, int n, npy_float64 *q, npy_float64 *etool, npy_float64 *tool, npy_float64 *J);
void _fkine(PyObject *links, int n, npy_float64 *q, npy_float64 *etool, npy_float64 *tool, npy_float64 *ret);
void A(Link *link, npy_float64 *ret, double eta);
void mult(npy_float64 *A, npy_float64 *B, npy_float64 *C);
void copy(npy_float64 *A, npy_float64 *B);
void rx(npy_float64 *data, double eta);
void ry(npy_float64 *data, double eta);
void rz(npy_float64 *data, double eta);
void tx(npy_float64 *data, double eta);
void ty(npy_float64 *data, double eta);
void tz(npy_float64 *data, double eta);
void _eye(npy_float64 *data);
void _inv(npy_float64 *m, npy_float64 *invOut);
void _r2q(npy_float64 *r, npy_float64 *q);
void _cross(npy_float64 *a, npy_float64 *b, npy_float64 *ret, int n);

static PyMethodDef fknmMethods[] = {
    {"Robot_link_T",
     (PyCFunction)Robot_link_T,
     METH_VARARGS,
     "Link"},
    {"ETS_hessian0",
     (PyCFunction)ETS_hessian0,
     METH_VARARGS,
     "Link"},
    {"ETS_hessiane",
     (PyCFunction)ETS_hessiane,
     METH_VARARGS,
     "Link"},
    {"ETS_jacobe",
     (PyCFunction)ETS_jacobe,
     METH_VARARGS,
     "Link"},
    {"ETS_jacob0",
     (PyCFunction)ETS_jacob0,
     METH_VARARGS,
     "Link"},
    {"ETS_fkine",
     (PyCFunction)ETS_fkine,
     METH_VARARGS,
     "Link"},
    {"ET_update",
     (PyCFunction)ET_update,
     METH_VARARGS,
     "Link"},
    {"ET_init",
     (PyCFunction)ET_init,
     METH_VARARGS,
     "Link"},
    {"ET_T",
     (PyCFunction)ET_T,
     METH_VARARGS,
     "Link"},
    {"link_init",
     (PyCFunction)link_init,
     METH_VARARGS,
     "Link"},
    {"link_A",
     (PyCFunction)link_A,
     METH_VARARGS,
     "Link"},
    {"link_update",
     (PyCFunction)link_update,
     METH_VARARGS,
     "Link"},
    {"fkine",
     (PyCFunction)fkine,
     METH_VARARGS,
     "Link"},
    {"compose",
     (PyCFunction)compose,
     METH_VARARGS,
     "Link"},
    {"r2q",
     (PyCFunction)r2q,
     METH_VARARGS,
     "Link"},
    {"jacob0",
     (PyCFunction)jacob0,
     METH_VARARGS,
     "Link"},
    {"jacobe",
     (PyCFunction)jacobe,
     METH_VARARGS,
     "Link"},
    {"fkine_all",
     (PyCFunction)fkine_all,
     METH_VARARGS,
     "Link"},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

static struct PyModuleDef fknmmodule =
    {
        PyModuleDef_HEAD_INIT,
        "fknm",
        "Fast Kinematics",
        -1,
        fknmMethods};

PyMODINIT_FUNC PyInit_fknm(void)
{
    import_array();
    return PyModule_Create(&fknmmodule);
}

static PyObject *Robot_link_T(PyObject *self, PyObject *args)
{
    npy_float64 *q, *T = NULL;
    PyObject *py_q, *py_np_q;
    PyArrayObject *py_self_q;
    PyObject *ets;
    PyObject *ets_list, *T_list;
    PyObject *iter_ets_list, *iter_T_list;
    int q_used = 0;
    Py_ssize_t n_links;
    PyArray_Descr *desc_q;

    if (!PyArg_ParseTuple(
            args, "OOO!O",
            &ets_list,
            &T_list,
            &PyArray_Type, &py_self_q,
            &py_q))
        return NULL;

    // Make sure q is number array
    // Cast to numpy array
    // Get data out
    if (py_q == Py_None || !_check_array_type(py_q))
    {
        q = (npy_float64 *)PyArray_DATA(py_self_q);
    }
    else
    {
        py_np_q = (npy_float64 *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        q = (npy_float64 *)PyArray_DATA(py_np_q);
        q_used = 1;
    }

    n_links = PyList_GET_SIZE(ets_list);
    for (int i = 0; i < n_links; i++)
    {
        PyObject *ets = PyList_GET_ITEM(ets_list, i);
        npy_float64 *T = (npy_float64 *)PyArray_DATA((PyArrayObject *)PyList_GET_ITEM(T_list, i));

        _ETS_fkine(ets, q, NULL, NULL, T);
    }

    // Free the memory
    if (q_used)
    {
        Py_DECREF(py_np_q);
    }

    return Py_None;
}

static PyObject *ETS_hessian0(PyObject *self, PyObject *args)
{
    npy_float64 *H, *J, *q, *T, *tool = NULL;
    PyObject *py_q, *py_J, *py_tool, *py_np_q, *py_np_tool, *py_np_J;
    PyObject *ets;
    int n, tool_used = 0, J_used = 0, q_used = 0;
    PyArray_Descr *desc_q, *desc_tool;

    if (!PyArg_ParseTuple(
            args, "iOOOO",
            &n,
            &ets,
            &py_q,
            &py_J,
            &py_tool))
        return NULL;

    // Check if J is None
    // Make sure J is number array
    // Cast to numpy array
    // Get data out
    if (py_J != Py_None)
    {
        if (!_check_array_type(py_J))
            return NULL;
        J_used = 1;
        py_np_J = (npy_float64 *)PyArray_FROMANY(py_J, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        J = (npy_float64 *)PyArray_DATA(py_np_J);
    }
    else
    {
        // Now we must use q instead
        // Make sure q is number array
        // Cast to numpy array
        // Get data out
        if (!_check_array_type(py_q))
            return NULL;
        q_used = 1;
        py_np_q = (npy_float64 *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        q = (npy_float64 *)PyArray_DATA(py_np_q);

        // Make our empty Jacobian
        npy_intp dimsJ[2] = {6, n};
        PyObject *py_J = PyArray_EMPTY(2, &dimsJ, NPY_DOUBLE, 0);
        J = (npy_float64 *)PyArray_DATA(py_J);

        // Check if tool is None
        // Make sure tool is number array
        // Cast to numpy array
        // Get data out
        if (py_tool != Py_None)
        {
            if (!_check_array_type(py_tool))
                return NULL;
            tool_used = 1;
            py_np_tool = (npy_float64 *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
            tool = (npy_float64 *)PyArray_DATA(py_np_tool);
        }

        // Calculate the Jacobian
        _ETS_jacob0(ets, n, q, tool, J);
    }

    // Make our empty Hessian
    npy_intp dimsH[3] = {n, 6, n};
    PyObject *py_H = PyArray_EMPTY(3, &dimsH, NPY_DOUBLE, 0);
    H = (npy_float64 *)PyArray_DATA(py_H);

    // Do the job
    _ETS_hessian(n, J, H);

    // Free the memory
    if (q_used)
    {
        Py_DECREF(py_np_q);
    }

    if (J_used)
    {
        Py_DECREF(py_np_J);
    }

    if (tool_used)
    {
        Py_DECREF(py_np_tool);
    }

    return py_H;
    // return Py_None;
}

static PyObject *ETS_hessiane(PyObject *self, PyObject *args)
{
    npy_float64 *H, *J, *q, *T, *tool = NULL;
    PyObject *py_q, *py_J, *py_tool, *py_np_q, *py_np_tool, *py_np_J;
    PyObject *ets;
    int n, tool_used = 0, J_used = 0, q_used = 0;
    PyArray_Descr *desc_q, *desc_tool;

    if (!PyArg_ParseTuple(
            args, "iOOOO",
            &n,
            &ets,
            &py_q,
            &py_J,
            &py_tool))
        return NULL;

    // Check if J is None
    // Make sure J is number array
    // Cast to numpy array
    // Get data out
    if (py_J != Py_None)
    {
        if (!_check_array_type(py_J))
            return NULL;
        J_used = 1;
        py_np_J = (npy_float64 *)PyArray_FROMANY(py_J, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        J = (npy_float64 *)PyArray_DATA(py_np_J);
    }
    else
    {
        // Now we must use q instead
        // Make sure q is number array
        // Cast to numpy array
        // Get data out
        if (!_check_array_type(py_q))
            return NULL;
        q_used = 1;
        py_np_q = (npy_float64 *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        q = (npy_float64 *)PyArray_DATA(py_np_q);

        // Make our empty Jacobian
        npy_intp dimsJ[2] = {6, n};
        PyObject *py_J = PyArray_EMPTY(2, &dimsJ, NPY_DOUBLE, 0);
        J = (npy_float64 *)PyArray_DATA(py_J);

        // Check if tool is None
        // Make sure tool is number array
        // Cast to numpy array
        // Get data out
        if (py_tool != Py_None)
        {
            if (!_check_array_type(py_tool))
                return NULL;
            tool_used = 1;
            py_np_tool = (npy_float64 *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
            tool = (npy_float64 *)PyArray_DATA(py_np_tool);
        }

        // Calculate the Jacobian
        _ETS_jacobe(ets, n, q, tool, J);
    }

    // Make our empty Hessian
    npy_intp dimsH[3] = {n, 6, n};
    PyObject *py_H = PyArray_EMPTY(3, &dimsH, NPY_DOUBLE, 0);
    H = (npy_float64 *)PyArray_DATA(py_H);

    // Do the job
    _ETS_hessian(n, J, H);

    // Free the memory
    if (q_used)
    {
        Py_DECREF(py_np_q);
    }

    if (J_used)
    {
        Py_DECREF(py_np_J);
    }

    if (tool_used)
    {
        Py_DECREF(py_np_tool);
    }

    return py_H;
    // return Py_None;
}

static PyObject *ETS_jacob0(PyObject *self, PyObject *args)
{
    npy_float64 *J, *q, *T, *tool = NULL;
    PyObject *py_q, *py_tool, *py_np_q, *py_np_tool;
    PyObject *ets;
    int n, tool_used = 0;
    PyArray_Descr *desc_q, *desc_tool;

    if (!PyArg_ParseTuple(
            args, "iOOO",
            &n,
            &ets,
            &py_q,
            &py_tool))
        return NULL;

    // Inputs can be:
    // None - Even q
    // Not arrays - Will raise exception
    // Have symbolic data - Will raise exception
    // q can be 1D or 2D, assumes dimesnions correct (n, 1xn or nx1)
    // tool can be SE3s or 4x4 numpy array

    // Make our empty Jacobian
    npy_intp dims[2] = {6, n};
    PyObject *py_J = PyArray_EMPTY(2, &dims, NPY_DOUBLE, 0);
    J = (npy_float64 *)PyArray_DATA(py_J);

    // Make sure q is number array
    // Cast to numpy array
    // Get data out
    if (!_check_array_type(py_q))
        return NULL;
    py_np_q = (npy_float64 *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
    q = (npy_float64 *)PyArray_DATA(py_np_q);

    // Check if tool is None
    // Make sure tool is number array
    // Cast to numpy array
    // Get data out
    if (py_tool != Py_None)
    {
        if (!_check_array_type(py_tool))
            return NULL;
        tool_used = 1;
        py_np_tool = (npy_float64 *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        tool = (npy_float64 *)PyArray_DATA(py_np_tool);
    }

    // Do the job
    _ETS_jacob0(ets, n, q, tool, J);

    // Free the memory
    Py_DECREF(py_np_q);

    if (tool_used)
    {
        Py_DECREF(py_np_tool);
    }

    return py_J;
}

static PyObject *ETS_jacobe(PyObject *self, PyObject *args)
{
    npy_float64 *J, *q, *T, *tool = NULL;
    PyObject *py_q, *py_tool, *py_np_q, *py_np_tool;
    PyObject *ets;
    int n, tool_used = 0;
    PyArray_Descr *desc_q, *desc_tool;

    if (!PyArg_ParseTuple(
            args, "iOOO",
            &n,
            &ets,
            &py_q,
            &py_tool))
        return NULL;

    // Inputs can be:
    // None - Even q
    // Not arrays - Will raise exception
    // Have symbolic data - Will raise exception
    // q can be 1D or 2D, assumes dimesnions correct (n, 1xn or nx1)
    // tool can be SE3s or 4x4 numpy array

    // Make our empty Jacobian
    npy_intp dims[2] = {6, n};
    PyObject *py_J = PyArray_EMPTY(2, &dims, NPY_DOUBLE, 0);
    J = (npy_float64 *)PyArray_DATA(py_J);

    // Make sure q is number array
    // Cast to numpy array
    // Get data out
    if (!_check_array_type(py_q))
        return NULL;
    py_np_q = (npy_float64 *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
    q = (npy_float64 *)PyArray_DATA(py_np_q);

    // Check if tool is None
    // Make sure tool is number array
    // Cast to numpy array
    // Get data out
    if (py_tool != Py_None)
    {
        if (!_check_array_type(py_tool))
            return NULL;
        tool_used = 1;
        py_np_tool = (npy_float64 *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        tool = (npy_float64 *)PyArray_DATA(py_np_tool);
    }

    // Do the job
    _ETS_jacobe(ets, n, q, tool, J);

    // Free the memory
    Py_DECREF(py_np_q);

    if (tool_used)
    {
        Py_DECREF(py_np_tool);
    }

    return py_J;
}

static PyObject *ETS_fkine(PyObject *self, PyObject *args)
{
    npy_intp dim2[2] = {4, 4}, dim3[3] = {1, 4, 4};
    int include_base, n, q_nd, trajn = 1, tool_used = 0, base_used = 0, nd = 2;
    npy_float64 *ret, *retp, *q, *qp, *base = NULL, *tool = NULL;
    PyObject *py_q, *py_base, *py_tool, *py_np_q, *py_np_tool, *py_np_base;
    PyObject *py_ret;
    PyObject *ets;
    npy_intp *q_shape;

    if (!PyArg_ParseTuple(
            args, "OOOOi",
            &ets,
            &py_q,
            &py_base,
            &py_tool,
            &include_base))
        return NULL;

    // Inputs can be:
    // None - Even q
    // Not arrays - Will raise exception
    // Have symbolic data - Will raise exception
    // q can be 2D or 1D, but assumes dimesnions correct (n, 1xn or nx1)
    // base and tool can be SE3s or 4x4 numpy array

    // Make sure q is number array
    // Cast to numpy array
    // Get data out
    if (!_check_array_type(py_q))
        return NULL;
    py_np_q = (npy_float64 *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
    q = (npy_float64 *)PyArray_DATA(py_np_q);

    // Check the dimesnions of q
    q_nd = PyArray_NDIM(py_np_q);
    q_shape = PyArray_SHAPE(py_np_q);

    // Work out how long the trajectory is
    if (q_nd > 1)
    {
        if (q_shape[0] == 1)
        {
            // We have a single q vector
            trajn = 1;
            n = q_shape[1];
        }
        else if (q_shape[1] == 1)
        {
            // We have a single q vector
            trajn = 1;
            n = q_shape[0];
        }
        else
        {
            // We have a trajectory of q
            trajn = q_shape[0];
            n = q_shape[1];
        }
    }

    // Allocate return array
    if (trajn == 1)
    {
        py_ret = PyArray_EMPTY(2, &dim2, NPY_DOUBLE, 0);
    }
    else
    {
        dim3[0] = trajn;
        py_ret = PyArray_EMPTY(3, &dim3, NPY_DOUBLE, 0);
    }

    // Get numpy reference to return array
    ret = (npy_float64 *)PyArray_DATA(py_ret);

    // Check if base is None
    // Make sure base is number array
    // Cast to numpy array
    // Get data out
    if (py_base != Py_None)
    {
        if (!_check_array_type(py_base))
            return NULL;

        if (include_base)
        {
            base_used = 1;
            py_np_base = (npy_float64 *)PyArray_FROMANY(py_base, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
            base = (npy_float64 *)PyArray_DATA(py_np_base);
        }
    }

    if (py_tool != Py_None)
    {
        if (!_check_array_type(py_tool))
            return NULL;
        tool_used = 1;
        py_np_tool = (npy_float64 *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        tool = (npy_float64 *)PyArray_DATA(py_np_tool);
    }

    // Do the actual job
    for (int i = 0; i < trajn; i++)
    {
        // Get pointers to the new section of return array and q array
        retp = ret + (4 * 4 * i);
        qp = q + (n * i);
        _ETS_fkine(ets, qp, base, tool, retp);
    }

    // Free memory
    Py_DECREF(py_np_q);

    if (tool_used)
        Py_DECREF(py_np_tool);

    if (base_used)
        Py_DECREF(py_np_base);

    return py_ret;
}

static PyObject *ET_update(PyObject *self, PyObject *args)
{
    ET *et;
    int jointtype;
    PyObject *ret, *py_et;
    PyArrayObject *py_T, *py_qlim;
    int isjoint, isflip, jindex;

    et = (ET *)PyMem_RawMalloc(sizeof(ET));

    if (!PyArg_ParseTuple(args, "OiiiiO!O!",
                          &py_et,
                          &isjoint,
                          &isflip,
                          &jindex,
                          &jointtype,
                          &PyArray_Type, &py_T,
                          &PyArray_Type, &py_qlim))
        return NULL;

    if (!(et = (ET *)PyCapsule_GetPointer(py_et, "ET")))
        return NULL;

    et->T = (npy_float64 *)PyArray_DATA(py_T);
    et->qlim = (npy_float64 *)PyArray_DATA(py_qlim);
    et->axis = jointtype;

    et->isjoint = isjoint;
    et->isflip = isflip;
    et->jindex = jindex;

    if (jointtype == 0)
    {
        et->op = rx;
    }
    else if (jointtype == 1)
    {
        et->op = ry;
    }
    else if (jointtype == 2)
    {
        et->op = rz;
    }
    else if (jointtype == 3)
    {
        et->op = tx;
    }
    else if (jointtype == 4)
    {
        et->op = ty;
    }
    else if (jointtype == 5)
    {
        et->op = tz;
    }

    ret = PyCapsule_New(et, "ET", NULL);
    return ret;
}

static PyObject *ET_init(PyObject *self, PyObject *args)
{
    ET *et;
    int jointtype;
    PyObject *ret;
    PyArrayObject *py_T, *py_qlim;

    et = (ET *)PyMem_RawMalloc(sizeof(ET));

    if (!PyArg_ParseTuple(args, "iiiiO!O!",
                          &et->isjoint,
                          &et->isflip,
                          &et->jindex,
                          &jointtype,
                          &PyArray_Type, &py_T,
                          &PyArray_Type, &py_qlim))
        return NULL;

    et->T = (npy_float64 *)PyArray_DATA(py_T);
    et->qlim = (npy_float64 *)PyArray_DATA(py_qlim);

    et->axis = jointtype;

    if (jointtype == 0)
    {
        et->op = rx;
    }
    else if (jointtype == 1)
    {
        et->op = ry;
    }
    else if (jointtype == 2)
    {
        et->op = rz;
    }
    else if (jointtype == 3)
    {
        et->op = tx;
    }
    else if (jointtype == 4)
    {
        et->op = ty;
    }
    else if (jointtype == 5)
    {
        et->op = tz;
    }

    ret = PyCapsule_New(et, "ET", NULL);
    return ret;
}

static PyObject *ET_T(PyObject *self, PyObject *args)
{
    npy_intp dims[2] = {4, 4};
    int nd = 2;
    ET *et;
    PyObject *py_et, *py_eta;
    PyObject *py_ret = PyArray_EMPTY(nd, &dims, NPY_DOUBLE, 0);
    double eta = 0;
    npy_float64 *ret;

    if (!PyArg_ParseTuple(args, "OO", &py_et, &py_eta))
        return NULL;

    if (!(et = (ET *)PyCapsule_GetPointer(py_et, "ET")))
        return NULL;

    if (py_eta != Py_None)
    {
        if (PyFloat_Check(py_eta))
        {
            eta = (double)PyFloat_AsDouble(py_eta);
        }
        else
        {
            PyErr_SetString(PyExc_TypeError, "Symbolic value");
            return NULL;
        }
    }

    ret = (npy_float64 *)PyArray_DATA(py_ret);
    _ET_T(et, ret, eta);

    return py_ret;
}

static PyObject *fkine_all(PyObject *self, PyObject *args)
{
    Link *link;
    npy_float64 *q, *base, *ret;
    PyArrayObject *py_q, *py_base;
    PyObject *links, *iter_links;
    int m;

    if (!PyArg_ParseTuple(
            args, "iOO!O!",
            &m,
            &links,
            &PyArray_Type, &py_q,
            &PyArray_Type, &py_base))
        return NULL;

    q = (npy_float64 *)PyArray_DATA(py_q);
    base = (npy_float64 *)PyArray_DATA(py_base);
    iter_links = PyObject_GetIter(links);
    ret = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));

    // Loop through each link in links which is m long
    for (int i = 0; i < m; i++)
    {
        if (!(link = (Link *)PyCapsule_GetPointer(PyIter_Next(iter_links), "Link")))
        {
            return NULL;
        }

        // Calculate the current link transform
        A(link, ret, q[link->jindex]);

        if (link->parent)
        {
            // Multiply parent._fk by link.A and store in link._fk
            mult(link->parent->fk, ret, link->fk);
        }
        else
        {
            // Multiply base by link.A and store in link._fk
            mult(base, ret, link->fk);
        }

        // Set dependant shapes
        for (int i = 0; i < link->n_shapes; i++)
        {
            copy(link->fk, link->shape_wT[i]);
            mult(link->fk, link->shape_base[i], link->shape_sT[i]);
            _r2q(link->shape_sT[i], link->shape_sq[i]);
        }
    }

    Py_DECREF(iter_links);
    free(ret);

    Py_RETURN_NONE;
}

static PyObject *jacobe(PyObject *self, PyObject *args)
{
    npy_float64 *J, *q, *etool, *tool;
    PyArrayObject *py_J, *py_q, *py_tool, *py_etool;
    PyObject *links;
    int m, n;

    if (!PyArg_ParseTuple(
            args, "iiOO!O!O!O!",
            &m,
            &n,
            &links,
            &PyArray_Type, &py_q,
            &PyArray_Type, &py_etool,
            &PyArray_Type, &py_tool,
            &PyArray_Type, &py_J))
        return NULL;

    q = (npy_float64 *)PyArray_DATA(py_q);
    J = (npy_float64 *)PyArray_DATA(py_J);
    tool = (npy_float64 *)PyArray_DATA(py_tool);
    etool = (npy_float64 *)PyArray_DATA(py_etool);

    _jacobe(links, m, n, q, etool, tool, J);

    Py_RETURN_NONE;
}

static PyObject *jacob0(PyObject *self, PyObject *args)
{
    npy_float64 *J, *q, *etool, *tool;
    PyArrayObject *py_J, *py_q, *py_tool, *py_etool;
    PyObject *links;
    int m, n;

    if (!PyArg_ParseTuple(
            args, "iiOO!O!O!O!",
            &m,
            &n,
            &links,
            &PyArray_Type, &py_q,
            &PyArray_Type, &py_etool,
            &PyArray_Type, &py_tool,
            &PyArray_Type, &py_J))
        return NULL;

    q = (npy_float64 *)PyArray_DATA(py_q);
    J = (npy_float64 *)PyArray_DATA(py_J);
    tool = (npy_float64 *)PyArray_DATA(py_tool);
    etool = (npy_float64 *)PyArray_DATA(py_etool);

    _jacob0(links, m, n, q, etool, tool, J);

    Py_RETURN_NONE;
}

static PyObject *fkine(PyObject *self, PyObject *args)
{
    npy_float64 *ret, *q, *etool, *tool;
    PyArrayObject *py_ret, *py_q, *py_etool, *py_tool;
    PyObject *links;
    int n;

    if (!PyArg_ParseTuple(
            args, "iOO!O!O!O!",
            &n,
            &links,
            &PyArray_Type, &py_q,
            &PyArray_Type, &py_etool,
            &PyArray_Type, &py_tool,
            &PyArray_Type, &py_ret))
        return NULL;

    q = (npy_float64 *)PyArray_DATA(py_q);
    ret = (npy_float64 *)PyArray_DATA(py_ret);
    tool = (npy_float64 *)PyArray_DATA(py_tool);
    etool = (npy_float64 *)PyArray_DATA(py_etool);

    _fkine(links, n, q, etool, tool, ret);

    Py_RETURN_NONE;
}

static PyObject *link_init(PyObject *self, PyObject *args)
{
    Link *link, *parent;
    int jointtype;
    PyObject *ret, *py_parent;

    PyObject *py_shape_base, *py_shape_wT, *py_shape_sT, *py_shape_sq;
    PyObject *iter_base, *iter_wT, *iter_sT, *iter_sq;
    PyArrayObject *pys_base, *pys_wT, *pys_sT, *pys_sq;
    PyArrayObject *py_A, *py_fk;

    link = (Link *)PyMem_RawMalloc(sizeof(Link));

    if (!PyArg_ParseTuple(args, "iiiiiO!O!OOOOO",
                          &link->isjoint,
                          &link->isflip,
                          &jointtype,
                          &link->jindex,
                          &link->n_shapes,
                          &PyArray_Type, &py_A,
                          &PyArray_Type, &py_fk,
                          &py_shape_base,
                          &py_shape_wT,
                          &py_shape_sT,
                          &py_shape_sq,
                          &py_parent))
        return NULL;

    if (py_parent == Py_None)
    {
        parent = NULL;
    }
    else if (!(parent = (Link *)PyCapsule_GetPointer(py_parent, "Link")))
    {
        return NULL;
    }

    link->A = (npy_float64 *)PyArray_DATA(py_A);
    link->fk = (npy_float64 *)PyArray_DATA(py_fk);

    // Set shape pointers
    iter_base = PyObject_GetIter(py_shape_base);
    iter_wT = PyObject_GetIter(py_shape_wT);
    iter_sT = PyObject_GetIter(py_shape_sT);
    iter_sq = PyObject_GetIter(py_shape_sq);

    link->shape_base = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));
    link->shape_wT = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));
    link->shape_sT = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));
    link->shape_sq = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));

    for (int i = 0; i < link->n_shapes; i++)
    {
        if (
            !(pys_base = (PyArrayObject *)PyIter_Next(iter_base)) ||
            !(pys_wT = (PyArrayObject *)PyIter_Next(iter_wT)) ||
            !(pys_sT = (PyArrayObject *)PyIter_Next(iter_sT)) ||
            !(pys_sq = (PyArrayObject *)PyIter_Next(iter_sq)))
            return NULL;

        link->shape_base[i] = (npy_float64 *)PyArray_DATA(pys_base);
        link->shape_wT[i] = (npy_float64 *)PyArray_DATA(pys_wT);
        link->shape_sT[i] = (npy_float64 *)PyArray_DATA(pys_sT);
        link->shape_sq[i] = (npy_float64 *)PyArray_DATA(pys_sq);
    }

    link->axis = jointtype;
    link->parent = parent;

    if (jointtype == 0)
    {
        link->op = rx;
    }
    else if (jointtype == 1)
    {
        link->op = ry;
    }
    else if (jointtype == 2)
    {
        link->op = rz;
    }
    else if (jointtype == 3)
    {
        link->op = tx;
    }
    else if (jointtype == 4)
    {
        link->op = ty;
    }
    else if (jointtype == 5)
    {
        link->op = tz;
    }

    Py_DECREF(iter_base);
    Py_DECREF(iter_wT);
    Py_DECREF(iter_sT);
    Py_DECREF(iter_sq);

    ret = PyCapsule_New(link, "Link", NULL);
    return ret;
}

static PyObject *link_update(PyObject *self, PyObject *args)
{
    Link *link, *parent;
    int isjoint, isflip;
    int jointtype, jindex, n_shapes;
    PyObject *lo, *py_parent;
    PyArrayObject *py_A, *py_fk;

    PyObject *py_shape_base, *py_shape_wT, *py_shape_sT, *py_shape_sq;
    PyObject *iter_base, *iter_wT, *iter_sT, *iter_sq;
    PyArrayObject *pys_base, *pys_wT, *pys_sT, *pys_sq;

    if (!PyArg_ParseTuple(args, "OiiiiiO!O!OOOOO",
                          &lo,
                          &isjoint,
                          &isflip,
                          &jointtype,
                          &jindex,
                          &n_shapes,
                          &PyArray_Type, &py_A,
                          &PyArray_Type, &py_fk,
                          &py_shape_base,
                          &py_shape_wT,
                          &py_shape_sT,
                          &py_shape_sq,
                          &py_parent))
        return NULL;

    if (py_parent == Py_None)
    {
        parent = NULL;
    }
    else if (!(parent = (Link *)PyCapsule_GetPointer(py_parent, "Link")))
    {
        return NULL;
    }

    if (!(link = (Link *)PyCapsule_GetPointer(lo, "Link")))
    {
        return NULL;
    }

    // Set shape pointers
    iter_base = PyObject_GetIter(py_shape_base);
    iter_wT = PyObject_GetIter(py_shape_wT);
    iter_sT = PyObject_GetIter(py_shape_sT);
    iter_sq = PyObject_GetIter(py_shape_sq);

    if (link->shape_base != 0)
        free(link->shape_base);
    if (link->shape_wT != 0)
        free(link->shape_wT);
    if (link->shape_sT != 0)
        free(link->shape_sT);
    if (link->shape_sq != 0)
        free(link->shape_sq);

    link->shape_base = 0;
    link->shape_wT = 0;
    link->shape_sT = 0;
    link->shape_sq = 0;

    link->shape_base = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));
    link->shape_wT = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));
    link->shape_sT = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));
    link->shape_sq = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));

    for (int i = 0; i < n_shapes; i++)
    {
        if (
            !(pys_base = (PyArrayObject *)PyIter_Next(iter_base)) ||
            !(pys_wT = (PyArrayObject *)PyIter_Next(iter_wT)) ||
            !(pys_sT = (PyArrayObject *)PyIter_Next(iter_sT)) ||
            !(pys_sq = (PyArrayObject *)PyIter_Next(iter_sq)))
            return NULL;

        link->shape_base[i] = (npy_float64 *)PyArray_DATA(pys_base);
        link->shape_wT[i] = (npy_float64 *)PyArray_DATA(pys_wT);
        link->shape_sT[i] = (npy_float64 *)PyArray_DATA(pys_sT);
        link->shape_sq[i] = (npy_float64 *)PyArray_DATA(pys_sq);
    }

    if (jointtype == 0)
    {
        link->op = rx;
    }
    else if (jointtype == 1)
    {
        link->op = ry;
    }
    else if (jointtype == 2)
    {
        link->op = rz;
    }
    else if (jointtype == 3)
    {
        link->op = tx;
    }
    else if (jointtype == 4)
    {
        link->op = ty;
    }
    else if (jointtype == 5)
    {
        link->op = tz;
    }

    link->isjoint = isjoint;
    link->isflip = isflip;
    link->A = (npy_float64 *)PyArray_DATA(py_A);
    link->fk = (npy_float64 *)PyArray_DATA(py_fk);
    link->jindex = jindex;
    link->axis = jointtype;
    link->parent = parent;
    link->n_shapes = n_shapes;

    Py_DECREF(iter_base);
    Py_DECREF(iter_wT);
    Py_DECREF(iter_sT);
    Py_DECREF(iter_sq);

    Py_RETURN_NONE;
}

static PyObject *link_A(PyObject *self, PyObject *args)
{
    Link *link;
    PyArrayObject *py_ret;
    PyObject *lo;
    npy_float64 *ret;
    double eta;

    if (!PyArg_ParseTuple(args, "dOO!", &eta, &lo, &PyArray_Type, &py_ret))
        return NULL;

    if (!(link = (Link *)PyCapsule_GetPointer(lo, "Link")))
        return NULL;

    ret = (npy_float64 *)PyArray_DATA(py_ret);
    A(link, ret, eta);

    Py_RETURN_NONE;
}

static PyObject *compose(PyObject *self, PyObject *args)
{
    npy_float64 *A, *B, *C;
    PyArrayObject *py_A, *py_B, *py_C;

    if (!PyArg_ParseTuple(
            args, "O!O!O!",
            &PyArray_Type, &py_A,
            &PyArray_Type, &py_B,
            &PyArray_Type, &py_C))
        return NULL;

    A = (npy_float64 *)PyArray_DATA(py_A);
    B = (npy_float64 *)PyArray_DATA(py_B);
    C = (npy_float64 *)PyArray_DATA(py_C);

    mult(A, B, C);

    Py_RETURN_NONE;
}

static PyObject *r2q(PyObject *self, PyObject *args)
{
    // r is actually an SE3
    npy_float64 *r, *q;
    PyArrayObject *py_r, *py_q;

    if (!PyArg_ParseTuple(
            args, "O!O!",
            &PyArray_Type, &py_r,
            &PyArray_Type, &py_q))
        return NULL;

    r = (npy_float64 *)PyArray_DATA(py_r);
    q = (npy_float64 *)PyArray_DATA(py_q);

    _r2q(r, q);

    Py_RETURN_NONE;
}

int _check_array_type(PyObject *toCheck)
{
    PyArray_Descr *desc;

    desc = PyArray_DescrFromObject(toCheck, NULL);

    // Check if desc is a number or a sympy symbol
    if (!PyDataType_ISNUMBER(desc))
    {
        PyErr_SetString(PyExc_TypeError, "Symbolic value");
        return 0;
    }

    return 1;
}

void _ETS_hessian(int n, npy_float64 *J, npy_float64 *H)
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

void _ETS_jacob0(PyObject *ets, int n, npy_float64 *q, npy_float64 *tool, npy_float64 *J)
{
    ET *et;
    npy_float64 *T = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *U = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *invU = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *temp = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *ret = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    Py_ssize_t m;

    int j = 0;

    _eye(U);

    // Get the forward  kinematics into T
    _ETS_fkine(ets, q, (npy_float64 *)NULL, tool, T);

    PyObject *iter_et = PyObject_GetIter(ets);

    m = PyList_GET_SIZE(ets);
    for (int i = 0; i < m; i++)
    {
        if (!(et = (ET *)PyCapsule_GetPointer(PyIter_Next(iter_et), "ET")))
            return;

        if (et->isjoint)
        {
            _ET_T(et, ret, q[et->jindex]);
            mult(U, ret, temp);
            copy(temp, U);

            if (i == m - 1 && tool != NULL)
            {
                mult(U, tool, temp);
                copy(temp, U);
            }

            _inv(U, invU);
            mult(invU, T, temp);

            if (et->axis == 0)
            {
                J[0 * n + j] = U[0 * 4 + 2] * temp[1 * 4 + 3] - U[0 * 4 + 1] * temp[2 * 4 + 3];
                J[1 * n + j] = U[1 * 4 + 2] * temp[1 * 4 + 3] - U[1 * 4 + 1] * temp[2 * 4 + 3];
                J[2 * n + j] = U[2 * 4 + 2] * temp[1 * 4 + 3] - U[2 * 4 + 1] * temp[2 * 4 + 3];
                J[3 * n + j] = U[0 * 4 + 0];
                J[4 * n + j] = U[1 * 4 + 0];
                J[5 * n + j] = U[2 * 4 + 0];
            }
            else if (et->axis == 1)
            {
                J[0 * n + j] = U[0 * 4 + 0] * temp[2 * 4 + 3] - U[0 * 4 + 2] * temp[0 * 4 + 3];
                J[1 * n + j] = U[1 * 4 + 0] * temp[2 * 4 + 3] - U[1 * 4 + 2] * temp[0 * 4 + 3];
                J[2 * n + j] = U[2 * 4 + 0] * temp[2 * 4 + 3] - U[2 * 4 + 2] * temp[0 * 4 + 3];
                J[3 * n + j] = U[0 * 4 + 1];
                J[4 * n + j] = U[1 * 4 + 1];
                J[5 * n + j] = U[2 * 4 + 1];
            }
            else if (et->axis == 2)
            {
                J[0 * n + j] = U[0 * 4 + 1] * temp[0 * 4 + 3] - U[0 * 4 + 0] * temp[1 * 4 + 3];
                J[1 * n + j] = U[1 * 4 + 1] * temp[0 * 4 + 3] - U[1 * 4 + 0] * temp[1 * 4 + 3];
                J[2 * n + j] = U[2 * 4 + 1] * temp[0 * 4 + 3] - U[2 * 4 + 0] * temp[1 * 4 + 3];
                J[3 * n + j] = U[0 * 4 + 2];
                J[4 * n + j] = U[1 * 4 + 2];
                J[5 * n + j] = U[2 * 4 + 2];
            }
            else if (et->axis == 3)
            {
                J[0 * n + j] = U[0 * 4 + 0];
                J[1 * n + j] = U[1 * 4 + 0];
                J[2 * n + j] = U[2 * 4 + 0];
                J[3 * n + j] = 0.0;
                J[4 * n + j] = 0.0;
                J[5 * n + j] = 0.0;
            }
            else if (et->axis == 4)
            {
                J[0 * n + j] = U[0 * 4 + 1];
                J[1 * n + j] = U[1 * 4 + 1];
                J[2 * n + j] = U[2 * 4 + 1];
                J[3 * n + j] = 0.0;
                J[4 * n + j] = 0.0;
                J[5 * n + j] = 0.0;
            }
            else if (et->axis == 5)
            {
                J[0 * n + j] = U[0 * 4 + 2];
                J[1 * n + j] = U[1 * 4 + 2];
                J[2 * n + j] = U[2 * 4 + 2];
                J[3 * n + j] = 0.0;
                J[4 * n + j] = 0.0;
                J[5 * n + j] = 0.0;
            }
            j++;
        }
        else
        {
            _ET_T(et, ret, q[et->jindex]);
            mult(U, ret, temp);
            copy(temp, U);
        }
    }

    Py_DECREF(iter_et);

    free(T);
    free(U);
    free(temp);
    free(ret);
    free(invU);
}

void _ETS_jacobe(PyObject *ets, int n, npy_float64 *q, npy_float64 *tool, npy_float64 *J)
{
    ET *et;
    npy_float64 *T = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *U = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *temp = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *ret = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    Py_ssize_t m;

    int j = n - 1;

    _eye(U);

    // Get the forward  kinematics into T
    _ETS_fkine(ets, q, (npy_float64 *)NULL, tool, T);

    PyList_Reverse(ets);
    PyObject *iter_et = PyObject_GetIter(ets);

    if (tool != NULL)
    {
        mult(tool, U, temp);
        copy(temp, U);
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
            mult(ret, U, temp);
            copy(temp, U);
            j--;
        }
        else
        {
            _ET_T(et, ret, q[et->jindex]);
            mult(ret, U, temp);
            copy(temp, U);
        }
    }

    PyList_Reverse(ets);

    Py_DECREF(iter_et);

    free(T);
    free(U);
    free(temp);
    free(ret);
}

void _ETS_fkine(PyObject *ets, npy_float64 *q, npy_float64 *base, npy_float64 *tool, npy_float64 *ret)
{
    npy_float64 *temp, *current;
    ET *et;
    Py_ssize_t m;

    temp = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    current = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    PyObject *iter_et = PyObject_GetIter(ets);

    if (base != NULL)
    {
        copy(base, current);
    }
    else
    {
        _eye(current);
    }

    m = PyList_GET_SIZE(ets);
    for (int i = 0; i < m; i++)
    {
        if (!(et = (ET *)PyCapsule_GetPointer(PyIter_Next(iter_et), "ET")))
            return;

        _ET_T(et, ret, q[et->jindex]);
        mult(current, ret, temp);
        copy(temp, current);
    }

    if (tool != NULL)
    {
        mult(current, tool, ret);
    }
    else
    {
        copy(current, ret);
    }

    Py_DECREF(iter_et);

    free(temp);
    free(current);
}

void _ET_T(ET *et, npy_float64 *ret, double eta)
{
    // Check if static and return static transform
    if (!et->isjoint)
    {
        copy(et->T, ret);
        return;
    }

    if (et->isflip)
    {
        eta = -eta;
    }

    // Calculate ET trasform based on eta
    et->op(ret, eta);
}

void _jacobe(PyObject *links, int m, int n, npy_float64 *q, npy_float64 *etool, npy_float64 *tool, npy_float64 *J)
{
    Link *link;
    npy_float64 *T = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *U = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *temp = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *ret = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    int j = n - 1;

    _eye(U);
    _fkine(links, m, q, etool, tool, T);

    PyList_Reverse(links);
    PyObject *iter_links = PyObject_GetIter(links);

    mult(etool, U, temp);
    copy(temp, U);
    mult(tool, U, temp);
    copy(temp, U);

    for (int i = 0; i < m; i++)
    {
        if (!(link = (Link *)PyCapsule_GetPointer(PyIter_Next(iter_links), "Link")))
            return;

        if (link->isjoint)
        {
            if (link->axis == 0)
            {
                J[0 * n + j] = U[2 * 4 + 0] * U[1 * 4 + 3] - U[1 * 4 + 0] * U[2 * 4 + 3];
                J[1 * n + j] = U[2 * 4 + 1] * U[1 * 4 + 3] - U[1 * 4 + 1] * U[2 * 4 + 3];
                J[2 * n + j] = U[2 * 4 + 2] * U[1 * 4 + 3] - U[1 * 4 + 2] * U[2 * 4 + 3];
                J[3 * n + j] = U[0 * 4 + 0];
                J[4 * n + j] = U[0 * 4 + 1];
                J[5 * n + j] = U[0 * 4 + 2];
            }
            else if (link->axis == 1)
            {
                J[0 * n + j] = U[0 * 4 + 0] * U[2 * 4 + 3] - U[2 * 4 + 0] * U[0 * 4 + 3];
                J[1 * n + j] = U[0 * 4 + 1] * U[2 * 4 + 3] - U[2 * 4 + 1] * U[0 * 4 + 3];
                J[2 * n + j] = U[0 * 4 + 2] * U[2 * 4 + 3] - U[2 * 4 + 2] * U[0 * 4 + 3];
                J[3 * n + j] = U[1 * 4 + 0];
                J[4 * n + j] = U[1 * 4 + 1];
                J[5 * n + j] = U[1 * 4 + 2];
            }
            else if (link->axis == 2)
            {
                J[0 * n + j] = U[1 * 4 + 0] * U[0 * 4 + 3] - U[0 * 4 + 0] * U[1 * 4 + 3];
                J[1 * n + j] = U[1 * 4 + 1] * U[0 * 4 + 3] - U[0 * 4 + 1] * U[1 * 4 + 3];
                J[2 * n + j] = U[1 * 4 + 2] * U[0 * 4 + 3] - U[0 * 4 + 2] * U[1 * 4 + 3];
                J[3 * n + j] = U[2 * 4 + 0];
                J[4 * n + j] = U[2 * 4 + 1];
                J[5 * n + j] = U[2 * 4 + 2];
            }
            else if (link->axis == 3)
            {
                J[0 * n + j] = U[0 * 4 + 0];
                J[1 * n + j] = U[0 * 4 + 1];
                J[2 * n + j] = U[0 * 4 + 2];
                J[3 * n + j] = 0.0;
                J[4 * n + j] = 0.0;
                J[5 * n + j] = 0.0;
            }
            else if (link->axis == 4)
            {
                J[0 * n + j] = U[1 * 4 + 0];
                J[1 * n + j] = U[1 * 4 + 1];
                J[2 * n + j] = U[1 * 4 + 2];
                J[3 * n + j] = 0.0;
                J[4 * n + j] = 0.0;
                J[5 * n + j] = 0.0;
            }
            else if (link->axis == 5)
            {
                J[0 * n + j] = U[2 * 4 + 0];
                J[1 * n + j] = U[2 * 4 + 1];
                J[2 * n + j] = U[2 * 4 + 2];
                J[3 * n + j] = 0.0;
                J[4 * n + j] = 0.0;
                J[5 * n + j] = 0.0;
            }

            A(link, ret, q[link->jindex]);
            mult(ret, U, temp);
            copy(temp, U);
            j--;
        }
        else
        {
            A(link, ret, q[link->jindex]);
            mult(ret, U, temp);
            copy(temp, U);
        }
    }
    PyList_Reverse(links);

    Py_DECREF(iter_links);

    free(T);
    free(U);
    free(temp);
    free(ret);
}

void _jacob0(PyObject *links, int m, int n, npy_float64 *q, npy_float64 *etool, npy_float64 *tool, npy_float64 *J)
{
    Link *link;
    npy_float64 *T = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *U = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *temp = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *ret = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *invU = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    int j = 0;

    _eye(U);
    _fkine(links, m, q, etool, tool, T);

    PyObject *iter_links = PyObject_GetIter(links);

    for (int i = 0; i < m; i++)
    {
        if (!(link = (Link *)PyCapsule_GetPointer(PyIter_Next(iter_links), "Link")))
            return;

        if (link->isjoint)
        {
            A(link, ret, q[link->jindex]);
            mult(U, ret, temp);
            copy(temp, U);

            if (i == m - 1)
            {
                mult(U, etool, temp);
                copy(temp, U);
                mult(U, tool, temp);
                copy(temp, U);
            }

            _inv(U, invU);
            mult(invU, T, temp);

            if (link->axis == 0)
            {
                J[0 * n + j] = U[0 * 4 + 2] * temp[1 * 4 + 3] - U[0 * 4 + 1] * temp[2 * 4 + 3];
                J[1 * n + j] = U[1 * 4 + 2] * temp[1 * 4 + 3] - U[1 * 4 + 1] * temp[2 * 4 + 3];
                J[2 * n + j] = U[2 * 4 + 2] * temp[1 * 4 + 3] - U[2 * 4 + 1] * temp[2 * 4 + 3];
                J[3 * n + j] = U[0 * 4 + 2];
                J[4 * n + j] = U[1 * 4 + 2];
                J[5 * n + j] = U[2 * 4 + 2];
            }
            else if (link->axis == 1)
            {
                J[0 * n + j] = U[0 * 4 + 0] * temp[2 * 4 + 3] - U[0 * 4 + 2] * temp[0 * 4 + 3];
                J[1 * n + j] = U[1 * 4 + 0] * temp[2 * 4 + 3] - U[1 * 4 + 2] * temp[0 * 4 + 3];
                J[2 * n + j] = U[2 * 4 + 0] * temp[2 * 4 + 3] - U[2 * 4 + 2] * temp[0 * 4 + 3];
                J[3 * n + j] = U[0 * 4 + 1];
                J[4 * n + j] = U[1 * 4 + 1];
                J[5 * n + j] = U[2 * 4 + 1];
            }
            else if (link->axis == 2)
            {
                J[0 * n + j] = U[0 * 4 + 1] * temp[0 * 4 + 3] - U[0 * 4 + 0] * temp[1 * 4 + 3];
                J[1 * n + j] = U[1 * 4 + 1] * temp[0 * 4 + 3] - U[1 * 4 + 0] * temp[1 * 4 + 3];
                J[2 * n + j] = U[2 * 4 + 1] * temp[0 * 4 + 3] - U[2 * 4 + 0] * temp[1 * 4 + 3];
                J[3 * n + j] = U[0 * 4 + 2];
                J[4 * n + j] = U[1 * 4 + 2];
                J[5 * n + j] = U[2 * 4 + 2];
            }
            else if (link->axis == 3)
            {
                J[0 * n + j] = U[0 * 4 + 0];
                J[1 * n + j] = U[1 * 4 + 0];
                J[2 * n + j] = U[2 * 4 + 0];
                J[3 * n + j] = 0.0;
                J[4 * n + j] = 0.0;
                J[5 * n + j] = 0.0;
            }
            else if (link->axis == 4)
            {
                J[0 * n + j] = U[0 * 4 + 1];
                J[1 * n + j] = U[1 * 4 + 1];
                J[2 * n + j] = U[2 * 4 + 1];
                J[3 * n + j] = 0.0;
                J[4 * n + j] = 0.0;
                J[5 * n + j] = 0.0;
            }
            else if (link->axis == 5)
            {
                J[0 * n + j] = U[0 * 4 + 2];
                J[1 * n + j] = U[1 * 4 + 2];
                J[2 * n + j] = U[2 * 4 + 2];
                J[3 * n + j] = 0.0;
                J[4 * n + j] = 0.0;
                J[5 * n + j] = 0.0;
            }
            j++;
        }
        else
        {
            A(link, ret, q[link->jindex]);
            mult(U, ret, temp);
            copy(temp, U);
        }
    }

    Py_DECREF(iter_links);

    free(T);
    free(U);
    free(temp);
    free(ret);
    free(invU);
}

void _fkine(PyObject *links, int n, npy_float64 *q, npy_float64 *etool, npy_float64 *tool, npy_float64 *ret)
{
    npy_float64 *temp, *current;
    Link *link;

    temp = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    current = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));

    PyObject *iter_links = PyObject_GetIter(links);

    // copy(tool, ret);

    if (!(link = (Link *)PyCapsule_GetPointer(PyIter_Next(iter_links), "Link")))
        return;

    A(link, current, q[link->jindex]);

    for (int i = 1; i < n; i++)
    {
        if (!(link = (Link *)PyCapsule_GetPointer(PyIter_Next(iter_links), "Link")))
            return;

        A(link, ret, q[link->jindex]);
        mult(current, ret, temp);
        copy(temp, current);
    }

    mult(current, etool, ret);
    copy(ret, current);
    mult(current, tool, ret);

    Py_DECREF(iter_links);

    free(temp);
    free(current);
}

void A(Link *link, npy_float64 *ret, double eta)
{
    npy_float64 *v;

    if (!link->isjoint)
    {
        copy(link->A, ret);
        return;
    }

    if (link->isflip)
    {
        eta = -eta;
    }

    // Calculate the variable part of the link
    v = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    link->op(v, eta);

    // Multiply ret = A * v
    mult(link->A, v, ret);
    free(v);
}

void copy(npy_float64 *A, npy_float64 *B)
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

void mult(npy_float64 *A, npy_float64 *B, npy_float64 *C)
{
    const int N = 4;
    int i, j, k;
    double num;

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            num = 0;
            for (k = 0; k < N; k++)
            {
                num += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = num;
        }
    }
}

void rx(npy_float64 *data, double eta)
{
    double st, ct;

    ct = cos(eta);
    st = sin(eta);

    data[0] = 1;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
    data[4] = 0;
    data[5] = ct;
    data[6] = -st;
    data[7] = 0;
    data[8] = 0;
    data[9] = st;
    data[10] = ct;
    data[11] = 0;
    data[12] = 0;
    data[13] = 0;
    data[14] = 0;
    data[15] = 1;
}

void ry(npy_float64 *data, double eta)
{
    double st, ct;

    ct = cos(eta);
    st = sin(eta);

    data[0] = ct;
    data[1] = 0;
    data[2] = st;
    data[3] = 0;
    data[4] = 0;
    data[5] = 1;
    data[6] = 0;
    data[7] = 0;
    data[8] = -st;
    data[9] = 0;
    data[10] = ct;
    data[11] = 0;
    data[12] = 0;
    data[13] = 0;
    data[14] = 0;
    data[15] = 1;
}

void rz(npy_float64 *data, double eta)
{
    double st, ct;

    ct = cos(eta);
    st = sin(eta);

    data[0] = ct;
    data[1] = -st;
    data[2] = 0;
    data[3] = 0;
    data[4] = st;
    data[5] = ct;
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

void tx(npy_float64 *data, double eta)
{
    data[0] = 1;
    data[1] = 0;
    data[2] = 0;
    data[3] = eta;
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

void ty(npy_float64 *data, double eta)
{
    data[0] = 1;
    data[1] = 0;
    data[2] = 0;
    data[3] = 0;
    data[4] = 0;
    data[5] = 1;
    data[6] = 0;
    data[7] = eta;
    data[8] = 0;
    data[9] = 0;
    data[10] = 1;
    data[11] = 0;
    data[12] = 0;
    data[13] = 0;
    data[14] = 0;
    data[15] = 1;
}

void tz(npy_float64 *data, double eta)
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
    data[11] = eta;
    data[12] = 0;
    data[13] = 0;
    data[14] = 0;
    data[15] = 1;
}

void _eye(npy_float64 *data)
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

void _inv(npy_float64 *m, npy_float64 *inv)
{
    inv[0] = m[0];
    inv[1] = m[4];
    inv[2] = m[8];

    inv[4] = m[1];
    inv[5] = m[5];
    inv[6] = m[9];

    inv[8] = m[2];
    inv[9] = m[6];
    inv[10] = m[10];

    inv[3] = -(inv[0] * m[3] + inv[1] * m[7] + inv[2] * m[11]);
    inv[7] = -(inv[4] * m[3] + inv[5] * m[7] + inv[6] * m[11]);
    inv[11] = -(inv[8] * m[3] + inv[9] * m[7] + inv[10] * m[11]);

    inv[12] = 0;
    inv[13] = 0;
    inv[14] = 0;
    inv[15] = 1;
}

void _r2q(npy_float64 *r, npy_float64 *q)
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

void _cross(npy_float64 *a, npy_float64 *b, npy_float64 *ret, int n)
{
    ret[0] = a[1 * n] * b[2 * n] - a[2 * n] * b[1 * n];
    ret[1 * n] = a[2 * n] * b[0] - a[0] * b[2 * n];
    ret[2 * n] = a[0] * b[1 * n] - a[1 * n] * b[0];
    // ret[0] = b[0 * n];
    // ret[1 * n] = b[1 * n];
    // ret[2 * n] = b[2 * n];
}
