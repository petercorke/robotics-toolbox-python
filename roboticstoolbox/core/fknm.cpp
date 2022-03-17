/**
 * \file fknm.cpp
 * \author Jesse Haviland
 *
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "fknm.h"
#include "methods.h"
#include "linalg.h"
#include "structs.h"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>

static PyMethodDef fknmMethods[] = {
    {"IK",
     (PyCFunction)IK,
     METH_VARARGS,
     "Link"},
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
    {"r2q",
     (PyCFunction)r2q,
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

static PyObject *IK(PyObject *self, PyObject *args)
{
    npy_float64 *q, *Tep, *ret;
    PyObject *py_q, *py_Tep;
    PyArrayObject *py_np_q, *py_np_Tep;
    PyObject *ets;
    int n;
    PyObject *py_ret;
    npy_intp dim1[2] = {2, 4};

    if (!PyArg_ParseTuple(
            args, "iOOO",
            &n,
            &ets,
            &py_q,
            &py_Tep))
        return NULL;

    // Make sure q is number array
    // Cast to numpy array
    // Get data out
    if (!_check_array_type(py_q))
        return NULL;
    py_np_q = (PyArrayObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
    q = (npy_float64 *)PyArray_DATA(py_np_q);

    if (!_check_array_type(py_Tep))
        return NULL;

    py_np_Tep = (PyArrayObject *)PyArray_FROMANY(py_Tep, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
    Tep = (npy_float64 *)PyArray_DATA(py_np_Tep);

    py_ret = PyArray_EMPTY(2, dim1, NPY_DOUBLE, 0);
    ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);

    // __mult4(2, 4, )

    _ETS_IK(ets, n, q, Tep, ret);

    // _angle_axis(Te, Tep, ret);

    // Free the memory
    Py_DECREF(py_np_q);
    Py_DECREF(py_np_Tep);

    return py_ret;
}

static PyObject *Robot_link_T(PyObject *self, PyObject *args)
{
    npy_float64 *q;
    PyObject *py_q, *py_np_q;
    PyArrayObject *py_self_q;
    PyObject *ets_list, *T_list;
    int q_used = 0;
    Py_ssize_t n_links;

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
        py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        q = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q);
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
    npy_float64 *H, *J, *q, *tool = NULL;
    PyObject *py_q, *py_J, *py_tool, *py_np_q, *py_np_tool, *py_np_J;
    PyObject *ets;
    int n, tool_used = 0, J_used = 0, q_used = 0;

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
        py_np_J = (PyObject *)PyArray_FROMANY(py_J, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_J);
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
        py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        q = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q);

        // Make our empty Jacobian
        npy_intp dimsJ[2] = {6, n};
        PyObject *py_J = PyArray_EMPTY(2, dimsJ, NPY_DOUBLE, 0);
        J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_J);

        // Check if tool is None
        // Make sure tool is number array
        // Cast to numpy array
        // Get data out
        if (py_tool != Py_None)
        {
            if (!_check_array_type(py_tool))
                return NULL;
            tool_used = 1;
            py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
            tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
        }

        // Calculate the Jacobian
        _ETS_jacob0(ets, n, q, tool, J);
    }

    // Make our empty Hessian
    npy_intp dimsH[3] = {n, 6, n};
    PyObject *py_H = PyArray_EMPTY(3, dimsH, NPY_DOUBLE, 0);
    H = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_H);

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
    npy_float64 *H, *J, *q, *tool = NULL;
    PyObject *py_q, *py_J, *py_tool, *py_np_q, *py_np_tool, *py_np_J;
    PyObject *ets;
    int n, tool_used = 0, J_used = 0, q_used = 0;

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
        py_np_J = (PyObject *)PyArray_FROMANY(py_J, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_J);
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
        py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        q = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q);

        // Make our empty Jacobian
        npy_intp dimsJ[2] = {6, n};
        PyObject *py_J = PyArray_EMPTY(2, dimsJ, NPY_DOUBLE, 0);
        J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_J);

        // Check if tool is None
        // Make sure tool is number array
        // Cast to numpy array
        // Get data out
        if (py_tool != Py_None)
        {
            if (!_check_array_type(py_tool))
                return NULL;
            tool_used = 1;
            py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
            tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
        }

        // Calculate the Jacobian
        _ETS_jacobe(ets, n, q, tool, J);
    }

    // Make our empty Hessian
    npy_intp dimsH[3] = {n, 6, n};
    PyObject *py_H = PyArray_EMPTY(3, dimsH, NPY_DOUBLE, 0);
    H = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_H);

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
    npy_float64 *J, *q, *tool = NULL;
    PyObject *py_q, *py_tool, *py_np_q, *py_np_tool;
    PyObject *ets;
    int n, tool_used = 0;

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
    PyObject *py_J = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_J);

    // Make sure q is number array
    // Cast to numpy array
    // Get data out
    if (!_check_array_type(py_q))
        return NULL;
    py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
    q = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q);

    // Check if tool is None
    // Make sure tool is number array
    // Cast to numpy array
    // Get data out
    if (py_tool != Py_None)
    {
        if (!_check_array_type(py_tool))
            return NULL;
        tool_used = 1;
        py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
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
    npy_float64 *J, *q, *tool = NULL;
    PyObject *py_q, *py_tool, *py_np_q, *py_np_tool;
    PyObject *ets;
    int n, tool_used = 0;

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
    PyObject *py_J = PyArray_EMPTY(2, dims, NPY_DOUBLE, 0);
    J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_J);

    // Make sure q is number array
    // Cast to numpy array
    // Get data out
    if (!_check_array_type(py_q))
        return NULL;
    py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
    q = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q);

    // Check if tool is None
    // Make sure tool is number array
    // Cast to numpy array
    // Get data out
    if (py_tool != Py_None)
    {
        if (!_check_array_type(py_tool))
            return NULL;
        tool_used = 1;
        py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
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
    int include_base, n = 0, q_nd, trajn = 1, tool_used = 0, base_used = 0;
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
    py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
    q = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q);

    // Check the dimesnions of q
    q_nd = PyArray_NDIM((PyArrayObject *)py_np_q);
    q_shape = PyArray_SHAPE((PyArrayObject *)py_np_q);

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
        py_ret = PyArray_EMPTY(2, dim2, NPY_DOUBLE, 0);
    }
    else
    {
        dim3[0] = trajn;
        py_ret = PyArray_EMPTY(3, dim3, NPY_DOUBLE, 0);
    }

    // Get numpy reference to return array
    ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);

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
            py_np_base = (PyObject *)PyArray_FROMANY(py_base, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
            base = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_base);
        }
    }

    if (py_tool != Py_None)
    {
        if (!_check_array_type(py_tool))
            return NULL;
        tool_used = 1;
        py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
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
    _ET_Alloc(et);
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
    _ET_Alloc(et);
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
    PyObject *py_ret = PyArray_EMPTY(nd, dims, NPY_DOUBLE, 0);
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

    ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);
    _ET_T(et, ret, eta);

    return py_ret;
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

void rx(npy_float64 *data, double eta)
{
    double st, ct;

    ct = cos(eta);
    st = sin(eta);

    // data[0] = 1;
    // data[4] = 0;
    // data[8] = 0;
    // data[12] = 0;
    // data[1] = 0;
    // data[5] = ct;
    // data[9] = -st;
    // data[13] = 0;
    // data[2] = 0;
    // data[6] = st;
    // data[10] = ct;
    // data[14] = 0;
    // data[3] = 0;
    // data[7] = 0;
    // data[11] = 0;
    // data[15] = 1;

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

    // data[0] = ct;
    // data[4] = 0;
    // data[8] = st;
    // data[12] = 0;
    // data[1] = 0;
    // data[5] = 1;
    // data[9] = 0;
    // data[13] = 0;
    // data[2] = -st;
    // data[6] = 0;
    // data[10] = ct;
    // data[14] = 0;
    // data[3] = 0;
    // data[7] = 0;
    // data[11] = 0;
    // data[15] = 1;

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

    // data[0] = ct;
    // data[4] = -st;
    // data[8] = 0;
    // data[12] = 0;
    // data[1] = st;
    // data[5] = ct;
    // data[9] = 0;
    // data[13] = 0;
    // data[2] = 0;
    // data[6] = 0;
    // data[10] = 1;
    // data[14] = 0;
    // data[3] = 0;
    // data[7] = 0;
    // data[11] = 0;
    // data[15] = 1;

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
    // data[0] = 1;
    // data[1] = 0;
    // data[2] = 0;
    // data[12] = eta;
    // data[4] = 0;
    // data[5] = 1;
    // data[6] = 0;
    // data[7] = 0;
    // data[8] = 0;
    // data[9] = 0;
    // data[10] = 1;
    // data[11] = 0;
    // data[3] = 0;
    // data[13] = 0;
    // data[14] = 0;
    // data[15] = 1;

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
    // data[0] = 1;
    // data[1] = 0;
    // data[2] = 0;
    // data[3] = 0;
    // data[4] = 0;
    // data[5] = 1;
    // data[6] = 0;
    // data[13] = eta;
    // data[8] = 0;
    // data[9] = 0;
    // data[10] = 1;
    // data[11] = 0;
    // data[12] = 0;
    // data[7] = 0;
    // data[14] = 0;
    // data[15] = 1;

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
    // data[0] = 1;
    // data[1] = 0;
    // data[2] = 0;
    // data[3] = 0;
    // data[4] = 0;
    // data[5] = 1;
    // data[6] = 0;
    // data[7] = 0;
    // data[8] = 0;
    // data[9] = 0;
    // data[10] = 1;
    // data[14] = eta;
    // data[12] = 0;
    // data[13] = 0;
    // data[11] = 0;
    // data[15] = 1;

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
