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
static PyObject *fkine_all(PyObject *self, PyObject *args);
static PyObject *jacob0(PyObject *self, PyObject *args);
static PyObject *jacobe(PyObject *self, PyObject *args);
static PyObject *fkine(PyObject *self, PyObject *args);
static PyObject *link_init(PyObject *self, PyObject *args);
static PyObject *link_A(PyObject *self, PyObject *args);
static PyObject *link_update(PyObject *self, PyObject *args);
static PyObject *compose(PyObject *self, PyObject *args);

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
int _inv(npy_float64 *m, npy_float64 *invOut);

static PyMethodDef fknmMethods[] = {
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
        }
    }

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

    PyObject *py_shape_base, *py_shape_wT, *py_shape_sT;
    PyObject *iter_base, *iter_wT, *iter_sT;
    PyArrayObject *pys_base, *pys_wT, *pys_sT;
    PyArrayObject *py_A, *py_fk;

    link = (Link *)PyMem_RawMalloc(sizeof(Link));

    if (!PyArg_ParseTuple(args, "iiiiiO!O!OOOO",
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

    link->shape_base = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));
    link->shape_wT = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));
    link->shape_sT = (npy_float64 **)PyMem_RawCalloc(link->n_shapes, sizeof(npy_float64));

    for (int i = 0; i < link->n_shapes; i++)
    {
        if (
            !(pys_base = (PyArrayObject *)PyIter_Next(iter_base)) ||
            !(pys_wT = (PyArrayObject *)PyIter_Next(iter_wT)) ||
            !(pys_sT = (PyArrayObject *)PyIter_Next(iter_sT)))
            return NULL;

        link->shape_base[i] = (npy_float64 *)PyArray_DATA(pys_base);
        link->shape_wT[i] = (npy_float64 *)PyArray_DATA(pys_wT);
        link->shape_sT[i] = (npy_float64 *)PyArray_DATA(pys_sT);
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

    PyObject *py_shape_base, *py_shape_wT, *py_shape_sT;
    PyObject *iter_base, *iter_wT, *iter_sT;
    PyArrayObject *pys_base, *pys_wT, *pys_sT;

    if (!PyArg_ParseTuple(args, "OiiiiiO!O!OOOO",
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

    link->shape_base = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));
    link->shape_wT = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));
    link->shape_sT = (npy_float64 **)PyMem_RawCalloc(n_shapes, sizeof(npy_float64));

    for (int i = 0; i < n_shapes; i++)
    {
        if (
            !(pys_base = (PyArrayObject *)PyIter_Next(iter_base)) ||
            !(pys_wT = (PyArrayObject *)PyIter_Next(iter_wT)) ||
            !(pys_sT = (PyArrayObject *)PyIter_Next(iter_sT)))
            return NULL;

        link->shape_base[i] = (npy_float64 *)PyArray_DATA(pys_base);
        link->shape_wT[i] = (npy_float64 *)PyArray_DATA(pys_wT);
        link->shape_sT[i] = (npy_float64 *)PyArray_DATA(pys_sT);
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

void _jacobe(PyObject *links, int m, int n, npy_float64 *q, npy_float64 *etool, npy_float64 *tool, npy_float64 *J)
{
    Link *link;
    npy_float64 *T = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *U = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *temp = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *ret = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    npy_float64 *invU = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
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
    copy(temp, current);
    mult(current, tool, ret);
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

int _inv(npy_float64 *m, npy_float64 *invOut)
{
    npy_float64 *inv = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    double det;
    int i;

    inv[0] = m[5] * m[10] * m[15] -
             m[5] * m[11] * m[14] -
             m[9] * m[6] * m[15] +
             m[9] * m[7] * m[14] +
             m[13] * m[6] * m[11] -
             m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] +
             m[4] * m[11] * m[14] +
             m[8] * m[6] * m[15] -
             m[8] * m[7] * m[14] -
             m[12] * m[6] * m[11] +
             m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] -
             m[4] * m[11] * m[13] -
             m[8] * m[5] * m[15] +
             m[8] * m[7] * m[13] +
             m[12] * m[5] * m[11] -
             m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] +
              m[4] * m[10] * m[13] +
              m[8] * m[5] * m[14] -
              m[8] * m[6] * m[13] -
              m[12] * m[5] * m[10] +
              m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] +
             m[1] * m[11] * m[14] +
             m[9] * m[2] * m[15] -
             m[9] * m[3] * m[14] -
             m[13] * m[2] * m[11] +
             m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] -
             m[0] * m[11] * m[14] -
             m[8] * m[2] * m[15] +
             m[8] * m[3] * m[14] +
             m[12] * m[2] * m[11] -
             m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] +
             m[0] * m[11] * m[13] +
             m[8] * m[1] * m[15] -
             m[8] * m[3] * m[13] -
             m[12] * m[1] * m[11] +
             m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] -
              m[0] * m[10] * m[13] -
              m[8] * m[1] * m[14] +
              m[8] * m[2] * m[13] +
              m[12] * m[1] * m[10] -
              m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] -
             m[1] * m[7] * m[14] -
             m[5] * m[2] * m[15] +
             m[5] * m[3] * m[14] +
             m[13] * m[2] * m[7] -
             m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] +
             m[0] * m[7] * m[14] +
             m[4] * m[2] * m[15] -
             m[4] * m[3] * m[14] -
             m[12] * m[2] * m[7] +
             m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] -
              m[0] * m[7] * m[13] -
              m[4] * m[1] * m[15] +
              m[4] * m[3] * m[13] +
              m[12] * m[1] * m[7] -
              m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] +
              m[0] * m[6] * m[13] +
              m[4] * m[1] * m[14] -
              m[4] * m[2] * m[13] -
              m[12] * m[1] * m[6] +
              m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] +
             m[1] * m[7] * m[10] +
             m[5] * m[2] * m[11] -
             m[5] * m[3] * m[10] -
             m[9] * m[2] * m[7] +
             m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] -
             m[0] * m[7] * m[10] -
             m[4] * m[2] * m[11] +
             m[4] * m[3] * m[10] +
             m[8] * m[2] * m[7] -
             m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] +
              m[0] * m[7] * m[9] +
              m[4] * m[1] * m[11] -
              m[4] * m[3] * m[9] -
              m[8] * m[1] * m[7] +
              m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] -
              m[0] * m[6] * m[9] -
              m[4] * m[1] * m[10] +
              m[4] * m[2] * m[9] +
              m[8] * m[1] * m[6] -
              m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    if (det == 0)
        return 0;

    det = 1.0 / det;

    for (i = 0; i < 16; i++)
        invOut[i] = inv[i] * det;

    return 1;
}
