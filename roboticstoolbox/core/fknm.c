/**
 * \file fknm.c
 * \author Jesse Haviland
 * \author Peter Corke
 * \brief MEX file body
 *
 *
 *  FKNM
 *
 *  TAU = FKNM(ROBOT*, Q, QD, QDD, GRAV, FEXT)
 *  ROBOT* = INIT(N, MDH, L, GRAV)
 *
 *  where Q, QD and QDD are row vectors of the manipulator state; pos,
 *  vel, and accel.
 *
 *  Returns the joint torque required to achieve the specified joint 
 *  position, velocity and acceleration state. Gravity is taken
 *  from the robot object.
 *
 *  GRAV overrides the gravity vector in the robot object.
 * 
 *  An external force/moment acting on the end of the manipulator may 
 *  also be specified by a 6-element vector FEXT [Fx Fy Fz Mx My Mz].
 * 
 *
 */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include "fknm.h"

// forward defines
// static PyObject *init(PyObject *self, PyObject *args);
// static PyObject *frne(PyObject *self, PyObject *args);
// static PyObject *delete (PyObject *self, PyObject *args);
// static void rot_mat(Link *l, double th, double d, DHType type);
static PyObject *link_init(PyObject *self, PyObject *args);
static PyObject *link_A(PyObject *self, PyObject *args);
static PyObject *link_update(PyObject *self, PyObject *args);

void mult(npy_float64 *A, npy_float64 *B, npy_float64 *C);
void copy(npy_float64 *A, npy_float64 *B);
void rx(npy_float64 *data, double eta);
void ry(npy_float64 *data, double eta);
void rz(npy_float64 *data, double eta);
void tx(npy_float64 *data, double eta);
void ty(npy_float64 *data, double eta);
void tz(npy_float64 *data, double eta);

static PyMethodDef fknmMethods[] = {
    // {"rx",
    //  (PyCFunction)rx,
    //  METH_VARARGS,
    //  "SE3 Rotation"},
    // {"ry",
    //  (PyCFunction)ry,
    //  METH_VARARGS,
    //  "SE3 Rotation"},
    // {"rz",
    //  (PyCFunction)rz,
    //  METH_VARARGS,
    //  "SE3 Rotation"},
    // {"tx",
    //  (PyCFunction)tx,
    //  METH_VARARGS,
    //  "SE3 Rotation"},
    // {"ty",
    //  (PyCFunction)ty,
    //  METH_VARARGS,
    //  "SE3 Rotation"},
    // {"tz",
    //  (PyCFunction)tz,
    //  METH_VARARGS,
    //  "SE3 Rotation"},
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

static PyObject *link_init(PyObject *self, PyObject *args)
{

    Link *link;
    int jointtype;
    PyObject *ret;

    link = (Link *)PyMem_RawMalloc(sizeof(Link));

    if (!PyArg_ParseTuple(args, "iiiO!", &link->isjoint, &link->isflip, &jointtype, &PyArray_Type, &link->A))
    {
        return NULL;
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

    ret = PyCapsule_New(link, "Link", NULL);
    return ret;
}

static PyObject *link_update(PyObject *self, PyObject *args)
{
    Link *link;
    int isjoint, isflip;
    int jointtype;
    PyObject *lo;
    PyArrayObject *A;

    if (!PyArg_ParseTuple(args, "OiiiO!", &lo, &isjoint, &isflip, &jointtype, &PyArray_Type, &A))
    {
        return NULL;
    }

    if (!(link = (Link *)PyCapsule_GetPointer(lo, "Link")))
    {
        return NULL;
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
    link->A = A;

    Py_RETURN_NONE;
}

static PyObject *link_A(PyObject *self, PyObject *args)
{
    Link *link;
    PyArrayObject *py_ret;
    PyObject *lo;
    npy_float64 *ret, *A, *v;
    double eta;

    if (!PyArg_ParseTuple(args, "dOO!", &eta, &lo, &PyArray_Type, &py_ret))
    {
        return NULL;
    }

    if (!(link = (Link *)PyCapsule_GetPointer(lo, "Link")))
    {
        return NULL;
    }

    ret = (npy_float64 *)PyArray_DATA(py_ret);
    A = (npy_float64 *)PyArray_DATA(link->A);

    if (!link->isjoint)
    {
        copy(A, ret);
        Py_RETURN_NONE;
    }

    if (link->isflip)
    {
        eta = -eta;
    }

    // Calculate the variable part of the link
    v = (npy_float64 *)PyMem_RawCalloc(16, sizeof(npy_float64));
    link->op(v, eta);

    // Multiply ret = A * v
    mult(A, v, ret);

    Py_RETURN_NONE;
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

/**
 * Return the link rotation matrix and translation vector.
 *
 * @param l Link object for which R and p* are required.
 * @param th Joint angle, overrides value in link object
 * @param d Link extension, overrides value in link object
 * @param type Kinematic convention.
 */
// static void
// rot_mat(
//     Link *l,
//     double th,
//     double d,
//     DHType type)
// {
//     double st, ct, sa, ca;

// #ifdef sun
//     sincos(th, &st, &ct);
//     sincos(l->alpha, &sa, &ca);
// #else
//     st = sin(th);
//     ct = cos(th);
//     sa = sin(l->alpha);
//     ca = cos(l->alpha);
// #endif

//     switch (type)
//     {
//     case STANDARD:
//         l->R.n.x = ct;
//         l->R.o.x = -ca * st;
//         l->R.a.x = sa * st;
//         l->R.n.y = st;
//         l->R.o.y = ca * ct;
//         l->R.a.y = -sa * ct;
//         l->R.n.z = 0.0;
//         l->R.o.z = sa;
//         l->R.a.z = ca;

//         l->r.x = l->A;
//         l->r.y = d * sa;
//         l->r.z = d * ca;
//         break;
//     case MODIFIED:
//         l->R.n.x = ct;
//         l->R.o.x = -st;
//         l->R.a.x = 0.0;
//         l->R.n.y = st * ca;
//         l->R.o.y = ca * ct;
//         l->R.a.y = -sa;
//         l->R.n.z = st * sa;
//         l->R.o.z = ct * sa;
//         l->R.a.z = ca;

//         l->r.x = l->A;
//         l->r.y = -d * sa;
//         l->r.z = d * ca;
//         break;
//     default:
//         perror("Invalid DH type (expecting 0 = DH or 1 = MDH)");
//     }
// }
