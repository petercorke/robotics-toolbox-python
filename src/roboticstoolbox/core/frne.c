/**
 * \file frne.c
 * \author Jesse Haviland
 * \author Peter Corke
 * \brief MEX file body
 *
 *
 *  FRNE
 *
 *  TAU = FRNE(ROBOT*, Q, QD, QDD, GRAV, FEXT)
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

#include <math.h>
#include <Python.h>
#include "frne.h"

// forward defines
static PyObject *init(PyObject *self, PyObject *args);
static PyObject *frne(PyObject *self, PyObject *args);
static PyObject *delete(PyObject *self, PyObject *args);
static void rot_mat (Link *l, double th, double d, DHType type);


// static char helloworld_docs[] =
//    "helloworld( ): Any message you want to put here!!\n";

static PyMethodDef frneMethods[] = {
    {
        "init",
        (PyCFunction)init,
        METH_VARARGS,
        "Create Robot"
    },
    {
        "frne",
        (PyCFunction)frne,
        METH_VARARGS,
        "Fast rne"
    },
    {
        "delete",
        (PyCFunction)delete,
        METH_VARARGS,
        "Delete robot memory"
    },
    {NULL, NULL, 0, NULL} /* Sentinel */
};


static struct PyModuleDef frnemodule =
{
    PyModuleDef_HEAD_INIT,
    "frne", 
    "Fast RNE",
    -1,
    frneMethods
};

PyMODINIT_FUNC PyInit_frne(void)
{
    return PyModule_Create(&frnemodule);
}


static PyObject *delete(PyObject *self, PyObject *args) {

    Robot *robot;
    PyObject *obj;

    if (!PyArg_ParseTuple(args, "O", &obj)) {
        return NULL;
    }

    if (!(robot = (Robot*) PyCapsule_GetPointer(obj, "Robot"))) {
        return NULL;
    }

    for (int i = 0; i < robot->njoints; i++) {
        PyMem_RawFree(robot->links[i].I);
        PyMem_RawFree(robot->links[i].Tc);
        PyMem_RawFree(robot->links[i].rbar);
    }

    PyMem_RawFree(robot->gravity);
    PyMem_RawFree(robot->links);
    PyMem_RawFree(robot);
    return Py_BuildValue("i", 1);
}


static PyObject *frne(PyObject *self, PyObject *args) {

    Robot *robot;
    PyObject *rO, *qO, *qdO, *qddO, *gravO, *fextO, *temp;
    // PyObject *python_float;
    double  *q, *qd, *qdd, *fext;
    // Vect *grav;
    int nq = 1; //, nqd = njoints, nqdd = njoints;
    int njoints;

    if (!PyArg_ParseTuple(args, "OOOOOO", &rO, &qO, &qdO, &qddO, &gravO, &fextO)) {
        return NULL;
    }

    if (!(robot = (Robot*) PyCapsule_GetPointer(rO, "Robot"))) {
        return NULL;
    }

    njoints = robot->njoints;

    // Allocate memory for joints
    q = (double *)PyMem_RawCalloc(njoints, sizeof(double));
    qd = (double *)PyMem_RawCalloc(njoints, sizeof(double));
    qdd = (double *)PyMem_RawCalloc(njoints, sizeof(double));
    fext = (double *)PyMem_RawCalloc(6, sizeof(double));

    // Create iterators for arrays
    PyObject *iq = PyObject_GetIter(qO);
    PyObject *iqd = PyObject_GetIter(qdO);
    PyObject *iqdd = PyObject_GetIter(qddO);
    PyObject *igrav = PyObject_GetIter(gravO);
    PyObject *ifext = PyObject_GetIter(fextO);

    // Create the gravity vector
    temp = PyIter_Next(igrav);
    robot->gravity->x = PyFloat_AsDouble(temp);
    Py_DECREF(temp);

    temp = PyIter_Next(igrav);
    robot->gravity->y = PyFloat_AsDouble(temp);
    Py_DECREF(temp);

    temp = PyIter_Next(igrav);
    robot->gravity->z = PyFloat_AsDouble(temp);
    Py_DECREF(temp);


    // Create the joint arrays
    for (int i = 0; i < njoints; i++) {
        temp = PyIter_Next(iq);
        q[i] = PyFloat_AsDouble(temp);
        Py_DECREF(temp);

        temp = PyIter_Next(iqd);
        qd[i] = PyFloat_AsDouble(temp);
        Py_DECREF(temp);

        temp = PyIter_Next(iqdd);
        qdd[i] = PyFloat_AsDouble(temp);
        Py_DECREF(temp);
    }

    // Create the fext array
    for (int i = 0; i < 6; i++) {
        temp = PyIter_Next(ifext);
        fext[i] = PyFloat_AsDouble(temp);
        Py_DECREF(temp);
    }

    // Create a matrix for the return argument */
    double  *tau;
    tau = (double *)PyMem_RawCalloc(njoints, sizeof(double));

    #define MEL(x,R,C)  (x[(R)+(C)*nq])

    // // For each point in the input trajectory
    // for (int p = 0; p < nq; p++) {
    int p = 0;

    // Update all position dependent variables
    for (int j = 0; j < njoints; j++) {
        Link *l = &robot->links[j];

        switch (l->jointtype) {
        case REVOLUTE:
            rot_mat(l, MEL(q,p,j)+l->offset, l->D, robot->dhtype);
            break;
        case PRISMATIC:
            rot_mat(l, l->theta, MEL(q,p,j)+l->offset, robot->dhtype);
            break;
        default:
            perror("Invalid joint type %d (expecting 'R' or 'P')");
        }
    }

    newton_euler(robot, tau, qd, qdd, fext, nq);

    PyMem_RawFree(q);
    PyMem_RawFree(qd);
    PyMem_RawFree(qdd);
    PyMem_RawFree(fext);

    Py_DECREF(iq);
    Py_DECREF(iqd);
    Py_DECREF(iqdd);
    Py_DECREF(igrav);
    Py_DECREF(ifext);

    PyObject* ret = PyList_New(njoints);
    for (int i = 0; i < njoints; ++i) {
        PyList_SetItem(ret, i, Py_BuildValue("d", tau[i]));
    }

    PyMem_RawFree(tau);

    return ret;
}


static PyObject *init(PyObject *self, PyObject *args) {

    Robot *robot;
    PyObject *L, *gravity;
    PyObject *ret;
    int njoints, mdh;

    if (!PyArg_ParseTuple(args, "iiOO", &njoints, &mdh, &L, &gravity)) {
        return NULL;
    }

    // Allocate memory for the robot
    robot = (Robot *)PyMem_RawMalloc(sizeof(Robot));

    // Fill out the robot structure
    robot->njoints = njoints;

    // Get MDH flag
    robot->dhtype = (DHType)mdh;

    // Build link structure
    robot->links = (Link *)PyMem_RawCalloc(njoints, sizeof(Link));

    // Create iterators for arrays
    PyObject *iter_L = PyObject_GetIter(L);
    PyObject *iter_grav = PyObject_GetIter(gravity);

    // Create the gravity vector
    robot->gravity = (Vect *)PyMem_RawMalloc(sizeof(Vect));
    robot->gravity->x = PyFloat_AsDouble(PyIter_Next(iter_grav));
    robot->gravity->y = PyFloat_AsDouble(PyIter_Next(iter_grav));
    robot->gravity->z = PyFloat_AsDouble(PyIter_Next(iter_grav));

    for (int i = 0; i < njoints; i++) {

        Link    *l = &robot->links[i];

        // Allocate memory for Vectors
        l->rbar = (Vect *)PyMem_RawMalloc(sizeof(Vect));
        l->I = (double *)PyMem_RawCalloc(9, sizeof(double));
        l->Tc = (double *)PyMem_RawCalloc(2, sizeof(double));

        l->alpha =  PyFloat_AsDouble(PyIter_Next(iter_L));
        l->A =      PyFloat_AsDouble(PyIter_Next(iter_L));
        l->theta =  PyFloat_AsDouble(PyIter_Next(iter_L));
        l->D =      PyFloat_AsDouble(PyIter_Next(iter_L));
        l->jointtype =  (DHType)PyFloat_AsDouble(PyIter_Next(iter_L));
        l->offset = PyFloat_AsDouble(PyIter_Next(iter_L));
        l->m =      PyFloat_AsDouble(PyIter_Next(iter_L));
        l->rbar->x =   PyFloat_AsDouble(PyIter_Next(iter_L));
        l->rbar->y =   PyFloat_AsDouble(PyIter_Next(iter_L));
        l->rbar->z =   PyFloat_AsDouble(PyIter_Next(iter_L));

        for (int j = 0; j < 9; j++) {
            l->I[j] =      PyFloat_AsDouble(PyIter_Next(iter_L));
        }

        l->Jm =     PyFloat_AsDouble(PyIter_Next(iter_L));
        l->G =      PyFloat_AsDouble(PyIter_Next(iter_L));
        l->B =      PyFloat_AsDouble(PyIter_Next(iter_L));
        l->Tc[0] =     PyFloat_AsDouble(PyIter_Next(iter_L));
        l->Tc[1] =     PyFloat_AsDouble(PyIter_Next(iter_L));
    }

    ret = PyCapsule_New(robot, "Robot", NULL);
    return ret;
}


/**
 * Return the link rotation matrix and translation vector.
 *
 * @param l Link object for which R and p* are required.
 * @param th Joint angle, overrides value in link object
 * @param d Link extension, overrides value in link object
 * @param type Kinematic convention.
 */
static void
rot_mat (
    Link    *l,
    double  th,
    double  d,
    DHType  type
) {
    double      st, ct, sa, ca;

#ifdef  sun
    sincos(th, &st, &ct);
    sincos(l->alpha, &sa, &ca);
#else
    st = sin(th);
    ct = cos(th);
    sa = sin(l->alpha);
    ca = cos(l->alpha);
#endif

    switch (type) {
case STANDARD:
    l->R.n.x = ct;      l->R.o.x = -ca*st;  l->R.a.x = sa*st;
    l->R.n.y = st;      l->R.o.y = ca*ct;   l->R.a.y = -sa*ct;
    l->R.n.z = 0.0;     l->R.o.z = sa;      l->R.a.z = ca;

    l->r.x = l->A;
    l->r.y = d * sa;
    l->r.z = d * ca;
    break;
case MODIFIED:
    l->R.n.x = ct;      l->R.o.x = -st;     l->R.a.x = 0.0;
    l->R.n.y = st*ca;   l->R.o.y = ca*ct;   l->R.a.y = -sa;
    l->R.n.z = st*sa;   l->R.o.z = ct*sa;   l->R.a.z = ca;

    l->r.x = l->A;
    l->r.y = -d * sa;
    l->r.z = d * ca;
    break;
default:
     perror("Invalid DH type (expecting 0 = DH or 1 = MDH)");
    }
}
