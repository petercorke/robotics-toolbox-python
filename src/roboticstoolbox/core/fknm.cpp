/**
 * \file fknm.cpp
 * \author Jesse Haviland
 *
 *
 */

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "fknm.h"
#include "methods.h"
#include "ik.h"
#include "linalg.h"
#include "structs.h"

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <Eigen/Dense>
#include <iostream>
#include <string.h>

static PyMethodDef fknmMethods[] = {
    {"Angle_Axis",
     (PyCFunction)Angle_Axis,
     METH_VARARGS,
     "Link"},
    {"IK_GN_c",
     (PyCFunction)IK_GN_c,
     METH_VARARGS,
     "Link"},
    {"IK_NR_c",
     (PyCFunction)IK_NR_c,
     METH_VARARGS,
     "Link"},
    {"IK_LM_c",
     (PyCFunction)IK_LM_c,
     METH_VARARGS,
     "Link"},
    // {"IK_LM_Wampler_c",
    //  (PyCFunction)IK_LM_Wampler_c,
    //  METH_VARARGS,
    //  "Link"},
    // {"IK_LM_Sugihara_c",
    //  (PyCFunction)IK_LM_Sugihara_c,
    //  METH_VARARGS,
    //  "Link"},
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
    {"ETS_init",
     (PyCFunction)ETS_init,
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

extern "C"
{

    static PyObject *Angle_Axis(PyObject *self, PyObject *args)
    {
        npy_float64 *np_Te, *np_Tep, *np_ret;
        PyObject *py_Te, *py_Tep;
        PyArrayObject *py_np_Te, *py_np_Tep;

        if (!PyArg_ParseTuple(
                args, "OO",
                &py_Te,
                &py_Tep))
            return NULL;

        // Inputs can be:
        // Te, Tep can be SE3s or 4x4 numpy array

        // Make sure Te, Tep is number array
        // Cast to numpy array
        // Get data out
        if (!_check_array_type(py_Te))
            return NULL;
        py_np_Te = (PyArrayObject *)PyArray_FROMANY(py_Te, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_Te = (npy_float64 *)PyArray_DATA(py_np_Te);

        if (!_check_array_type(py_Tep))
            return NULL;
        py_np_Tep = (PyArrayObject *)PyArray_FROMANY(py_Tep, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_Tep = (npy_float64 *)PyArray_DATA(py_np_Tep);

        // Make our empty error vector
        npy_intp dims[1] = {6};
        PyObject *py_ret = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
        np_ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);
        MapVectorX ret(np_ret, 6);

        // Get eigen matrices
        // Tep in row major from Python
        MapMatrix4dr row_Tep(np_Tep);
        MapMatrix4dr row_Te(np_Te);

        // Convert to col major here
        Matrix4dc Tep = row_Tep;
        Matrix4dc Te = row_Te;

        // Get map matrix
        MapMatrix4dc map_Te(&Te(0));

        // Do the job
        _angle_axis(map_Te, Tep, ret);

        return py_ret;
    }

    static PyObject *IK_GN_c(PyObject *self, PyObject *args)
    {
        ETS *ets;
        npy_float64 *np_Tep, *np_ret, *np_q0, *np_we;
        PyArrayObject *py_np_Tep;
        PyObject *py_ets, *py_ret, *py_Tep, *py_q0, *py_np_q0, *py_we, *py_np_we;
        PyObject *py_tup, *py_it, *py_search, *py_solution, *py_E;
        npy_intp dim[1] = {1};
        int ilimit, slimit, q0_used = 0, we_used = 0, reject_jl, use_pinv;
        double tol, E, pinv_damping;

        int it = 0, search = 1, solution = 0;

        if (!PyArg_ParseTuple(
                args, "OOOiidiOid",
                &py_ets,
                &py_Tep,
                &py_q0,
                &ilimit,
                &slimit,
                &tol,
                &reject_jl,
                &py_we,
                &use_pinv,
                &pinv_damping))
            return NULL;

        if (!_check_array_type(py_Tep))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        // Assign empty q0 and we
        MapVectorX q0(NULL, 0);
        MapVectorX we(NULL, 0);

        // Check if q0 is None
        if (py_q0 != Py_None)
        {
            // Make sure q is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_q0))
                return NULL;
            q0_used = 1;
            py_np_q0 = (PyObject *)PyArray_FROMANY(py_q0, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_q0 = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q0);
            // MapVectorX q0(np_q0, ets->n);
            new (&q0) MapVectorX(np_q0, ets->n);
        }

        // Check if we is None
        if (py_we != Py_None)
        {
            // Make sure we is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_we))
                return NULL;
            we_used = 1;
            py_np_we = (PyObject *)PyArray_FROMANY(py_we, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_we = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_we);
            new (&we) MapVectorX(np_we, 6);
        }

        // Set the dimension of the returned array to match the number of joints
        dim[0] = ets->n;

        py_np_Tep = (PyArrayObject *)PyArray_FROMANY(py_Tep, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_Tep = (npy_float64 *)PyArray_DATA(py_np_Tep);

        // Tep in row major from Python
        MapMatrix4dr row_Tep(np_Tep);

        // Convert to col major here
        Matrix4dc Tep = row_Tep;

        py_ret = PyArray_EMPTY(1, dim, NPY_DOUBLE, 0);
        np_ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);
        MapVectorX ret(np_ret, ets->n);

        _IK_GN(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, we, use_pinv, pinv_damping);

        // Free the memory
        Py_DECREF(py_np_Tep);

        if (q0_used)
        {
            Py_DECREF(py_np_q0);
        }

        if (we_used)
        {
            Py_DECREF(py_np_we);
        }

        // Build the return tuple
        py_it = Py_BuildValue("i", it);
        py_search = Py_BuildValue("i", search);
        py_solution = Py_BuildValue("i", solution);
        py_E = Py_BuildValue("d", E);

        py_tup = PyTuple_Pack(5, py_ret, py_solution, py_it, py_search, py_E);

        Py_DECREF(py_it);
        Py_DECREF(py_search);
        Py_DECREF(py_solution);
        Py_DECREF(py_E);
        Py_DECREF(py_ret);

        return py_tup;
    }

    static PyObject *IK_NR_c(PyObject *self, PyObject *args)
    {
        ETS *ets;
        npy_float64 *np_Tep, *np_ret, *np_q0, *np_we;
        PyArrayObject *py_np_Tep;
        PyObject *py_ets, *py_ret, *py_Tep, *py_q0, *py_np_q0, *py_we, *py_np_we;
        PyObject *py_tup, *py_it, *py_search, *py_solution, *py_E;
        npy_intp dim[1] = {1};
        int ilimit, slimit, q0_used = 0, we_used = 0, reject_jl, use_pinv;
        double tol, E, pinv_damping;

        int it = 0, search = 1, solution = 0;

        if (!PyArg_ParseTuple(
                args, "OOOiidiOid",
                &py_ets,
                &py_Tep,
                &py_q0,
                &ilimit,
                &slimit,
                &tol,
                &reject_jl,
                &py_we,
                &use_pinv,
                &pinv_damping))
            return NULL;

        if (!_check_array_type(py_Tep))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        // Assign empty q0 and we
        MapVectorX q0(NULL, 0);
        MapVectorX we(NULL, 0);

        // Check if q0 is None
        if (py_q0 != Py_None)
        {
            // Make sure q is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_q0))
                return NULL;
            q0_used = 1;
            py_np_q0 = (PyObject *)PyArray_FROMANY(py_q0, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_q0 = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q0);
            // MapVectorX q0(np_q0, ets->n);
            new (&q0) MapVectorX(np_q0, ets->n);
        }

        // Check if we is None
        if (py_we != Py_None)
        {
            // Make sure we is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_we))
                return NULL;
            we_used = 1;
            py_np_we = (PyObject *)PyArray_FROMANY(py_we, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_we = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_we);
            new (&we) MapVectorX(np_we, 6);
        }

        // Set the dimension of the returned array to match the number of joints
        dim[0] = ets->n;

        py_np_Tep = (PyArrayObject *)PyArray_FROMANY(py_Tep, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_Tep = (npy_float64 *)PyArray_DATA(py_np_Tep);

        // Tep in row major from Python
        MapMatrix4dr row_Tep(np_Tep);

        // Convert to col major here
        Matrix4dc Tep = row_Tep;

        py_ret = PyArray_EMPTY(1, dim, NPY_DOUBLE, 0);
        np_ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);
        MapVectorX ret(np_ret, ets->n);

        _IK_NR(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, we, use_pinv, pinv_damping);

        // Free the memory
        Py_DECREF(py_np_Tep);

        if (q0_used)
        {
            Py_DECREF(py_np_q0);
        }

        if (we_used)
        {
            Py_DECREF(py_np_we);
        }

        // Build the return tuple
        py_it = Py_BuildValue("i", it);
        py_search = Py_BuildValue("i", search);
        py_solution = Py_BuildValue("i", solution);
        py_E = Py_BuildValue("d", E);

        py_tup = PyTuple_Pack(5, py_ret, py_solution, py_it, py_search, py_E);

        Py_DECREF(py_it);
        Py_DECREF(py_search);
        Py_DECREF(py_solution);
        Py_DECREF(py_E);
        Py_DECREF(py_ret);

        return py_tup;
    }

    static PyObject *IK_LM_c(PyObject *self, PyObject *args)
    {
        ETS *ets;
        npy_float64 *np_Tep, *np_ret, *np_q0, *np_we;
        PyArrayObject *py_np_Tep;
        PyObject *py_ets, *py_ret, *py_Tep, *py_q0, *py_np_q0, *py_we, *py_np_we;
        PyObject *py_tup, *py_it, *py_search, *py_solution, *py_E;
        npy_intp dim[1] = {1};
        int ilimit, slimit, q0_used = 0, we_used = 0, reject_jl;
        double tol, E, lambda;
        const char *method;

        int it = 0, search = 1, solution = 0;

        if (!PyArg_ParseTuple(
                args, "OOOiidiOds",
                &py_ets,
                &py_Tep,
                &py_q0,
                &ilimit,
                &slimit,
                &tol,
                &reject_jl,
                &py_we,
                &lambda,
                &method))
            return NULL;

        if (!_check_array_type(py_Tep))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        // Assign empty q0 and we
        MapVectorX q0(NULL, 0);
        MapVectorX we(NULL, 0);

        // Check if q0 is None
        if (py_q0 != Py_None)
        {
            // Make sure q is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_q0))
                return NULL;
            q0_used = 1;
            py_np_q0 = (PyObject *)PyArray_FROMANY(py_q0, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_q0 = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q0);
            // MapVectorX q0(np_q0, ets->n);
            new (&q0) MapVectorX(np_q0, ets->n);
        }

        // Check if we is None
        if (py_we != Py_None)
        {
            // Make sure we is number array
            // Cast to numpy array
            // Get data out
            if (!_check_array_type(py_we))
                return NULL;
            we_used = 1;
            py_np_we = (PyObject *)PyArray_FROMANY(py_we, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            np_we = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_we);
            new (&we) MapVectorX(np_we, 6);
        }

        // Set the dimension of the returned array to match the number of joints
        dim[0] = ets->n;

        py_np_Tep = (PyArrayObject *)PyArray_FROMANY(py_Tep, NPY_DOUBLE, 1, 2, NPY_ARRAY_DEFAULT);
        np_Tep = (npy_float64 *)PyArray_DATA(py_np_Tep);

        // Tep in row major from Python
        MapMatrix4dr row_Tep(np_Tep);

        // Convert to col major here
        Matrix4dc Tep = row_Tep;

        py_ret = PyArray_EMPTY(1, dim, NPY_DOUBLE, 0);
        np_ret = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_ret);
        MapVectorX ret(np_ret, ets->n);

        // std::cout << Tep << std::endl;
        // std::cout << ret << std::endl;

        if (method[0] == 's')
        {
            // std::cout << "sugi" << std::endl;
            _IK_LM_Sugihara(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, lambda, we);
        }
        else if (method[0] == 'w')
        {
            // std::cout << "wampl" << std::endl;
            _IK_LM_Wampler(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, lambda, we);
        }
        else
        {
            // std::cout << "chan" << std::endl;
            _IK_LM_Chan(ets, Tep, q0, ilimit, slimit, tol, reject_jl, ret, &it, &search, &solution, &E, lambda, we);
        }

        // Free the memory
        Py_DECREF(py_np_Tep);

        if (q0_used)
        {
            Py_DECREF(py_np_q0);
        }

        if (we_used)
        {
            Py_DECREF(py_np_we);
        }

        // Build the return tuple
        py_it = Py_BuildValue("i", it);
        py_search = Py_BuildValue("i", search);
        py_solution = Py_BuildValue("i", solution);
        py_E = Py_BuildValue("d", E);

        py_tup = PyTuple_Pack(5, py_ret, py_solution, py_it, py_search, py_E);

        Py_DECREF(py_it);
        Py_DECREF(py_search);
        Py_DECREF(py_solution);
        Py_DECREF(py_E);
        Py_DECREF(py_ret);

        return py_tup;
    }

    static PyObject *Robot_link_T(PyObject *self, PyObject *args)
    {
        ETS *ets;
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
            PyObject *py_ets = PyList_GET_ITEM(ets_list, i);
            // Extract the ETS object from the python object
            if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
                return NULL;

            npy_float64 *T = (npy_float64 *)PyArray_DATA((PyArrayObject *)PyList_GET_ITEM(T_list, i));
            MapMatrix4dc eT(T);

            // TODO Add this back
            _ETS_fkine(ets, q, NULL, NULL, eT);
        }

        // Free the memory
        if (q_used)
        {
            Py_DECREF(py_np_q);
        }

        Py_RETURN_NONE;
    }

    static PyObject *ETS_hessian0(PyObject *self, PyObject *args)
    {
        ETS *ets;
        npy_float64 *H, *J, *q, *tool = NULL;
        PyObject *py_q, *py_J, *py_tool, *py_np_q, *py_np_tool, *py_np_J;
        PyObject *py_ets;
        int tool_used = 0, J_used = 0, q_used = 0;

        if (!PyArg_ParseTuple(
                args, "OOOO",
                &py_ets,
                &py_q,
                &py_J,
                &py_tool))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        MapMatrixJc eJ(NULL, 6, ets->n);

        // Check if J is None
        // Make sure J is number array
        // Cast to numpy array
        // Get data out
        if (py_J != Py_None)
        {
            if (!_check_array_type(py_J))
                return NULL;
            J_used = 1;
            py_np_J = (PyObject *)PyArray_FROMANY(py_J, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_J);
            // MapMatrixJc eJ(J, 6, ets->n);
            new (&eJ) MapMatrixJc(J, 6, ets->n);
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
            py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            q = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q);

            // Make our empty Jacobian
            npy_intp dimsJ[2] = {6, ets->n};
            PyObject *py_J = PyArray_EMPTY(2, dimsJ, NPY_DOUBLE, 1);
            J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_J);
            // MapMatrixJc eJ(J, 6, ets->n);
            new (&eJ) MapMatrixJc(J, 6, ets->n);

            // Check if tool is None
            // Make sure tool is number array
            // Cast to numpy array
            // Get data out
            if (py_tool != Py_None)
            {
                if (!_check_array_type(py_tool))
                    return NULL;
                tool_used = 1;
                py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
                tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
            }

            // Calculate the Jacobian
            _ETS_jacob0(ets, q, tool, eJ);
        }

        // Make our empty Hessian
        npy_intp dimsH[3] = {ets->n, 6, ets->n};
        PyObject *py_H = PyArray_EMPTY(3, dimsH, NPY_DOUBLE, 0);
        H = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_H);
        MapMatrixHr eH(H, ets->n * 6, ets->n);

        // Do the job
        _ETS_hessian(ets->n, eJ, eH);

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
        ETS *ets;
        npy_float64 *H, *J, *q, *tool = NULL;
        PyObject *py_q, *py_J, *py_tool, *py_np_q, *py_np_tool, *py_np_J;
        PyObject *py_ets;
        int tool_used = 0, J_used = 0, q_used = 0;

        if (!PyArg_ParseTuple(
                args, "OOOO",
                &py_ets,
                &py_q,
                &py_J,
                &py_tool))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        MapMatrixJc eJ(NULL, 6, ets->n);

        // Check if J is None
        // Make sure J is number array
        // Cast to numpy array
        // Get data out
        if (py_J != Py_None)
        {
            if (!_check_array_type(py_J))
                return NULL;
            J_used = 1;
            py_np_J = (PyObject *)PyArray_FROMANY(py_J, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_J);
            // MapMatrixJc eJ(J, 6, ets->n);
            new (&eJ) MapMatrixJc(J, 6, ets->n);
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
            py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            q = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q);

            // Make our empty Jacobian
            npy_intp dimsJ[2] = {6, ets->n};
            PyObject *py_J = PyArray_EMPTY(2, dimsJ, NPY_DOUBLE, 1);
            J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_J);
            // MapMatrixJc eJ(J, 6, ets->n);
            new (&eJ) MapMatrixJc(J, 6, ets->n);

            // Check if tool is None
            // Make sure tool is number array
            // Cast to numpy array
            // Get data out
            if (py_tool != Py_None)
            {
                if (!_check_array_type(py_tool))
                    return NULL;
                tool_used = 1;
                py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
                tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
            }

            // Calculate the Jacobian
            _ETS_jacobe(ets, q, tool, eJ);
        }

        // Make our empty Hessian
        npy_intp dimsH[3] = {ets->n, 6, ets->n};
        PyObject *py_H = PyArray_EMPTY(3, dimsH, NPY_DOUBLE, 0);
        H = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_H);
        MapMatrixHr eH(H, ets->n * 6, ets->n);

        // Do the job
        _ETS_hessian(ets->n, eJ, eH);

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
        ETS *ets;
        npy_float64 *J, *q, *tool = NULL;
        PyObject *py_q, *py_tool, *py_np_q, *py_np_tool;
        PyObject *py_ets;
        int tool_used = 0;

        if (!PyArg_ParseTuple(
                args, "OOO",
                &py_ets,
                &py_q,
                &py_tool))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        // Inputs can be:
        // None - Even q
        // Not arrays - Will raise exception
        // Have symbolic data - Will raise exception
        // q can be 1D or 2D, assumes dimesnions correct (n, 1xn or nx1)
        // tool can be SE3s or 4x4 numpy array

        // Make our empty Jacobian
        npy_intp dims[2] = {6, ets->n};
        PyObject *py_J = PyArray_EMPTY(2, dims, NPY_DOUBLE, 1);
        J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_J);
        MapMatrixJc eJ(J, 6, ets->n);

        // Make sure q is number array
        // Cast to numpy array
        // Get data out
        if (!_check_array_type(py_q))
            return NULL;
        py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
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
            py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
        }

        // Do the job
        _ETS_jacob0(ets, q, tool, eJ);

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
        ETS *ets;
        npy_float64 *J, *q, *tool = NULL;
        PyObject *py_q, *py_tool, *py_np_q, *py_np_tool;
        PyObject *py_ets;
        int tool_used = 0;

        if (!PyArg_ParseTuple(
                args, "OOO",
                &py_ets,
                &py_q,
                &py_tool))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
            return NULL;

        // Inputs can be:
        // None - Even q
        // Not arrays - Will raise exception
        // Have symbolic data - Will raise exception
        // q can be 1D or 2D, assumes dimesnions correct (n, 1xn or nx1)
        // tool can be SE3s or 4x4 numpy array

        // Make our empty Jacobian
        npy_intp dims[2] = {6, ets->n};
        PyObject *py_J = PyArray_EMPTY(2, dims, NPY_DOUBLE, 1);
        J = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_J);
        MapMatrixJc eJ(J, 6, ets->n);

        // Make sure q is number array
        // Cast to numpy array
        // Get data out
        if (!_check_array_type(py_q))
            return NULL;
        py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
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
            py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
        }

        // Do the job
        // for (int i = 0; i < 1000000; i++)
        // {
        //     _ETS_jacobe(ets, q, tool, eJ);
        // }
        _ETS_jacobe(ets, q, tool, eJ);

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
        ETS *ets;
        npy_intp dim2[2] = {4, 4}, dim3[3] = {1, 4, 4};
        int include_base, n = 0, q_nd, trajn = 1, tool_used = 0, base_used = 0;
        npy_float64 *ret, *retp, *q, *qp, *base = NULL, *tool = NULL;
        PyObject *py_q, *py_base, *py_tool, *py_np_q, *py_np_tool, *py_np_base;
        PyObject *py_ret, *py_ets;
        npy_intp *q_shape;

        if (!PyArg_ParseTuple(
                args, "OOOOi",
                &py_ets,
                &py_q,
                &py_base,
                &py_tool,
                &include_base))
            return NULL;

        // Extract the ETS object from the python object
        if (!(ets = (ETS *)PyCapsule_GetPointer(py_ets, "ETS")))
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
        py_np_q = (PyObject *)PyArray_FROMANY(py_q, NPY_DOUBLE, 1, 2, NPY_ARRAY_C_CONTIGUOUS);
        q = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_q);

        // std::cout << "q: " << q[0] << ", " << q[1] << ", " << q[2] << ", " << q[3] << ", " << q[4] << ", " << q[5] << std::endl;

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
            py_ret = PyArray_EMPTY(2, dim2, NPY_DOUBLE, 1);
        }
        else
        {
            // if using a trajectory, make a duplicate of ret as we will need to
            // extreme reshape it due to Fortran ordering
            // Fortran ordering of 3D array wants (4, 4, n) while numpy looping
            // typically likes to have (n, 4, 4)

            // therefore we make the returned python array (n, 4, 4) and row-major
            // and later on we transpose each (4, 4) component
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
                py_np_base = (PyObject *)PyArray_FROMANY(py_base, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
                base = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_base);
            }
        }

        if (py_tool != Py_None)
        {
            if (!_check_array_type(py_tool))
                return NULL;
            tool_used = 1;
            py_np_tool = (PyObject *)PyArray_FROMANY(py_tool, NPY_DOUBLE, 1, 2, NPY_ARRAY_F_CONTIGUOUS);
            tool = (npy_float64 *)PyArray_DATA((PyArrayObject *)py_np_tool);
        }

        // Do the actual job
        for (int i = 0; i < trajn; i++)
        {
            retp = ret + (4 * 4 * i);

            MapMatrix4dc e_retp(retp);
            qp = q + (n * i);
            _ETS_fkine(ets, qp, base, tool, e_retp);

            // Transpose if we have a trajectory
            // as the returned trajectory is row-major
            if (trajn > 1)
            {
                e_retp.transposeInPlace();
            }
        }

        // Free memory
        Py_DECREF(py_np_q);

        if (tool_used)
            Py_DECREF(py_np_tool);

        if (base_used)
            Py_DECREF(py_np_base);

        return py_ret;
    }

    static PyObject *ETS_init(PyObject *self, PyObject *args)
    {
        ET *et;
        ETS *ets;
        PyObject *etsl, *ret;
        int j = 0;

        ets = (ETS *)PyMem_RawMalloc(sizeof(ETS));

        if (!PyArg_ParseTuple(args, "Oii",
                              &etsl,
                              &ets->n,
                              &ets->m))
            return NULL;

        ets->ets = (ET **)PyMem_RawMalloc(ets->m * sizeof(ET *));

        PyObject *iter_et = PyObject_GetIter(etsl);

        for (int i = 0; i < ets->m; i++)
        {
            if (!(ets->ets[i] = (ET *)PyCapsule_GetPointer(PyIter_Next(iter_et), "ET")))
                return NULL;
        }

        ets->qlim_l = (double *)PyMem_RawMalloc(ets->n * sizeof(double));
        ets->qlim_h = (double *)PyMem_RawMalloc(ets->n * sizeof(double));
        ets->q_range2 = (double *)PyMem_RawMalloc(ets->n * sizeof(double));

        // Cache joint limits
        for (int i = 0; i < ets->m; i++)
        {
            et = ets->ets[i];

            if (et->isjoint)
            {
                ets->qlim_l[j] = et->qlim[0];
                ets->qlim_h[j] = et->qlim[1];
                ets->q_range2[j] = (et->qlim[1] - et->qlim[0]) / 2.0;

                j += 1;
            }
        }

        Py_DECREF(iter_et);

        ret = PyCapsule_New(ets, "ETS", NULL);
        return ret;
    }

    static PyObject *ET_update(PyObject *self, PyObject *args)
    {
        ET *et;
        int jointtype;
        PyObject *ret, *py_et;
        PyArrayObject *py_T, *py_qlim;
        npy_float64 *np_qlim;
        int isjoint, isflip, jindex;

        et = (ET *)PyMem_RawMalloc(sizeof(ET));

        if (!PyArg_ParseTuple(args, "OiiiiiO!O!",
                              &py_et,
                              &et->isstaticsym,
                              &isjoint,
                              &isflip,
                              &jindex,
                              &jointtype,
                              &PyArray_Type, &py_T,
                              &PyArray_Type, &py_qlim))
            return NULL;

        if (!(et = (ET *)PyCapsule_GetPointer(py_et, "ET")))
            return NULL;

        np_qlim = (npy_float64 *)PyArray_DATA(py_qlim);
        et->qlim[0] = np_qlim[0];
        et->qlim[1] = np_qlim[1];

        et->T = (npy_float64 *)PyArray_DATA(py_T);
        new (&et->Tm) MapMatrix4dc(et->T);
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
        npy_float64 *np_qlim;

        et = (ET *)PyMem_RawMalloc(sizeof(ET));

        if (!PyArg_ParseTuple(args, "iiiiiO!O!",
                              &et->isstaticsym,
                              &et->isjoint,
                              &et->isflip,
                              &et->jindex,
                              &jointtype,
                              &PyArray_Type, &py_T,
                              &PyArray_Type, &py_qlim))
            return NULL;

        np_qlim = (npy_float64 *)PyArray_DATA(py_qlim);
        et->qlim = (double *)PyMem_RawMalloc(2 * sizeof(double));
        et->qlim[0] = np_qlim[0];
        et->qlim[1] = np_qlim[1];

        et->T = (npy_float64 *)PyArray_DATA(py_T);
        new (&et->Tm) MapMatrix4dc(et->T);

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
        PyObject *py_ret = PyArray_EMPTY(nd, dims, NPY_DOUBLE, 1);
        double eta = 0;
        npy_float64 *ret;

        if (!PyArg_ParseTuple(args, "OO", &py_et, &py_eta))
            return NULL;

        if (!(et = (ET *)PyCapsule_GetPointer(py_et, "ET")))
            return NULL;

        if (et->isstaticsym)
        {
            PyErr_SetString(PyExc_TypeError, "Symbolic value");
            return NULL;
        }

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
        // MapMatrix4dc e_ret(ret);

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

        data[0] = 1;
        data[4] = 0;
        data[8] = 0;
        data[12] = 0;
        data[1] = 0;
        data[5] = ct;
        data[9] = -st;
        data[13] = 0;
        data[2] = 0;
        data[6] = st;
        data[10] = ct;
        data[14] = 0;
        data[3] = 0;
        data[7] = 0;
        data[11] = 0;
        data[15] = 1;

        // data[0] = 1;
        // data[1] = 0;
        // data[2] = 0;
        // data[3] = 0;
        // data[4] = 0;
        // data[5] = ct;
        // data[6] = -st;
        // data[7] = 0;
        // data[8] = 0;
        // data[9] = st;
        // data[10] = ct;
        // data[11] = 0;
        // data[12] = 0;
        // data[13] = 0;
        // data[14] = 0;
        // data[15] = 1;
    }

    void ry(npy_float64 *data, double eta)
    {
        double st, ct;

        ct = cos(eta);
        st = sin(eta);

        data[0] = ct;
        data[4] = 0;
        data[8] = st;
        data[12] = 0;
        data[1] = 0;
        data[5] = 1;
        data[9] = 0;
        data[13] = 0;
        data[2] = -st;
        data[6] = 0;
        data[10] = ct;
        data[14] = 0;
        data[3] = 0;
        data[7] = 0;
        data[11] = 0;
        data[15] = 1;

        // data[0] = ct;
        // data[1] = 0;
        // data[2] = st;
        // data[3] = 0;
        // data[4] = 0;
        // data[5] = 1;
        // data[6] = 0;
        // data[7] = 0;
        // data[8] = -st;
        // data[9] = 0;
        // data[10] = ct;
        // data[11] = 0;
        // data[12] = 0;
        // data[13] = 0;
        // data[14] = 0;
        // data[15] = 1;
    }

    void rz(npy_float64 *data, double eta)
    {
        double st, ct;

        ct = cos(eta);
        st = sin(eta);

        data[0] = ct;
        data[4] = -st;
        data[8] = 0;
        data[12] = 0;
        data[1] = st;
        data[5] = ct;
        data[9] = 0;
        data[13] = 0;
        data[2] = 0;
        data[6] = 0;
        data[10] = 1;
        data[14] = 0;
        data[3] = 0;
        data[7] = 0;
        data[11] = 0;
        data[15] = 1;

        // data[0] = ct;
        // data[1] = -st;
        // data[2] = 0;
        // data[3] = 0;
        // data[4] = st;
        // data[5] = ct;
        // data[6] = 0;
        // data[7] = 0;
        // data[8] = 0;
        // data[9] = 0;
        // data[10] = 1;
        // data[11] = 0;
        // data[12] = 0;
        // data[13] = 0;
        // data[14] = 0;
        // data[15] = 1;
    }

    void tx(npy_float64 *data, double eta)
    {
        data[0] = 1;
        data[1] = 0;
        data[2] = 0;
        data[12] = eta;
        data[4] = 0;
        data[5] = 1;
        data[6] = 0;
        data[7] = 0;
        data[8] = 0;
        data[9] = 0;
        data[10] = 1;
        data[11] = 0;
        data[3] = 0;
        data[13] = 0;
        data[14] = 0;
        data[15] = 1;

        // data[0] = 1;
        // data[1] = 0;
        // data[2] = 0;
        // data[3] = eta;
        // data[4] = 0;
        // data[5] = 1;
        // data[6] = 0;
        // data[7] = 0;
        // data[8] = 0;
        // data[9] = 0;
        // data[10] = 1;
        // data[11] = 0;
        // data[12] = 0;
        // data[13] = 0;
        // data[14] = 0;
        // data[15] = 1;
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
        data[13] = eta;
        data[8] = 0;
        data[9] = 0;
        data[10] = 1;
        data[11] = 0;
        data[12] = 0;
        data[7] = 0;
        data[14] = 0;
        data[15] = 1;

        // data[0] = 1;
        // data[1] = 0;
        // data[2] = 0;
        // data[3] = 0;
        // data[4] = 0;
        // data[5] = 1;
        // data[6] = 0;
        // data[7] = eta;
        // data[8] = 0;
        // data[9] = 0;
        // data[10] = 1;
        // data[11] = 0;
        // data[12] = 0;
        // data[13] = 0;
        // data[14] = 0;
        // data[15] = 1;
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
        data[14] = eta;
        data[12] = 0;
        data[13] = 0;
        data[11] = 0;
        data[15] = 1;

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
        // data[11] = eta;
        // data[12] = 0;
        // data[13] = 0;
        // data[14] = 0;
        // data[15] = 1;
    }

} /* extern "C" */