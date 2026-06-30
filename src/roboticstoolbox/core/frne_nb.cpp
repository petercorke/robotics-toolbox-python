/**
 * frne_nb.cpp — nanobind port of frne.c
 *
 * Pure-C maths (ne.c, vmath.c) unchanged.
 * Python glue rewritten with nanobind: RobotObj wraps Robot* with a proper
 * destructor so the PyCapsule(NULL destructor) memory leak is fixed.
 *
 * The `delete` export is kept for backward compatibility with the facade
 * (DHRobot.delete_rne calls it explicitly); its destructor is the no-op path.
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

extern "C" {
#include "frne.h"   // Robot, Link, Vect, Rot, DHType, newton_euler
}

#include <math.h>

// ── Robot wrapper ─────────────────────────────────────────────────────────────

struct RobotObj {
    Robot *ptr = nullptr;

    void free_robot() {
        if (!ptr) return;
        for (int i = 0; i < ptr->njoints; i++) {
            PyMem_RawFree(ptr->links[i].I);
            PyMem_RawFree(ptr->links[i].Tc);
            PyMem_RawFree(ptr->links[i].rbar);
        }
        PyMem_RawFree(ptr->gravity);
        PyMem_RawFree(ptr->links);
        PyMem_RawFree(ptr);
        ptr = nullptr;
    }

    ~RobotObj() { free_robot(); }
};

// rot_mat: update link R and r from joint angle/extension (identical to frne.c)
static void rot_mat(Link *l, double th, double d, DHType type) {
    double st = sin(th), ct = cos(th);
    double sa = sin(l->alpha), ca = cos(l->alpha);

    switch (type) {
    case STANDARD:
        l->R.n.x = ct;      l->R.o.x = -ca*st;  l->R.a.x = sa*st;
        l->R.n.y = st;      l->R.o.y =  ca*ct;  l->R.a.y = -sa*ct;
        l->R.n.z = 0.0;     l->R.o.z =  sa;     l->R.a.z = ca;
        l->r.x = l->A;
        l->r.y = d * sa;
        l->r.z = d * ca;
        break;
    case MODIFIED:
        l->R.n.x = ct;      l->R.o.x = -st;     l->R.a.x = 0.0;
        l->R.n.y = st*ca;   l->R.o.y =  ca*ct;  l->R.a.y = -sa;
        l->R.n.z = st*sa;   l->R.o.z =  ct*sa;  l->R.a.z = ca;
        l->r.x = l->A;
        l->r.y = -d * sa;
        l->r.z =  d * ca;
        break;
    default:
        break;
    }
}

// ── module ────────────────────────────────────────────────────────────────────

NB_MODULE(_frne_c, m) {
    m.doc() = "Fast Newton-Euler inverse dynamics (nanobind port)";

    nb::class_<RobotObj>(m, "_RobotObj")
        .def("__repr__", [](const RobotObj &) { return "<RobotObj>"; });

    // ── init ──────────────────────────────────────────────────────────────────
    m.def("init",
        [](int njoints, int mdh,
           nb::ndarray<double> L_arr,
           nb::ndarray<double> grav_arr) -> std::unique_ptr<RobotObj>
        {
            auto obj = std::make_unique<RobotObj>();

            Robot *robot = (Robot *)PyMem_RawMalloc(sizeof(Robot));
            obj->ptr = robot;

            robot->njoints = njoints;
            robot->dhtype  = (DHType)mdh;
            robot->links   = (Link *)PyMem_RawCalloc(njoints, sizeof(Link));
            robot->gravity = (Vect *)PyMem_RawMalloc(sizeof(Vect));

            const double *grav = (const double *)grav_arr.data();
            robot->gravity->x = grav[0];
            robot->gravity->y = grav[1];
            robot->gravity->z = grav[2];

            const double *L = (const double *)L_arr.data();
            int idx = 0;
            for (int i = 0; i < njoints; i++) {
                Link *l = &robot->links[i];
                l->rbar = (Vect *)PyMem_RawMalloc(sizeof(Vect));
                l->I    = (double *)PyMem_RawCalloc(9, sizeof(double));
                l->Tc   = (double *)PyMem_RawCalloc(2, sizeof(double));

                l->alpha        = L[idx++];
                l->A            = L[idx++];
                l->theta        = L[idx++];
                l->D            = L[idx++];
                l->jointtype    = (Sigma)(int)L[idx++];
                l->offset       = L[idx++];
                l->m            = L[idx++];
                l->rbar->x      = L[idx++];
                l->rbar->y      = L[idx++];
                l->rbar->z      = L[idx++];
                for (int j = 0; j < 9; j++) l->I[j] = L[idx++];
                l->Jm  = L[idx++];
                l->G   = L[idx++];
                l->B   = L[idx++];
                l->Tc[0] = L[idx++];
                l->Tc[1] = L[idx++];
            }
            return obj;
        });

    // ── frne ──────────────────────────────────────────────────────────────────
    m.def("frne",
        [](RobotObj *obj,
           nb::ndarray<double> q_arr,
           nb::ndarray<double> qd_arr,
           nb::ndarray<double> qdd_arr,
           nb::ndarray<double> grav_arr,
           nb::ndarray<double> fext_arr) -> std::vector<double>
        {
            Robot *robot = obj->ptr;
            int nj = robot->njoints;

            const double *q    = (const double *)q_arr.data();
            const double *qd   = (const double *)qd_arr.data();
            const double *qdd  = (const double *)qdd_arr.data();
            const double *grav = (const double *)grav_arr.data();
            const double *fext = (const double *)fext_arr.data();

            robot->gravity->x = grav[0];
            robot->gravity->y = grav[1];
            robot->gravity->z = grav[2];

            // Allocate temporaries (newton_euler takes non-const)
            std::vector<double> qd_buf(qd, qd + nj);
            std::vector<double> qdd_buf(qdd, qdd + nj);
            std::vector<double> fext_buf(fext, fext + 6);
            std::vector<double> tau(nj, 0.0);

            for (int j = 0; j < nj; j++) {
                Link *l = &robot->links[j];
                switch (l->jointtype) {
                case REVOLUTE:
                    rot_mat(l, q[j] + l->offset, l->D, robot->dhtype);
                    break;
                case PRISMATIC:
                    rot_mat(l, l->theta, q[j] + l->offset, robot->dhtype);
                    break;
                default:
                    throw nb::value_error("Invalid joint type (expect R or P)");
                }
            }

            newton_euler(robot, tau.data(), qd_buf.data(), qdd_buf.data(),
                         fext_buf.data(), 1);
            return tau;
        });

    // ── delete ────────────────────────────────────────────────────────────────
    // Kept for facade compatibility. Sets ptr=nullptr so the destructor is a
    // no-op, giving the caller explicit "free now" semantics.
    m.def("delete",
        [](RobotObj *obj) { obj->free_robot(); });
}
