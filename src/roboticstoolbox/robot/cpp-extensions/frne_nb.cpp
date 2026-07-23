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
#include <nanobind/stl/pair.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

extern "C" {
#include "frne.h"   // Robot, Link, Vect, Rot, DHType, newton_euler
}

#include <algorithm>
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
           nb::ndarray<double> L_arr) -> std::unique_ptr<RobotObj>
        {
            auto obj = std::make_unique<RobotObj>();

            Robot *robot = (Robot *)PyMem_RawMalloc(sizeof(Robot));
            obj->ptr = robot;

            robot->njoints = njoints;
            robot->dhtype  = (DHType)mdh;
            robot->links   = (Link *)PyMem_RawCalloc(njoints, sizeof(Link));
            // gravity is set per-call in frne() (it's cheap and lets callers
            // override it per-call, same as before) -- zero-init here only
            // so the pointer is never read uninitialized.
            robot->gravity = (Vect *)PyMem_RawCalloc(1, sizeof(Vect));

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
    // q_arr/qd_arr/qdd_arr are (trajn, njoints) -- a full trajectory, looped
    // over entirely in C++ rather than once per Python->C call (previously
    // DHRobot.rne() called this once per row; ~35% of wall time on a 1000-row
    // trajectory was that Python-side looping/marshalling overhead, not the
    // C computation itself -- see rne.md issue 5/plan step 7). gravity,
    // base_rot and fext are constant across the whole trajectory, matching
    // what the old per-row Python loop already passed every iteration.
    m.def("frne",
        [](RobotObj *obj,
           nb::ndarray<double> q_arr,
           nb::ndarray<double> qd_arr,
           nb::ndarray<double> qdd_arr,
           nb::ndarray<double> grav_arr,
           nb::ndarray<double> base_rot_arr,
           nb::ndarray<double> fext_arr)
           -> std::pair<std::vector<double>, std::vector<double>>
        {
            Robot *robot = obj->ptr;
            int nj = robot->njoints;
            int trajn = (int)q_arr.shape(0);

            const double *q    = (const double *)q_arr.data();
            const double *qd   = (const double *)qd_arr.data();
            const double *qdd  = (const double *)qdd_arr.data();
            const double *grav = (const double *)grav_arr.data();
            const double *fext = (const double *)fext_arr.data();

            // gravity is given in the world frame; ne.c's recursion has
            // always implicitly expected it pre-rotated into the root link
            // frame and negated to the effective upward acceleration
            // (previously done by hand in DHRobot.rne()/_copy_to_cpp(),
            // duplicated and easy to get wrong -- see tech-debt.md /
            // rne.md). base_rot_arr is self.base.R, row-major flattened;
            // Rot's n/o/a are its columns. Computed once -- constant across
            // the trajectory.
            const double *br = (const double *)base_rot_arr.data();
            Rot base_R;
            base_R.n = {br[0], br[3], br[6]};
            base_R.o = {br[1], br[4], br[7]};
            base_R.a = {br[2], br[5], br[8]};

            Vect grav_world = {grav[0], grav[1], grav[2]};
            Vect grav_root;
            rot_trans_vect_mult(&grav_root, &base_R, &grav_world);
            robot->gravity->x = -grav_root.x;
            robot->gravity->y = -grav_root.y;
            robot->gravity->z = -grav_root.z;

            std::vector<double> fext_buf(fext, fext + 6);
            std::vector<double> tau(trajn * nj, 0.0);
            std::vector<double> wbase(trajn * 6, 0.0);

            // Per-row temporaries, reused across the trajectory rather than
            // reallocated each iteration.
            std::vector<double> qd_buf(nj), qdd_buf(nj), tau_row(nj);

            for (int i = 0; i < trajn; i++) {
                const double *qi   = q   + i * nj;
                const double *qdi  = qd  + i * nj;
                const double *qddi = qdd + i * nj;

                std::copy(qdi,  qdi  + nj, qd_buf.begin());
                std::copy(qddi, qddi + nj, qdd_buf.begin());

                for (int j = 0; j < nj; j++) {
                    Link *l = &robot->links[j];
                    switch (l->jointtype) {
                    case REVOLUTE:
                        rot_mat(l, qi[j] + l->offset, l->D, robot->dhtype);
                        break;
                    case PRISMATIC:
                        rot_mat(l, l->theta, qi[j] + l->offset, robot->dhtype);
                        break;
                    default:
                        throw nb::value_error("Invalid joint type (expect R or P)");
                    }
                }

                newton_euler(robot, tau_row.data(), qd_buf.data(), qdd_buf.data(),
                             fext_buf.data(), 1);
                std::copy(tau_row.begin(), tau_row.end(), tau.begin() + i * nj);

                // Base wrench: f(0)/n(0) are already computed by the backward
                // recursion above (force/moment on link 0 due to the base) --
                // just not previously exposed. Rotate out of link 0's own
                // frame into the base/world frame, matching rne_python's
                // wbase convention (Rm[0] @ f, Rm[0] @ n).
                Link *l0 = &robot->links[0];
                Vect f_base, n_base;
                rot_vect_mult(&f_base, &l0->R, &l0->f);
                rot_vect_mult(&n_base, &l0->R, &l0->n);
                wbase[i * 6 + 0] = f_base.x;
                wbase[i * 6 + 1] = f_base.y;
                wbase[i * 6 + 2] = f_base.z;
                wbase[i * 6 + 3] = n_base.x;
                wbase[i * 6 + 4] = n_base.y;
                wbase[i * 6 + 5] = n_base.z;
            }

            return {tau, wbase};
        });

    // ── delete ────────────────────────────────────────────────────────────────
    // Kept for facade compatibility. Sets ptr=nullptr so the destructor is a
    // no-op, giving the caller explicit "free now" semantics.
    m.def("delete",
        [](RobotObj *obj) { obj->free_robot(); });
}
