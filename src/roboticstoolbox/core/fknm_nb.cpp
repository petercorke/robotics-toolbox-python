/**
 * fknm_nb.cpp — nanobind port of fknm.cpp
 *
 * Core maths (methods.cpp, ik.cpp, linalg.cpp) unchanged.
 * Python glue rewritten: NB_MODULE + nb::class_ replace PyMethodDef /
 * PyModuleDef / PyCapsule; typed function parameters replace PyArg_ParseTuple.
 * ET and ETS handles are now proper C++ objects with destructors — no more
 * memory leaks from PyCapsule(NULL destructor).
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include "methods.h"
#include "ik.h"
#include "linalg.h"
#include "structs.h"

#include <math.h>
#include <Eigen/Dense>
#include <string.h>

namespace nb = nanobind;
// unconstrained: used for ET_init T (borrows caller's buffer long-term)
using darray = nb::ndarray<double>;
// C-contiguous: q vectors — raw pointer access assumes stride-1
using dcarray = nb::ndarray<double, nb::c_contig>;
using dnumpy  = nb::ndarray<nb::numpy, double>;
// Eigen::Ref accepts any layout; nanobind reads strides from the ndarray
// Dynamic strides: accepts C-contiguous, F-contiguous, or any-stride 4×4 input.
// Eigen copies element-by-element using the actual numpy strides, so no
// explicit Python-side layout conversion is needed before calling C++.
using EigenRef4d  = Eigen::Ref<const Eigen::Matrix4d, 0,
                        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;
using EigenRefVXd = Eigen::Ref<const Eigen::VectorXd>;
// Dynamic-stride ref for any-layout 6×n Jacobian (may be C- or F-contiguous)
using EigenRefJd  = Eigen::Ref<const Eigen::MatrixXd, 0,
                        Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

// ── axis transform functions ────────────────────────────────────────────────

// the transforms are stored in column-major order which is most efficient for Eigen::Map

static void rx(double *d, double e) {
    double ct = cos(e), st = sin(e);
    d[0]=1;  d[4]=0;  d[8]=0;   d[12]=0;
    d[1]=0;  d[5]=ct; d[9]=-st; d[13]=0;
    d[2]=0;  d[6]=st; d[10]=ct; d[14]=0;
    d[3]=0;  d[7]=0;  d[11]=0;  d[15]=1;
}
static void ry(double *d, double e) {
    double ct = cos(e), st = sin(e);
    d[0]=ct;  d[4]=0; d[8]=st;  d[12]=0;
    d[1]=0;   d[5]=1; d[9]=0;   d[13]=0;
    d[2]=-st; d[6]=0; d[10]=ct; d[14]=0;
    d[3]=0;   d[7]=0; d[11]=0;  d[15]=1;
}
static void rz(double *d, double e) {
    double ct = cos(e), st = sin(e);
    d[0]=ct; d[4]=-st; d[8]=0;  d[12]=0;
    d[1]=st; d[5]=ct;  d[9]=0;  d[13]=0;
    d[2]=0;  d[6]=0;   d[10]=1; d[14]=0;
    d[3]=0;  d[7]=0;   d[11]=0; d[15]=1;
}
static void tx(double *d, double e) {
    d[0]=1; d[4]=0; d[8]=0;  d[12]=e;
    d[1]=0; d[5]=1; d[9]=0;  d[13]=0;
    d[2]=0; d[6]=0; d[10]=1; d[14]=0;
    d[3]=0; d[7]=0; d[11]=0; d[15]=1;
}
static void ty(double *d, double e) {
    d[0]=1; d[4]=0; d[8]=0;  d[12]=0;
    d[1]=0; d[5]=1; d[9]=0;  d[13]=e;
    d[2]=0; d[6]=0; d[10]=1; d[14]=0;
    d[3]=0; d[7]=0; d[11]=0; d[15]=1;
}
static void tz(double *d, double e) {
    d[0]=1; d[4]=0; d[8]=0;  d[12]=0;
    d[1]=0; d[5]=1; d[9]=0;  d[13]=0;
    d[2]=0; d[6]=0; d[10]=1; d[14]=e;
    d[3]=0; d[7]=0; d[11]=0; d[15]=1;
}
static void assign_op(ET *et, int axis) {
    switch (axis) {
        case 0: et->op = rx; break;
        case 1: et->op = ry; break;
        case 2: et->op = rz; break;
        case 3: et->op = tx; break;
        case 4: et->op = ty; break;
        default: et->op = tz; break;
    }
}

// ── owner handles (proper destructors fix the PyCapsule memory leaks) ───────

struct ETObj {
    ET *ptr = nullptr;
    ~ETObj() {
        if (ptr) { PyMem_RawFree(ptr->qlim); PyMem_RawFree(ptr); }
    }
};

struct ETSObj {
    ETS *ptr = nullptr;
    ~ETSObj() {
        if (!ptr) return;
        PyMem_RawFree(ptr->ets);
        PyMem_RawFree(ptr->qlim_l);
        PyMem_RawFree(ptr->qlim_h);
        PyMem_RawFree(ptr->q_range2);
        PyMem_RawFree(ptr);
    }
};

// ── output-array helpers (avoid numpy C API for allocation) ─────────────────

static dnumpy make_1d(int n) {
    auto *d = new double[n]();
    nb::capsule own(d, [](void *p) noexcept { delete[] (double *)p; });
    size_t sh[1] = {(size_t)n};
    return dnumpy(d, 1, sh, own);
}

// F-contiguous (column-major) 2D array
// nanobind ndarray strides are in elements, not bytes
static dnumpy make_2d_F(int rows, int cols) {
    auto *d = new double[rows * cols]();
    nb::capsule own(d, [](void *p) noexcept { delete[] (double *)p; });
    size_t sh[2] = {(size_t)rows, (size_t)cols};
    int64_t st[2] = {1, (int64_t)rows};
    return dnumpy(d, 2, sh, own, st);
}

// C-contiguous (row-major) 3D array
static dnumpy make_3d_C(int a, int b, int c) {
    auto *d = new double[a * b * c]();
    nb::capsule own(d, [](void *p) noexcept { delete[] (double *)p; });
    size_t sh[3] = {(size_t)a, (size_t)b, (size_t)c};
    return dnumpy(d, 3, sh, own);
}

// ── IK helper: extract optional array into Eigen Map ────────────────────────

static MapVectorX opt_to_map(const std::optional<darray> &opt, int n,
                             std::vector<double> &buf)
{
    if (!opt)
        return MapVectorX(nullptr, 0);
    const auto &arr = *opt;
    const double *src = (const double *)arr.data();
    buf.assign(src, src + n);
    return MapVectorX(buf.data(), n);
}

// ── IK return helper ─────────────────────────────────────────────────────────

static std::tuple<dnumpy, int, int, int, double>
ik_result(int n, double *ret_data, nb::capsule own, int solution, int it,
          int search, double E)
{
    size_t sh[1] = {(size_t)n};
    dnumpy arr(ret_data, 1, sh, own);
    return {arr, solution, it, search, E};
}

// ── module ───────────────────────────────────────────────────────────────────

NB_MODULE(_fknm_c, m) {
    m.doc() = "Fast kinematics, Jacobian, Hessian, IK (nanobind port)";

    nb::class_<ETObj>(m, "_ETObj")
        .def("__repr__", [](const ETObj &) { return "<ETObj>"; });

    nb::class_<ETSObj>(m, "_ETSObj")
        .def("__repr__", [](const ETSObj &) { return "<ETSObj>"; });

    // ── ET_init ─────────────────────────────────────────────────────────────
    m.def("ET_init",
        [](int isstaticsym, int isjoint, int isflip, int jindex,
           int axis_type, darray T, darray qlim) -> std::unique_ptr<ETObj>
        {
            auto obj = std::make_unique<ETObj>();
            ET *et = (ET *)PyMem_RawMalloc(sizeof(ET));
            obj->ptr = et;

            et->isstaticsym = isstaticsym;
            et->isjoint     = isjoint;
            et->isflip      = isflip;
            et->jindex      = jindex;
            et->axis        = axis_type;

            et->qlim = (double *)PyMem_RawMalloc(2 * sizeof(double));
            const double *ql = (const double *)qlim.data();
            et->qlim[0] = ql[0];
            et->qlim[1] = ql[1];

            et->T = (double *)T.data();
            new (&et->Tm) MapMatrix4dc(et->T);

            assign_op(et, axis_type);
            return obj;
        });

    // ── ET_update ────────────────────────────────────────────────────────────
    m.def("ET_update",
        [](ETObj *obj, bool isstaticsym, bool isjoint, bool isflip, int jindex,
           int axis_type, darray T, darray qlim)
        {
            ET *et = obj->ptr;
            const double *ql = (const double *)qlim.data();
            et->qlim[0] = ql[0];
            et->qlim[1] = ql[1];

            et->T = (double *)T.data();
            new (&et->Tm) MapMatrix4dc(et->T);

            et->axis        = axis_type;
            et->isjoint     = (int)isjoint;
            et->isflip      = (int)isflip;
            et->jindex      = jindex;
            et->isstaticsym = (int)isstaticsym;
            assign_op(et, axis_type);
        });

    // ── ET_T ─────────────────────────────────────────────────────────────────
    m.def("ET_T",
        [](ETObj *obj, nb::object eta_obj) -> dnumpy
        {
            ET *et = obj->ptr;
            if (et->isstaticsym)
                throw nb::type_error("Symbolic value");

            double eta = 0.0;
            if (!eta_obj.is_none()) {
                if (!PyFloat_Check(eta_obj.ptr()) && !PyLong_Check(eta_obj.ptr()))
                    throw nb::type_error("Symbolic value");
                eta = nb::cast<double>(eta_obj);
            }

            auto ret = make_2d_F(4, 4);
            _ET_T(et, (double *)ret.data(), eta);
            return ret;
        });

    // ── ETS_init ─────────────────────────────────────────────────────────────
    m.def("ETS_init",
        [](std::vector<ETObj *> et_list, int n, int m_val)
            -> std::unique_ptr<ETSObj>
        {
            auto obj = std::make_unique<ETSObj>();
            ETS *ets = (ETS *)PyMem_RawMalloc(sizeof(ETS));
            obj->ptr = ets;

            ets->n   = n;
            ets->m   = m_val;
            ets->ets = (ET **)PyMem_RawMalloc(m_val * sizeof(ET *));
            for (int i = 0; i < m_val; i++)
                ets->ets[i] = et_list[i]->ptr;

            ets->qlim_l   = (double *)PyMem_RawMalloc(n * sizeof(double));
            ets->qlim_h   = (double *)PyMem_RawMalloc(n * sizeof(double));
            ets->q_range2 = (double *)PyMem_RawMalloc(n * sizeof(double));

            int j = 0;
            for (int i = 0; i < m_val; i++) {
                ET *et = ets->ets[i];
                if (et->isjoint) {
                    ets->qlim_l[j]   = et->qlim[0];
                    ets->qlim_h[j]   = et->qlim[1];
                    ets->q_range2[j] = (et->qlim[1] - et->qlim[0]) / 2.0;
                    j++;
                }
            }
            return obj;
        });

    // ── ETS_fkine ────────────────────────────────────────────────────────────
    m.def("ETS_fkine",
        [](ETSObj *ets_obj, dcarray q_arr,
           std::optional<EigenRef4d> base_opt,
           std::optional<EigenRef4d> tool_opt,
           int include_base) -> dnumpy
        {
            ETS *ets = ets_obj->ptr;
            double *q = (double *)q_arr.data();

            // Copy base / tool to local column-major storage so _ETS_fkine
            // receives a plain double* with column-major layout.
            Matrix4dc base_mat, tool_mat;
            double *base = nullptr, *tool = nullptr;
            if (base_opt && include_base) { base_mat = *base_opt; base = &base_mat(0); }
            if (tool_opt)                 { tool_mat = *tool_opt; tool = &tool_mat(0); }

            // Determine trajectory length
            int q_ndim = (int)q_arr.ndim();
            int n_q, trajn = 1;
            if (q_ndim == 1) {
                n_q = (int)q_arr.shape(0);
            } else {
                size_t r = q_arr.shape(0), c = q_arr.shape(1);
                if (r == 1)      { trajn = 1;   n_q = (int)c; }
                else if (c == 1) { trajn = 1;   n_q = (int)r; }
                else             { trajn = (int)r; n_q = (int)c; }
            }

            if (trajn == 1) {
                auto ret = make_2d_F(4, 4);
                MapMatrix4dc e_ret((double *)ret.data());
                _ETS_fkine(ets, q, base, tool, e_ret);
                return ret;
            } else {
                // Return (trajn, 4, 4) C-contiguous — each 4×4 is transposed
                auto ret = make_3d_C(trajn, 4, 4);
                double *ret_data = (double *)ret.data();
                for (int i = 0; i < trajn; i++) {
                    double *retp = ret_data + 16 * i;
                    MapMatrix4dc e_retp(retp);
                    _ETS_fkine(ets, q + n_q * i, base, tool, e_retp);
                    e_retp.transposeInPlace();
                }
                return ret;
            }
        });

    // ── ETS_jacob0 ───────────────────────────────────────────────────────────
    m.def("ETS_jacob0",
        [](ETSObj *ets_obj, dcarray q_arr, std::optional<EigenRef4d> tool_opt)
            -> dnumpy
        {
            ETS *ets = ets_obj->ptr;
            double *q = (double *)q_arr.data();
            Matrix4dc tool_mat;
            double *tool = nullptr;
            if (tool_opt) { tool_mat = *tool_opt; tool = &tool_mat(0); }

            auto J = make_2d_F(6, ets->n);
            MapMatrixJc eJ((double *)J.data(), 6, ets->n);
            _ETS_jacob0(ets, q, tool, eJ);
            return J;
        });

    // ── ETS_jacobe ───────────────────────────────────────────────────────────
    m.def("ETS_jacobe",
        [](ETSObj *ets_obj, dcarray q_arr, std::optional<EigenRef4d> tool_opt)
            -> dnumpy
        {
            ETS *ets = ets_obj->ptr;
            double *q = (double *)q_arr.data();
            Matrix4dc tool_mat;
            double *tool = nullptr;
            if (tool_opt) { tool_mat = *tool_opt; tool = &tool_mat(0); }

            auto J = make_2d_F(6, ets->n);
            MapMatrixJc eJ((double *)J.data(), 6, ets->n);
            _ETS_jacobe(ets, q, tool, eJ);
            return J;
        });

    // ── ETS_hessian0 / ETS_hessiane ──────────────────────────────────────────
    auto make_hessian = [](ETSObj *ets_obj, std::optional<dcarray> q_opt,
                            std::optional<EigenRefJd> J_opt,
                            std::optional<EigenRef4d> tool_opt,
                            bool use_jacob0) -> dnumpy
    {
        ETS *ets = ets_obj->ptr;
        int n = ets->n;

        Matrix4dc tool_mat;
        double *tool = nullptr;
        if (tool_opt) { tool_mat = *tool_opt; tool = &tool_mat(0); }

        std::vector<double> J_buf(6 * n, 0.0);
        MapMatrixJc eJ(J_buf.data(), 6, n);

        if (J_opt) {
            // Copy through Eigen assignment — handles any stride (C- or F-contiguous)
            eJ = *J_opt;
        } else {
            double *q = (double *)q_opt->data();
            if (use_jacob0)
                _ETS_jacob0(ets, q, tool, eJ);
            else
                _ETS_jacobe(ets, q, tool, eJ);
        }

        auto H = make_3d_C(n, 6, n);
        MapMatrixHr eH((double *)H.data(), n * 6, n);
        _ETS_hessian(n, eJ, eH);
        return H;
    };

    m.def("ETS_hessian0",
        [make_hessian](ETSObj *ets_obj, std::optional<dcarray> q,
                       std::optional<EigenRefJd> J, std::optional<EigenRef4d> tool) {
            return make_hessian(ets_obj, q, J, tool, true);
        });

    m.def("ETS_hessiane",
        [make_hessian](ETSObj *ets_obj, std::optional<dcarray> q,
                       std::optional<EigenRefJd> J, std::optional<EigenRef4d> tool) {
            return make_hessian(ets_obj, q, J, tool, false);
        });

    // ── Angle_Axis ───────────────────────────────────────────────────────────
    m.def("Angle_Axis",
        [](EigenRef4d Te_ref, EigenRef4d Tep_ref) -> dnumpy
        {
            Matrix4dc Te  = Te_ref;
            Matrix4dc Tep = Tep_ref;
            MapMatrix4dc map_Te(&Te(0));

            auto ret = make_1d(6);
            MapVectorX e((double *)ret.data(), 6);
            _angle_axis(map_Te, Tep, e);
            return ret;
        });

    // ── Robot_link_T ─────────────────────────────────────────────────────────
    m.def("Robot_link_T",
        [](std::vector<ETSObj *> ets_list, nb::list T_list,
           darray self_q, std::optional<darray> q_opt)
        {
            double *q = q_opt ? (double *)q_opt->data()
                               : (double *)self_q.data();

            for (size_t i = 0; i < ets_list.size(); i++) {
                ETS *ets = ets_list[i]->ptr;
                auto T_arr = nb::cast<darray>(T_list[i]);
                MapMatrix4dc eT((double *)T_arr.data());
                _ETS_fkine(ets, q, nullptr, nullptr, eT);
            }
        });

    // ── r2q ──────────────────────────────────────────────────────────────────
    m.def("r2q",
        [](darray r_arr, darray q_arr)
        {
            _r2q((double *)r_arr.data(), (double *)q_arr.data());
        });

    // ── IK_GN_c ──────────────────────────────────────────────────────────────
    m.def("IK_GN_c",
        [](ETSObj *ets_obj, EigenRef4d Tep_ref, std::optional<darray> q0_opt,
           int ilimit, int slimit, double tol, int reject_jl,
           std::optional<darray> we_opt, int use_pinv, double pinv_damping)
            -> std::tuple<dnumpy, int, int, int, double>
        {
            ETS *ets = ets_obj->ptr;
            int it = 0, search = 1, solution = 0;
            double E = 0;

            Matrix4dc Tep = Tep_ref;

            auto *ret_data = new double[ets->n]();
            nb::capsule own(ret_data, [](void *p) noexcept { delete[] (double *)p; });
            MapVectorX ret(ret_data, ets->n);

            std::vector<double> q0_buf, we_buf;
            MapVectorX q0 = opt_to_map(q0_opt, ets->n, q0_buf);
            MapVectorX we = opt_to_map(we_opt, 6,      we_buf);

            _IK_GN(ets, Tep, q0, ilimit, slimit, tol, reject_jl,
                   ret, &it, &search, &solution, &E, we, use_pinv, pinv_damping);

            return ik_result(ets->n, ret_data, own, solution, it, search, E);
        });

    // ── IK_NR_c ──────────────────────────────────────────────────────────────
    m.def("IK_NR_c",
        [](ETSObj *ets_obj, EigenRef4d Tep_ref, std::optional<darray> q0_opt,
           int ilimit, int slimit, double tol, int reject_jl,
           std::optional<darray> we_opt, int use_pinv, double pinv_damping)
            -> std::tuple<dnumpy, int, int, int, double>
        {
            ETS *ets = ets_obj->ptr;
            int it = 0, search = 1, solution = 0;
            double E = 0;

            Matrix4dc Tep = Tep_ref;

            auto *ret_data = new double[ets->n]();
            nb::capsule own(ret_data, [](void *p) noexcept { delete[] (double *)p; });
            MapVectorX ret(ret_data, ets->n);

            std::vector<double> q0_buf, we_buf;
            MapVectorX q0 = opt_to_map(q0_opt, ets->n, q0_buf);
            MapVectorX we = opt_to_map(we_opt, 6,      we_buf);

            _IK_NR(ets, Tep, q0, ilimit, slimit, tol, reject_jl,
                   ret, &it, &search, &solution, &E, we, use_pinv, pinv_damping);

            return ik_result(ets->n, ret_data, own, solution, it, search, E);
        });

    // ── IK_LM_c ──────────────────────────────────────────────────────────────
    m.def("IK_LM_c",
        [](ETSObj *ets_obj, EigenRef4d Tep_ref, std::optional<darray> q0_opt,
           int ilimit, int slimit, double tol, int reject_jl,
           std::optional<darray> we_opt, double lambda, std::string method)
            -> std::tuple<dnumpy, int, int, int, double>
        {
            ETS *ets = ets_obj->ptr;
            int it = 0, search = 1, solution = 0;
            double E = 0;

            Matrix4dc Tep = Tep_ref;

            auto *ret_data = new double[ets->n]();
            nb::capsule own(ret_data, [](void *p) noexcept { delete[] (double *)p; });
            MapVectorX ret(ret_data, ets->n);

            std::vector<double> q0_buf, we_buf;
            MapVectorX q0 = opt_to_map(q0_opt, ets->n, q0_buf);
            MapVectorX we = opt_to_map(we_opt, 6,      we_buf);

            if (!method.empty() && method[0] == 's')
                _IK_LM_Sugihara(ets, Tep, q0, ilimit, slimit, tol, reject_jl,
                                 ret, &it, &search, &solution, &E, lambda, we);
            else if (!method.empty() && method[0] == 'w')
                _IK_LM_Wampler(ets, Tep, q0, ilimit, slimit, tol, reject_jl,
                                ret, &it, &search, &solution, &E, lambda, we);
            else
                _IK_LM_Chan(ets, Tep, q0, ilimit, slimit, tol, reject_jl,
                             ret, &it, &search, &solution, &E, lambda, we);

            return ik_result(ets->n, ret_data, own, solution, it, search, E);
        });
}
