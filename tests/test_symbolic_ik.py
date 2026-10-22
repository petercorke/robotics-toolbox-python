"""Tests of ``roboticstoolbox.robot.symbolic_ik``

These were at the end of the module, where CI did not run them.
"""

import pytest
from roboticstoolbox.robot.DHRobot import DHRobot
import numpy as np
import re
import sympy as sp
import time
import warnings
import roboticstoolbox.robot.symbolic_ik as sik
from roboticstoolbox.robot.symbolic_ik import (
    IKSolutions,
    IKStatus,
    PolyRoots,
    _POSE_SYMBOLS,
    _abstract_constants,
    _compile,
    _emit,
    _equation_pool,
    _get_flint,
    _joints_in,
    _linear_form,
    _link_matrices,
    _make_plan,
    _pieper,
    _pose_matrix,
    _sin_cos_symbols,
    _solve_order,
    _symbolize,
    _wrap,
    generate_code,
    ikine_sym,
)


def _poly_angles(coeffs) -> np.ndarray:
    """Joint angles for the real roots of a polynomial in tan(q/2), see PolyRoots."""
    c = np.array(coeffs, dtype=float)
    scale = np.abs(c).max()
    if not scale > 1e-9:
        return np.array([])
    keep = len(c)
    while keep > 1 and abs(c[keep - 1]) <= 1e-12 * scale:
        keep -= 1
    angles = [np.pi] if keep < len(c) else []
    if keep > 1:
        for r in np.roots(c[:keep][::-1]):
            if abs(r.imag) <= 1e-6 * (1 + abs(r.real)):
                angles.append(2 * np.arctan(r.real))
    return np.array(angles)


def _compile_roots(sol: PolyRoots):
    """Compile a PolyRoots solution to ``(symbols, function)``, giving the angles."""
    args = sorted(sol.free_symbols, key=lambda s: s.name)
    func = sp.lambdify(args, list(sol.coeffs), modules="numpy", cse=True)
    return args, lambda *v: _poly_angles(func(*v))


def _numeric_solutions(
    Q: list[list[sp.Expr]],
    values: dict[str, float],
    revolute: list[bool] | None = None,
) -> list[np.ndarray]:
    """
    Evaluate symbolic IK solutions for every combination of branches

    :param values: numeric value of symbols, keyed by symbol *name* since the
        symbols in a model may have been created with different assumptions
    :param revolute: whether each joint is revolute, default all

    Solutions are evaluated in dependency order.  The resulting sin and cos (or
    the joint value for a prismatic joint) are fed to the solutions for the
    joints that depend on them.  A branch that is not real, because it is
    outside the workspace, is dropped.
    """
    n = len(Q)
    revolute = revolute or [True] * n
    order = _solve_order(Q)
    compiled = [
        [_compile_roots(e) if isinstance(e, PolyRoots) else _compile(e) for e in sols]
        for sols in Q
    ]

    def recurse(step: int, vals: dict[str, float], qs: dict[int, float]) -> list[dict]:
        if step == n:
            return [qs]
        j = order[step]
        out = []
        for args, func in compiled[j]:
            for qj in np.atleast_1d(func(*[vals[a.name] for a in args])):
                qj = float(qj)
                if np.isnan(qj):
                    continue
                if revolute[j]:
                    new = {f"S{j + 1}": np.sin(qj), f"C{j + 1}": np.cos(qj)}
                else:
                    new = {f"q{j + 1}": qj}
                out += recurse(step + 1, {**vals, **new}, {**qs, j: qj})
        return out

    return [np.array([x[j] for j in range(n)]) for x in recurse(0, dict(values), {})]


def _pose_values(T: np.ndarray) -> dict[str, float]:
    """Map the elements of a numeric 4x4 pose to the symbolic pose elements."""
    return {s.name: v for s, v in zip(_POSE_SYMBOLS, T[:3, :].flatten())}


def test_planar2():
    import roboticstoolbox as rtb

    robot = rtb.models.DH.Planar2(symbolic=True)
    Q = ikine_sym(robot)

    assert Q is not None
    assert len(Q) == 2
    assert all(len(s) == 2 for s in Q)

    # check against the numeric model
    numeric = rtb.models.DH.Planar2()  # a1 = a2 = 1
    q0 = np.array([0.3, 0.7])
    T = numeric.fkine(q0).A

    values = {"a1": 1.0, "a2": 1.0, **_pose_values(T)}
    sols = _numeric_solutions(Q, values)
    assert len(sols) == 4

    # not every branch combination is a valid solution, since a solution for
    # one joint can be spurious once the others are chosen.  Valid ones reach
    # the target, there are two (elbow up and down), and one is the original
    valid = [qs for qs in sols if np.allclose(numeric.fkine(qs).t[:2], T[:2, 3])]
    assert len(valid) == 2
    assert any(np.allclose(_wrap(qs), q0, atol=1e-6) for qs in valid)


def _elbow_robot() -> DHRobot:
    """3-link arm with a vertical waist, as for the first 3 joints of a Puma."""
    from roboticstoolbox import RevoluteDH

    return DHRobot(
        [
            RevoluteDH(d=0.5, a=0, alpha=np.pi / 2),
            RevoluteDH(a=0.4),
            RevoluteDH(a=0.3),
        ],
        name="elbow",
    )


def test_elbow3():
    robot = _elbow_robot()
    Q = ikine_sym(robot)

    assert Q is not None
    assert len(Q) == 3

    _, values = _symbolize(robot)
    q0 = np.array([0.4, 0.6, -0.9])
    T = robot.fkine(q0).A

    sols = _numeric_solutions(Q, {**values, **_pose_values(T)})

    # every branch combination is not necessarily a valid solution (a
    # solution for one joint can be spurious), but valid ones must reach the
    # target and the original must be found
    valid = [qs for qs in sols if np.allclose(robot.fkine(qs).t, T[:3, 3], atol=1e-6)]
    assert len(valid) >= 2
    assert any(np.allclose(_wrap(qs), q0, atol=1e-6) for qs in valid)


def test_pieper_forms():
    """All forms of the equations are the same constraint T = A1 ... AN."""
    numeric = _elbow_robot()
    robot, values = _symbolize(numeric)
    A = _link_matrices(robot)
    q0 = [0.4, 0.6, -0.9]
    T = numeric.fkine(q0).A

    subs = {s: v for s, v in zip(_POSE_SYMBOLS, T[:3, :].flatten())}
    subs |= {sp.Symbol(k, real=True): v for k, v in values.items()}
    for j, qj in enumerate(q0, 1):
        sj, cj = _sin_cos_symbols(j)
        subs |= {sj: np.sin(qj), cj: np.cos(qj)}

    for nl in range(4):
        for nr in range(4 - nl):
            left, right = _pieper(A, _pose_matrix(), nl, nr)
            residual = np.array((left - right).subs(subs).evalf(), dtype=float)
            np.testing.assert_allclose(residual, 0, atol=1e-12)


def test_bad_n():
    import pytest
    import roboticstoolbox as rtb

    robot = rtb.models.DH.Planar2(symbolic=True)
    with pytest.raises(ValueError):
        ikine_sym(robot, 4)


def _dh_models_6() -> list[str]:
    """Names of the 6-axis standard DH models in ``roboticstoolbox.models.DH``."""
    import inspect

    import roboticstoolbox as rtb

    names = []
    for name, cls in inspect.getmembers(rtb.models.DH, inspect.isclass):
        if issubclass(cls, DHRobot):
            robot = cls()  # type: ignore[call-arg]  # the models take no arguments
            if robot.n == 6 and not any(link.mdh for link in robot.links):
                names.append(name)
    return names


_EXPECTED_SOLVED: set[str] = {
    "IRB140",
    "KR5",
    "Puma560",
    "Stanford",
    "UR3",
    "UR5",
    "UR10",
}


def _wrap_joints(q: np.ndarray, revolute: list[bool]) -> np.ndarray:
    """Wrap the revolute joint angles to (-pi, pi]."""
    q = np.array(q, dtype=float)
    r = np.array(revolute)
    q[r] = _wrap(q[r])
    return q


def _numeric_check(
    robot: DHRobot, Q: list[list[sp.Expr]], trials: int = 2, seed: int = 0
) -> list[int]:
    """
    Check a symbolic IK solution against forward kinematics

    For random joint configurations, compute the pose, evaluate all branches of
    the symbolic solution, and check that some of them reach the pose, and that
    the original configuration is among those that do.

    :returns: number of valid solutions for each trial
    """
    rng = np.random.default_rng(seed)
    _, values = _symbolize(robot)
    revolute = [link.isrevolute for link in robot.links]

    counts = []
    for _ in range(trials):
        q0 = np.where(
            revolute, rng.uniform(-1.5, 1.5, robot.n), rng.uniform(0.2, 0.6, robot.n)
        )
        T = robot.fkine(q0).A
        sols = _numeric_solutions(Q, {**values, **_pose_values(T)}, revolute)

        valid = [q for q in sols if np.abs(robot.fkine(q).A - T).max() < 1e-6]
        assert valid, "no valid solution"
        target = _wrap_joints(q0, revolute)
        assert any(
            np.allclose(_wrap_joints(q, revolute), target, atol=1e-5) for q in valid
        ), "original configuration not found"
        counts.append(len(valid))
    return counts


@pytest.mark.slow
def test_dh_models():
    """
    Solve each 6-axis DH model.  Where a solution is found check its form, and
    check it numerically against forward kinematics.
    """
    import roboticstoolbox as rtb

    for name in _dh_models_6():
        robot = getattr(rtb.models.DH, name)()
        t0 = time.monotonic()
        Q = ikine_sym(robot, max_time=120 if name in _EXPECTED_SOLVED else 15)
        print(f"{name:10s} {'solved' if Q else 'None  '} {time.monotonic() - t0:.1f}s")

        if name in _EXPECTED_SOLVED:
            assert Q is not None, name
        if Q is None:
            continue

        assert len(Q) == 6
        assert all(len(s) > 0 for s in Q)
        # only prismatic joint variables ``qj`` may remain in a solution
        prismatic = {
            f"q{j}" for j, link in enumerate(robot.links, 1) if link.isprismatic
        }
        leftover = {
            s.name
            for sols in Q
            for e in sols
            for s in e.free_symbols
            if re.fullmatch(r"q\d+", s.name)
        }
        assert leftover <= prismatic, (name, leftover)

        counts = _numeric_check(robot, Q)
        print(f"{'':10s} valid solutions per trial: {counts}")


def _run_generated(
    language: str, code: str, T: np.ndarray, name: str = "ikine"
) -> tuple[str, list[np.ndarray]]:
    """
    Run generated code on a pose

    :returns: ``(status, solutions)``, where status is ``OK``, ``SINGULAR`` or
        ``UNREACHABLE``
    """
    if language == "python":
        namespace: dict = {}
        exec(code, namespace)  # noqa: S102
        result = namespace[name](T)
        return result.status.name, [np.array(q) for q in result.solutions]

    import subprocess
    import tempfile
    from pathlib import Path

    names = {"0": "OK", "1": "SINGULAR", "2": "UNREACHABLE"}
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        if language == "c":
            (tmp / "ik.c").write_text(code)
            (tmp / "main.c").write_text(
                """#include <stdio.h>
#include <stdlib.h>
#include "ik.c"
int main(int argc, char **argv) {
    double T[16];
    for (int i = 0; i < 16; ++i) T[i] = atof(argv[i + 1]);
    static double sols[IKINE_MAX_SOLUTIONS][IKINE_DOF];
    int count;
    ik_status st = %s(T, 1e-6, sols, &count);
    printf("%%d\\n", (int)st);
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < IKINE_DOF; ++j) printf("%%.17g ", sols[i][j]);
        printf("\\n");
    }
    return 0;
}
"""
                % name
            )
            compile_cmd = ["cc", "-O1", "-Wall", "-o", "ik", "main.c", "-lm"]
        elif language == "c++":
            (tmp / "ik.hpp").write_text(code)
            (tmp / "main.cpp").write_text(
                """#include <cstdio>
#include <cstdlib>
#include "ik.hpp"
int main(int argc, char **argv) {
    double T[16];
    for (int i = 0; i < 16; ++i) T[i] = std::atof(argv[i + 1]);
    IkResult r = %s(T);
    std::printf("%%d\\n", static_cast<int>(r.status));
    for (const auto &q : r.solutions) {
        for (double v : q) std::printf("%%.17g ", v);
        std::printf("\\n");
    }
    return 0;
}
"""
                % name
            )
            compile_cmd = ["c++", "-std=c++17", "-O1", "-Wall", "-o", "ik", "main.cpp"]
        elif language == "rust":
            (tmp / "ik_mod.rs").write_text(code)
            (tmp / "main.rs").write_text(
                """mod ik_mod;
fn main() {
    let a: Vec<f64> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let mut t = [0.0f64; 16];
    t.copy_from_slice(&a);
    let r = ik_mod::%s(&t, 1e-6);
    println!("{}", r.status as i32);
    for q in &r.solutions {
        let v: Vec<String> = q.iter().map(|x| format!("{:.17e}", x)).collect();
        println!("{}", v.join(" "));
    }
}
"""
                % name
            )
            compile_cmd = ["rustc", "-O", "-o", "ik", "main.rs"]
        else:
            raise ValueError(language)

        subprocess.run(compile_cmd, cwd=tmp, check=True, capture_output=True)
        out = subprocess.run(
            [str(tmp / "ik"), *[repr(float(v)) for v in T.flatten()]],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    return names[out[0]], [
        np.array([float(v) for v in line.split()]) for line in out[1:]
    ]


@pytest.mark.slow
def test_generated_code():
    """
    The generated code gives valid solutions, and reports unreachable poses and
    singularities, in each language.
    """
    import shutil

    import roboticstoolbox as rtb
    from spatialmath import SE3

    rng = np.random.default_rng(0)
    languages = ["python"]
    languages += [
        lang
        for lang, cc in (("c", "cc"), ("c++", "c++"), ("rust", "rustc"))
        if shutil.which(cc)
    ]

    for name in ("Puma560", "Stanford", "UR5"):
        robot = getattr(rtb.models.DH, name)()
        revolute = [link.isrevolute for link in robot.links]
        plan = _make_plan(robot)

        q0 = np.where(revolute, rng.uniform(-1.5, 1.5, 6), rng.uniform(0.2, 0.6, 6))
        q_singular = q0.copy()
        q_singular[4] = 0.0  # wrist singularity
        # far away, with a generic orientation (an identity orientation can make
        # the wrist solutions degenerate, which reports a singularity instead)
        far = (SE3(10, 10, 10) * SE3.RPY([0.3, 0.4, 0.5])).A

        for language in languages:
            code = _emit(plan, language, "ikine")

            # a pose that can be reached
            T = robot.fkine(q0).A
            status, solutions = _run_generated(language, code, T)
            assert status == "OK", (name, language, status)
            for q in solutions:
                assert np.abs(robot.fkine(q).A - T).max() < 1e-6
            target = _wrap_joints(q0, revolute)
            assert any(
                np.allclose(_wrap_joints(q, revolute), target, atol=1e-5)
                for q in solutions
            ), (name, language)

            # a pose outside the workspace.  A prismatic joint here has no limit
            # so such an arm can reach any distance
            if all(revolute):
                status, solutions = _run_generated(language, code, far)
                assert status == "UNREACHABLE" and not solutions, (
                    name,
                    language,
                    status,
                )

            # a singularity: branches are dropped and the status says so
            status, solutions = _run_generated(
                language, code, robot.fkine(q_singular).A
            )
            assert status == "SINGULAR", (name, language, status)
            print(
                f"{name:10s} {language:7s} ok, singular pose gave {len(solutions)} solutions"
            )


@pytest.mark.slow
def test_pruning():
    """
    Wrong alternatives are dropped as they are chosen, so the final check
    against the forward kinematics should make no difference.
    """
    from dataclasses import replace

    import roboticstoolbox as rtb

    rng = np.random.default_rng(3)
    for name in ("IRB140", "KR5", "Puma560", "Stanford", "UR3", "UR5", "UR10"):
        robot = getattr(rtb.models.DH, name)()
        revolute = [link.isrevolute for link in robot.links]
        plan = _make_plan(robot, verify=True)
        checks = {st.joint: len(st.checks) for st in plan.steps}

        verified: dict = {}
        unverified: dict = {}
        exec(_emit(plan, "python", "ikine"), verified)  # noqa: S102
        exec(_emit(replace(plan, verify=False), "python", "ikine"), unverified)  # noqa: S102

        extra = 0
        for _ in range(5):
            q0 = np.where(revolute, rng.uniform(-1.5, 1.5, 6), rng.uniform(0.2, 0.6, 6))
            T = robot.fkine(q0).A
            a = [np.array(q) for q in verified["ikine"](T).solutions]
            b = [np.array(q) for q in unverified["ikine"](T).solutions]

            # every solution that survives the pruning must also pass the check
            assert all(
                any(
                    np.allclose(
                        _wrap_joints(x, revolute), _wrap_joints(y, revolute), atol=1e-6
                    )
                    for y in a
                )
                for x in b
            ), (name, "pruning left a wrong solution")
            extra += len(b) - len(a)
        print(
            f"{name:9s} equations checked per joint {checks}, extra solutions {extra}"
        )


def _nearest(sols: IKSolutions, q0, weights=None) -> np.ndarray:
    best = sols.nearest(q0, weights)
    assert best is not None
    return best


def test_ik_solutions():
    sols = IKSolutions(
        np.array([[0.1, 0.5, 0.2], [-3.1, 0.5, 0.2], [0.1, 2.0, 0.2]]),
        IKStatus.OK,
        np.array([True, True, False]),
    )
    assert len(sols) == 3

    # angles wrap: 3.1 is close to -3.1 (0.083 rad apart) but not to 0.1
    np.testing.assert_allclose(_nearest(sols, [3.1, 0.5, 0.2]), [-3.1, 0.5, 0.2])
    np.testing.assert_allclose(sols.distance([3.1, 0.5, 0.2])[1], 2 * np.pi - 6.2)

    # a prismatic joint is not wrapped
    sols = IKSolutions(
        np.array([[0.0, 0.0], [0.0, 2 * np.pi]]), IKStatus.OK, np.array([True, False])
    )
    np.testing.assert_allclose(sols.distance([0.0, 0.0]), [0.0, 2 * np.pi])

    # a weight of zero ignores a joint, and a weight scales its difference
    sols = IKSolutions(
        np.array([[0.0, 1.0], [0.5, 0.0]]), IKStatus.OK, np.array([True, True])
    )
    np.testing.assert_allclose(_nearest(sols, [0.0, 0.0]), [0.5, 0.0])
    np.testing.assert_allclose(_nearest(sols, [0.0, 0.0], weights=[1, 0]), [0.0, 1.0])
    np.testing.assert_allclose(sols.distance([0, 0], weights=[2, 1]), [1.0, 1.0])

    # no solutions, and why
    empty = IKSolutions(np.empty((0, 2)), IKStatus.UNREACHABLE, np.array([True, True]))
    assert len(empty) == 0 and empty.nearest([0, 0]) is None
    assert empty.status is IKStatus.UNREACHABLE


def test_from_generated():
    """Wrap the result of generated code."""
    import roboticstoolbox as rtb

    robot = rtb.models.DH.Puma560()
    namespace: dict = {}
    exec(generate_code(robot, "python"), namespace)  # noqa: S102

    q0 = np.array([0.3, 0.5, -0.4, 0.2, 0.6, 0.7])
    sols = IKSolutions.from_generated(namespace["ikine"](robot.fkine(q0).A), [True] * 6)
    assert sols.status is IKStatus.OK and len(sols) == 8
    np.testing.assert_allclose(_nearest(sols, q0), q0, atol=1e-6)


def test_flint_matches_sympy():
    """The FLINT and SymPy implementations of the polynomial steps agree."""
    import roboticstoolbox as rtb

    if _get_flint() is None:
        return
    symrobot, values = _symbolize(rtb.models.DH.Puma560())
    A, _ = _abstract_constants(_link_matrices(symrobot))
    pool = _equation_pool(A, 6)

    compared = 0
    for eq in pool[:80]:
        for j in _joints_in(eq):
            _linear_form.cache_clear()
            fast = _linear_form(eq, j, True)
            _linear_form.cache_clear()
            saved = _get_flint()
            sik._flint = False  # force the SymPy implementation
            try:
                slow = _linear_form(eq, j, True)
            finally:
                sik._flint = saved
                _linear_form.cache_clear()
            assert (fast is None) == (slow is None)
            if fast is not None:
                compared += 1
                assert slow is not None
                assert all(sp.expand(f - s) == 0 for f, s in zip(fast, slow))
    assert compared > 10


def test_without_flint():
    """Solving works without python-flint, using SymPy."""
    import roboticstoolbox as rtb

    saved = sik._flint
    sik._flint = False
    _linear_form.cache_clear()
    try:
        Q = ikine_sym(rtb.models.DH.Planar2(symbolic=True))
        assert Q is not None and len(Q) == 2
    finally:
        sik._flint = saved
        _linear_form.cache_clear()


def test_missing_flint_warns_once():
    """Without python-flint there is one warning, which says how to install it."""
    import sys

    saved = sik._flint
    sik._flint = None
    real = sys.modules.get("flint")
    sys.modules["flint"] = None  # type: ignore[assignment]  # makes the import fail
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert _get_flint() is None
            assert _get_flint() is None
        messages = [str(w.message) for w in caught]
        assert len(messages) == 1
        assert "pip install python-flint" in messages[0]
    finally:
        sik._flint = saved
        if real is None:
            del sys.modules["flint"]
        else:
            sys.modules["flint"] = real


@pytest.mark.slow
def test_polynomial_mode():
    """
    Solving every revolute joint as the roots of a polynomial in tan(q/2) gives
    the same solutions as the closed forms.
    """
    import roboticstoolbox as rtb

    for name in ("Puma560", "IRB140", "Stanford", "UR5"):
        robot = getattr(rtb.models.DH, name)()
        Q0 = ikine_sym(robot)
        assert Q0 is not None
        closed = _numeric_check(robot, Q0, trials=3)
        Q = ikine_sym(robot, polynomial=True)
        assert Q is not None
        assert any(isinstance(s[0], PolyRoots) for s in Q)
        assert _numeric_check(robot, Q, trials=3) == closed, name


@pytest.mark.slow
def test_polynomial_generated_code():
    """The generated root finder works in each language, and finds all solutions."""
    import shutil

    import roboticstoolbox as rtb

    languages = ["python"]
    languages += [
        lang for lang, cc in (("c", "cc"), ("c++", "c++")) if shutil.which(cc)
    ]

    rng = np.random.default_rng(7)
    for name in ("Puma560", "UR5"):
        robot = getattr(rtb.models.DH, name)()
        revolute = [link.isrevolute for link in robot.links]
        closed = _make_plan(robot)
        poly = _make_plan(robot, polynomial=True)
        assert any(step.poly for step in poly.steps)

        q0 = np.where(revolute, rng.uniform(-1.5, 1.5, 6), rng.uniform(0.2, 0.6, 6))
        T = robot.fkine(q0).A
        for language in languages:
            _, expected = _run_generated(language, _emit(closed, language, "ikine"), T)
            status, solutions = _run_generated(
                language, _emit(poly, language, "ikine"), T
            )
            assert status == "OK", (name, language, status)
            assert len(solutions) == len(expected), (name, language)
            for q in solutions:
                assert np.abs(robot.fkine(q).A - T).max() < 1e-6
            target = _wrap_joints(q0, revolute)
            assert any(
                np.allclose(_wrap_joints(q, revolute), target, atol=1e-5)
                for q in solutions
            ), (name, language)


def test_polynomial_rust_not_supported():
    import pytest
    import roboticstoolbox as rtb

    with pytest.raises(NotImplementedError):
        generate_code(rtb.models.DH.Puma560(), "rust", polynomial=True)


def test_header():
    """Every generated file says where it came from, when, and for which robot."""
    import roboticstoolbox as rtb

    robot = rtb.models.DH.Stanford()
    plan = _make_plan(robot)
    for language in ("python", "c", "c++", "rust"):
        code = _emit(plan, language, "ikine")
        head = code[: code.index("ikine")]
        assert "Generated by Robotics Toolbox for Python by Peter Corke" in head, (
            language
        )
        assert "https://github.com/petercorke/robotics-toolbox-python" in head
        assert re.search(r"Generated \d{4}-\d\d-\d\d \d\d:\d\d:\d\d", head), language
        assert "Stanford arm" in head and "RRPRRR" in head  # from str(robot)
        assert f"alpha  = {[float(link.alpha) for link in robot.links]}" in head
        assert "flip   = [False" in head


def test_generated_rust_smoke():
    """Check the overall form of the Rust code, which :func:`test_generated_code`
    also compiles and runs if there is a Rust compiler."""
    import roboticstoolbox as rtb

    code = generate_code(rtb.models.DH.Puma560(), "rust", "ikine")
    assert "pub fn ikine(t: &[f64; 16], tol: f64) -> IkResult" in code
    assert code.count("{") == code.count("}")
    assert code.count("(") == code.count(")")
