"""Tests of ``roboticstoolbox.robot.symbolic_ik_rr``

These were at the end of the module, where CI did not run them.
"""

import pytest
from roboticstoolbox.robot.symbolic_ik import IKStatus
import numpy as np
from roboticstoolbox.robot.symbolic_ik_rr import (
    AnalyticIK,
    ClosedFormIK,
    General6RIK,
    _C_KERNEL,
    classify,
    make_solver,
)


def test_classify():
    import roboticstoolbox as rtb

    models = rtb.models.DH
    # spherical wrist: joints 4, 5, 6
    for name in ("Puma560", "IRB140", "KR5", "Stanford"):
        s = classify(getattr(models, name)())
        assert 4 in s.spherical, name
        assert not s.general
    # the UR arms: parallel axes 2, 3, 4
    for name in ("UR3", "UR5", "UR10"):
        s = classify(getattr(models, name)())
        assert s.parallel == (2,) and not s.spherical, name
    # neither
    for name in ("Jaco", "Mico"):
        assert classify(getattr(models, name)()).general, name


def test_classify_tolerance():
    """Round-off does not matter, unlike the exact comparison of isspherical."""
    import roboticstoolbox as rtb
    from roboticstoolbox import DHRobot, RevoluteDH

    robot = rtb.models.DH.Puma560()
    assert robot.isspherical()
    noisy = DHRobot(
        [
            RevoluteDH(
                d=link.d + 1e-13, a=link.a + (1e-13 if link.a == 0 else 0),
                alpha=link.alpha * (1 + 1e-12), offset=link.offset,
            )
            for link in robot.links
        ]
    )  # fmt: skip
    assert not noisy.isspherical()  # exact comparison fails
    assert classify(noisy, tol=1e-8).spherical  # a tolerance does not


@pytest.mark.slow
def test_choice_of_split():
    """The Jaco has pathological conditioning for two of the splits."""
    import roboticstoolbox as rtb

    solver = General6RIK(rtb.models.DH.Jaco())
    assert solver.k == 1
    assert solver.condition[1] < 1e4 < solver.condition[0]
    assert solver.condition[1] < 1e4 < solver.condition[2]


@pytest.mark.slow
def test_general_6r():
    import roboticstoolbox as rtb

    for name in ("Jaco", "Mico", "UR5"):
        robot = getattr(rtb.models.DH, name)()
        solver = General6RIK(robot)
        assert solver.validate(100) >= 0.99, name

        # solutions reach the pose
        q0 = np.array([0.3, -0.8, 1.1, 0.4, -0.6, 0.9])
        T = robot.fkine(q0).A
        sols = solver._solve_pose(T)
        assert sols.status is IKStatus.OK and len(sols) >= 2
        for q in sols.q:
            assert np.abs(robot.fkine(q).A - T).max() < 1e-6
        best = sols.nearest(q0)
        assert best is not None
        np.testing.assert_allclose(best, q0, atol=1e-5)


def test_unreachable():
    import roboticstoolbox as rtb
    from spatialmath import SE3

    solver = General6RIK(rtb.models.DH.Jaco())
    sols = solver._solve_pose((SE3(5, 5, 5) * SE3.RPY([0.3, 0.4, 0.5])).A)
    assert sols.status is IKStatus.UNREACHABLE and len(sols) == 0


def test_not_applicable():
    """A spherical wrist is not the general case, and validation says so."""
    import roboticstoolbox as rtb

    robot = rtb.models.DH.Puma560()
    assert not classify(robot).general
    assert General6RIK(robot, k=1).validate(30) < 0.5


def test_bad_robot():
    import pytest
    import roboticstoolbox as rtb

    with pytest.raises(ValueError):
        General6RIK(rtb.models.DH.Planar2())
    with pytest.raises(ValueError):
        General6RIK(rtb.models.DH.Stanford())  # a prismatic joint


def test_make_solver_prismatic():
    """An arm with a prismatic joint and no wrist structure is refused clearly."""
    import pytest
    from roboticstoolbox import DHRobot, PrismaticDH, RevoluteDH

    # the RPRRPR arm of Raghavan and Roth, 1993, numerical example of Appendix 1
    deg = np.pi / 180
    robot = DHRobot(
        [
            RevoluteDH(d=0.21, a=1.46, alpha=135 * deg),
            PrismaticDH(theta=65 * deg, a=0.56, alpha=78 * deg, qlim=[0, 1]),
            RevoluteDH(d=0.29, a=0.38, alpha=23 * deg),
            RevoluteDH(d=-0.54, a=0.56, alpha=46 * deg),
            PrismaticDH(theta=54 * deg, a=1.08, alpha=35 * deg, qlim=[0, 1]),
            RevoluteDH(d=0.48, a=0.67, alpha=47 * deg),
        ]
    )
    assert classify(robot).general
    with pytest.raises(NotImplementedError, match=r"prismatic joint\(s\) 2, 5"):
        make_solver(robot)


def test_generated_general_6r():
    """The generated code runs on its own, and agrees with the class."""
    import roboticstoolbox as rtb

    rng = np.random.default_rng(5)
    for name in ("Jaco", "Mico"):
        robot = getattr(rtb.models.DH, name)()
        solver = General6RIK(robot)
        namespace: dict = {}
        exec(solver.generate("python", "ikine"), namespace)  # noqa: S102

        hits = 0
        for _ in range(50):
            q0 = rng.uniform(-np.pi, np.pi, 6)
            T = robot.fkine(q0).A
            result = namespace["ikine"](T)
            expected = solver._solve_pose(T)
            assert result.status.name == expected.status.name
            assert len(result.solutions) == len(expected)
            for q in result.solutions:
                assert np.abs(robot.fkine(np.array(q)).A - T).max() < 1e-6
            hits += any(General6RIK._same(np.array(q), q0) for q in result.solutions)
        assert hits >= 49, name

        # a pose outside the workspace
        from spatialmath import SE3

        far = (SE3(5, 5, 5) * SE3.RPY([0.3, 0.4, 0.5])).A
        assert namespace["ikine"](far).status.name == "UNREACHABLE"

        # without SciPy, which is optional
        namespace["scipy"] = None
        hits = sum(
            any(
                General6RIK._same(np.array(q), q0)
                for q in namespace["ikine"](robot.fkine(q0).A).solutions
            )
            for q0 in rng.uniform(-np.pi, np.pi, (50, 6))
        )
        assert hits >= 47, (name, hits)


def test_generate_other_languages():
    import pytest
    import roboticstoolbox as rtb

    with pytest.raises(NotImplementedError):
        General6RIK(rtb.models.DH.Jaco()).generate("rust")


def _run_kernel(cases: str, language: str = "c") -> list[str]:
    """
    Compile the C kernel with a small driver, and run it

    :param cases: text commands for the driver, see the C code below
    :param language: ``"c"`` or ``"c++"``, to check it is valid as both
    :returns: lines of output.  The compiler must give no warnings.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    driver = (
        r"""
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
"""
        + _C_KERNEL
        + r"""
static void rd(double *a, int k) { for (int i = 0; i < k; ++i) if (scanf("%lf", &a[i]) != 1) exit(2); }
int main(void) {
    char mode;
    while (scanf(" %c", &mode) == 1) {
        int n, m, k;
        double A[1024], B[1024], X[1024], Q[1024], s[32], wr[32], wi[32];
        if (mode == 'e') {  /* eigenvalues of an n-by-n matrix */
            if (scanf("%d", &n) != 1) return 2;
            rd(A, n * n);
            int ok = ikrr_eig(A, n, wr, wi);
            printf("%d\n", ok);
            for (int i = 0; i < n; ++i) printf("%.17g %.17g\n", wr[i], wi[i]);
        } else if (mode == 's') {  /* singular values and vectors of an n-by-n matrix */
            if (scanf("%d", &n) != 1) return 2;
            rd(A, n * n);
            ikrr_svd(A, n, s, X);
            for (int i = 0; i < n; ++i) printf("%.17g\n", s[i]);
            for (int i = 0; i < n * n; ++i) printf("%.17g\n", X[i]);
        } else if (mode == 'q') {  /* QR of an m-by-n matrix, and a least squares problem */
            if (scanf("%d %d", &m, &n) != 2) return 2;
            rd(A, m * n); rd(B, m);
            ikrr_qr(A, m, n, Q);
            ikrr_qr_solve(A, Q, m, n, B, X);
            for (int i = 0; i < n; ++i) printf("%.17g\n", X[i]);
            for (int i = 0; i < m * m; ++i) printf("%.17g\n", Q[i]);
            for (int i = 0; i < m * n; ++i) printf("%.17g\n", A[i]);
        } else if (mode == 'l') {  /* A X = B, n-by-n and n-by-k */
            if (scanf("%d %d", &n, &k) != 2) return 2;
            rd(A, n * n); rd(B, n * k);
            int ok = ikrr_lu_solve(A, n, B, k);
            printf("%d\n", ok);
            for (int i = 0; i < n * k; ++i) printf("%.17g\n", B[i]);
        }
    }
    return 0;
}
"""
    )
    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / ("kernel.c" if language == "c" else "kernel.cpp")
        source.write_text(driver)
        command = (
            [
                "cc",
                "-std=c99",
                "-O2",
                "-Wall",
                "-Wextra",
                "-o",
                "kernel",
                source.name,
                "-lm",
            ]
            if language == "c"
            else [
                "c++",
                "-std=c++17",
                "-O2",
                "-Wall",
                "-Wextra",
                "-o",
                "kernel",
                source.name,
            ]
        )
        build = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
        assert build.returncode == 0 and build.stderr == "", build.stderr
        run = subprocess.run(
            [str(tmp / "kernel")],
            input=cases,
            capture_output=True,
            text=True,
            check=True,
        )
    return run.stdout.split()


def _numbers(a: np.ndarray) -> str:
    return " ".join(repr(float(x)) for x in np.asarray(a).ravel())


@pytest.mark.slow
def test_c_kernel_eigenvalues():
    import shutil

    rng = np.random.default_rng(0)
    matrices = [rng.normal(size=(24, 24)) for _ in range(100)]
    matrices += [
        rng.normal(size=(n, n)) for n in (1, 2, 3, 5, 8, 12) for _ in range(10)
    ]
    # the matrices the method really uses: companion matrices for the Jaco
    import roboticstoolbox as rtb

    robot = rtb.models.DH.Jaco()
    solver = General6RIK(robot)
    module = solver._module(solver.k)
    for q, T in solver._random_poses(30, 3):
        *_, M0, M1, M2 = module["_polynomial"](T)
        Z, I = np.zeros((12, 12)), np.eye(12)
        matrices.append(
            np.block([[Z, I], [-np.linalg.solve(M2, M0), -np.linalg.solve(M2, M1)]])
        )

    cases = "".join(f"e {len(A)} {_numbers(A)}\n" for A in matrices)
    for language in ["c"] + (["c++"] if shutil.which("c++") else []):
        out = iter(_run_kernel(cases, language))
        for A in matrices:
            assert next(out) == "1", "failed to converge"
            n = len(A)
            ours = np.array(
                [complex(float(next(out)), float(next(out))) for _ in range(n)]
            )
            theirs = np.linalg.eigvals(A)
            scale = max(1.0, np.abs(theirs).max())
            # each eigenvalue has a partner in the other set
            for z in ours:
                assert np.abs(theirs - z).min() < 1e-7 * scale, (n, z)
            for z in theirs:
                assert np.abs(ours - z).min() < 1e-7 * scale, (n, z)


@pytest.mark.slow
def test_c_kernel_linear_algebra():
    rng = np.random.default_rng(1)
    # SVD of square matrices, including a singular one
    squares = [rng.normal(size=(12, 12)) for _ in range(30)]
    singular = rng.normal(size=(12, 12))
    singular[:, 5] = singular[:, 2] * 3.0
    squares.append(singular)
    # QR and least squares
    tall = [(rng.normal(size=(14, 8)), rng.normal(size=14)) for _ in range(30)]
    tall += [(rng.normal(size=(12, 6)), rng.normal(size=12)) for _ in range(30)]
    # linear systems
    systems = [
        (rng.normal(size=(12, 12)), rng.normal(size=(12, 24))) for _ in range(30)
    ]

    cases = "".join(f"s {len(A)} {_numbers(A)}\n" for A in squares)
    cases += "".join(
        f"q {A.shape[0]} {A.shape[1]} {_numbers(A)} {_numbers(b)}\n" for A, b in tall
    )
    cases += "".join(
        f"l {len(A)} {B.shape[1]} {_numbers(A)} {_numbers(B)}\n" for A, B in systems
    )
    cases += "l 2 1 1 2 2 4 1 2\n"  # singular
    out = iter(_run_kernel(cases))

    for A in squares:
        n = len(A)
        s = np.array([float(next(out)) for _ in range(n)])
        V = np.array([float(next(out)) for _ in range(n * n)]).reshape(n, n)
        np.testing.assert_allclose(s, np.linalg.svd(A, compute_uv=False), atol=1e-12)
        np.testing.assert_allclose(V.T @ V, np.eye(n), atol=1e-12)
        np.testing.assert_allclose(
            np.linalg.norm(A @ V, axis=0), s, atol=1e-12
        )  # A v_j = s_j u_j

    for A, b in tall:
        m, n = A.shape
        x = np.array([float(next(out)) for _ in range(n)])
        Qf = np.array([float(next(out)) for _ in range(m * m)]).reshape(m, m)
        R = np.array([float(next(out)) for _ in range(m * n)]).reshape(m, n)
        np.testing.assert_allclose(x, np.linalg.lstsq(A, b, rcond=None)[0], atol=1e-10)
        np.testing.assert_allclose(Qf @ R, A, atol=1e-12)
        np.testing.assert_allclose(Qf.T @ Qf, np.eye(m), atol=1e-12)
        np.testing.assert_allclose(np.tril(R, -1), 0, atol=0)
        np.testing.assert_allclose(A.T @ Qf[:, n:], 0, atol=1e-12)  # null space of A.T

    for A, B in systems:
        assert next(out) == "1"
        X = np.array([float(next(out)) for _ in range(B.size)]).reshape(B.shape)
        np.testing.assert_allclose(A @ X, B, atol=1e-10)
    assert next(out) == "0"  # singular, the right hand side is printed anyway
    next(out), next(out)
    assert next(out, None) is None


def _run_general_6r(language: str, code: str, poses: list[np.ndarray]):
    """
    Compile generated C or C++ for the general 6R method, run it on poses

    :returns: for each pose ``(status, solutions)``.  The compiler must give no
        warnings.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    if language == "c":
        main = r"""
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    int n;
    if (scanf("%d", &n) != 1) return 2;
    for (int p = 0; p < n; ++p) {
        double T[16];
        static double sols[IKINE_MAX_SOLUTIONS][IKINE_DOF];
        int count;
        for (int i = 0; i < 16; ++i) if (scanf("%lf", &T[i]) != 1) return 2;
        ik_status status = ikine(T, 1e-6, sols, &count);
        printf("%d %d\n", (int)status, count);
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < IKINE_DOF; ++j) printf("%.17g ", sols[i][j]);
            printf("\n");
        }
    }
    return 0;
}
"""
        source, command = (
            "ik.c",
            ["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-o", "ik", "ik.c", "-lm"],
        )
    else:
        main = r"""
#include <cstdio>
int main() {
    int n;
    if (std::scanf("%d", &n) != 1) return 2;
    for (int p = 0; p < n; ++p) {
        double T[16];
        for (int i = 0; i < 16; ++i) if (std::scanf("%lf", &T[i]) != 1) return 2;
        IkResult r = ikine(T);
        std::printf("%d %d\n", (int)r.status, (int)r.solutions.size());
        for (const auto &q : r.solutions) {
            for (double v : q) std::printf("%.17g ", v);
            std::printf("\n");
        }
    }
    return 0;
}
"""
        source, command = (
            "ik.cpp",
            ["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-o", "ik", "ik.cpp"],
        )

    with tempfile.TemporaryDirectory() as tmp_name:
        tmp = Path(tmp_name)
        (tmp / source).write_text(code + main)
        build = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
        assert build.returncode == 0 and build.stderr == "", build.stderr
        stdin = f"{len(poses)}\n" + "\n".join(_numbers(T) for T in poses)
        out = subprocess.run(
            [str(tmp / "ik")], input=stdin, capture_output=True, text=True, check=True
        ).stdout.splitlines()

    results, line = [], 0
    for _ in poses:
        status, count = (int(x) for x in out[line].split())
        solutions = [
            np.array([float(x) for x in out[line + 1 + i].split()])
            for i in range(count)
        ]
        results.append((("OK", "SINGULAR", "UNREACHABLE")[status], solutions))
        line += 1 + count
    return results


@pytest.mark.slow
def test_generated_c_general_6r():
    """The generated C and C++ agree with the Python class, and need no library."""
    import shutil

    import roboticstoolbox as rtb
    from spatialmath import SE3

    languages = [lang for lang, cc in (("c", "cc"), ("c++", "c++")) if shutil.which(cc)]
    for name in ("Jaco", "Mico", "UR5"):
        robot = getattr(rtb.models.DH, name)()
        solver = General6RIK(robot)
        poses = [T for _, T in solver._random_poses(60, 21)]
        poses.append((SE3(5, 5, 5) * SE3.RPY([0.3, 0.4, 0.5])).A)  # unreachable
        expected = [solver._solve_pose(T) for T in poses]

        for language in languages:
            code = solver.generate(language, "ikine")
            assert "Generated by Robotics Toolbox for Python by Peter Corke" in code
            for (status, solutions), want, T in zip(
                _run_general_6r(language, code, poses), expected, poses
            ):
                assert status == want.status.name, (name, language)
                assert len(solutions) == len(want), (name, language)
                for q in solutions:
                    assert np.abs(robot.fkine(q).A - T).max() < 1e-6
                    assert any(General6RIK._same(q, other) for other in want.q), (
                        name,
                        language,
                    )


def test_solver_interface():
    """Both solvers solve, generate in several languages, and validate."""
    import pytest
    import roboticstoolbox as rtb

    closed = make_solver(rtb.models.DH.Puma560(), trials=20)
    general = make_solver(rtb.models.DH.Jaco(), trials=20)
    assert isinstance(closed, ClosedFormIK) and isinstance(general, General6RIK)
    assert closed.method == "closed form" and general.method == "general 6R"
    assert all(isinstance(s, AnalyticIK) for s in (closed, general))

    for language in ("python", "c", "c++", "rust"):
        code = closed.generate(language, "puma_ik")
        assert "puma_ik" in code and "Generated by Robotics Toolbox" in code, language
    for language in ("python", "c", "c++"):
        code = general.generate(language, "jaco_ik")
        assert "jaco_ik" in code and "Generated by Robotics Toolbox" in code, language
    with pytest.raises(NotImplementedError):
        general.generate("rust")
    with pytest.raises(ValueError):
        closed.generate("fortran")

    # an arm with a prismatic joint is validated too
    assert make_solver(rtb.models.DH.Stanford(), trials=20).validate(20) >= 0.95


@pytest.mark.slow
def test_make_solver():
    import pytest
    import roboticstoolbox as rtb

    models = rtb.models.DH
    for name, kind in (
        ("Puma560", ClosedFormIK),
        ("Stanford", ClosedFormIK),
        ("UR5", ClosedFormIK),
        ("Jaco", General6RIK),
    ):
        robot = getattr(models, name)()
        solver = make_solver(robot, trials=30)
        assert isinstance(solver, kind), name

        q0 = np.where(
            [link.isrevolute for link in robot.links],
            [0.3, -0.8, 1.1, 0.4, -0.6, 0.9],
            [0.3, 0.5, 0.4, 0.2, 0.6, 0.5],
        )
        sols = solver._solve_pose(robot.fkine(q0).A)
        assert sols.status is IKStatus.OK
        best = sols.nearest(q0)
        assert best is not None
        np.testing.assert_allclose(best, q0, atol=1e-5)

    with pytest.raises(ValueError, match="recovered"):
        make_solver(models.Jaco(), trials=10, min_success=1.01)
