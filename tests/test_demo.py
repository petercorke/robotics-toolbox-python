#!/usr/bin/env python
"""
Smoke tests for the console-script demos in roboticstoolbox.demo.

eigdemo is matplotlib-only, so it runs to completion under the Agg backend
(no window ever opens, plt.show() just warns and returns). tripleangledemo
and twistdemo use Swift, which normally never returns on its own -- they
accept a ``--test`` flag that bounds Swift's env.run() to a short sim-time
duration instead of running until the browser disconnects, combined with
SWIFT_HEADLESS=1 so no browser is required at all.
"""

import subprocess
import sys
import unittest

_TIMEOUT = 30
_ENV = {"MPLBACKEND": "Agg"}


def _run(module: str, *args: str, **extra_env: str) -> subprocess.CompletedProcess:
    import os

    env = dict(os.environ, **_ENV, **extra_env)
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=_TIMEOUT,
        env=env,
    )


class TestEigdemo(unittest.TestCase):
    def test_default_matrix(self):
        result = _run("roboticstoolbox.demo.eigdemo")
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())
        out = result.stdout.decode()
        self.assertIn("matrix A =", out)
        self.assertIn("λ1 =", out)

    def test_custom_matrix(self):
        result = _run("roboticstoolbox.demo.eigdemo", "1", "0", "0", "2")
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())
        self.assertIn("matrix A =", result.stdout.decode())

    def test_help(self):
        result = _run("roboticstoolbox.demo.eigdemo", "--help")
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())
        self.assertIn("eigdemo", result.stdout.decode())


class TestSwiftDemos(unittest.TestCase):
    def setUp(self):
        try:
            import swift  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            self.skipTest("swift-sim not installed")

    def test_tripleangledemo(self):
        result = _run(
            "roboticstoolbox.demo.tripleangledemo", "--test", SWIFT_HEADLESS="1"
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())

    def test_twistdemo(self):
        result = _run("roboticstoolbox.demo.twistdemo", "--test", SWIFT_HEADLESS="1")
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
