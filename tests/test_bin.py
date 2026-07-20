#!/usr/bin/env python
"""
Smoke tests for command-line entry points in roboticstoolbox.bin.

``--help`` tests verify that imports and argument parsing work and the tool
exits cleanly.  Startup tests verify that the tool reaches the interactive
IPython prompt: stdin is closed, so IPython's non-interactive EOF detection
ends the session cleanly instead of blocking, and we check the exit code
and captured output.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TIMEOUT = 15


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command via the current Python interpreter's entry-point module."""
    return subprocess.run(
        [sys.executable, "-m"] + args,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=_TIMEOUT,
        **kwargs,
    )


class TestRtbtool(unittest.TestCase):
    def test_help(self):
        result = _run(["roboticstoolbox.bin.rtbtool", "--help"])
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())

    def test_startup(self):
        """Tool should reach the interactive prompt without error."""
        result = _run(["roboticstoolbox.bin.rtbtool", "--no-banner"])
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())

    def test_missing_script(self):
        result = _run(["roboticstoolbox.bin.rtbtool", "/no/such/script.py"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"script does not exist", result.stderr)

    def test_run_script(self):
        """Script argument should execute with the RTB namespace available."""
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "sentinel.py"
            script.write_text('print("SENTINEL_OUTPUT", panda.name)\n')
            result = _run(["roboticstoolbox.bin.rtbtool", str(script)])
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())
        self.assertIn(b"SENTINEL_OUTPUT Panda", result.stdout)

    def test_options_envvar(self):
        """RTB_OPTIONS should be parsed the same as command-line arguments."""
        env = dict(os.environ, RTB_OPTIONS="--prompt envtest>")
        result = _run(["roboticstoolbox.bin.rtbtool", "--no-banner"], env=env)
        self.assertEqual(result.returncode, 0, msg=result.stderr.decode())
        self.assertIn(b"envtest>", result.stdout)


if __name__ == "__main__":
    unittest.main()
