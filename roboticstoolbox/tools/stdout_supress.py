from contextlib import contextmanager
import tempfile
import sys
import os


@contextmanager
def stdout_supress():
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
    finally:
        tfile.close()
        os.close(saved_stdout_fd)
