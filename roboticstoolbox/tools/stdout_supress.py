from contextlib import contextmanager
import tempfile
import sys
import os

import ctypes.util
import ctypes
import platform

# # https://github.com/minrk/wurlitzer/blob/master/wurlitzer.py

# if platform.system() == "Linux":
#     libc = ctypes.CDLL(None)
#     c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')
# elif platform.system() == "Darwin":
#     libc = ctypes.CDLL(None)
#     c_stdout = ctypes.c_void_p.in_dll(libc, '__stdoutp')
# elif platform.system() == "Windows":
#     class FILE(ctypes.Structure):
#         _fields_ = [
#             ("_ptr", ctypes.c_char_p),
#             ("_cnt", ctypes.c_int),
#             ("_base", ctypes.c_char_p),
#             ("_flag", ctypes.c_int),
#             ("_file", ctypes.c_int),
#             ("_charbuf", ctypes.c_int),
#             ("_bufsize", ctypes.c_int),
#             ("_tmpfname", ctypes.c_char_p),
#         ]

#     # Gives you the name of the library that you should really use
#     # (and then load through ctypes.CDLL
#     msvcrt = ctypes.CDLL(ctypes.util.find_msvcrt())
#     # libc was used in the original example in _redirect_stdout()
#     libc = msvcrt
#     iob_func = msvcrt.__iob_func
#     iob_func.restype = ctypes.POINTER(FILE)
#     iob_func.argtypes = []

#     array = iob_func()

#     s_stdin = ctypes.addressof(array[0])
#     c_stdout = ctypes.addressof(array[1])


# @contextmanager
# def stdout_supress():
#     original_stdout_fd = sys.stdout.fileno()

#     def _redirect_stdout(to_fd):
#         # Flush the C-level buffer stdout
#         libc.fflush(c_stdout)

#         # # Flush and close sys.stdout - also closes the file descriptor (fd)
#         # sys.stdout.close()

#         # Make original_stdout_fd point to the same file as to_fd
#         os.dup2(to_fd, original_stdout_fd)

#     # Save a copy of the original stdout fd in saved_stdout_fd
#     saved_stdout_fd = os.dup(original_stdout_fd)
#     try:
#         # Create a temporary file and redirect stdout to it
#         tfile = tempfile.TemporaryFile(mode='w+b')
#         _redirect_stdout(tfile.fileno())
#         # Yield to caller, then redirect stdout back to the saved fd
#         yield
#         _redirect_stdout(saved_stdout_fd)
#     finally:
#         tfile.close()
#         os.close(saved_stdout_fd)
