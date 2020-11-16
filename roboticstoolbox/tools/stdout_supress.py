# Code Author Github user minrk
# Originally from https://github.com/minrk/wurlitzer/blob/master/wurlitzer.py


from __future__ import print_function

from contextlib import contextmanager
import ctypes
import errno
from fcntl import fcntl, F_GETFL, F_SETFL
import io
import os

try:
    from queue import Queue
except ImportError:   # pragma nocover
    from Queue import Queue

import selectors
import sys
import threading
import time

libc = ctypes.CDLL(None)

STDOUT = 2
PIPE = 3

try:
    c_stdout_p = ctypes.c_void_p.in_dll(libc, 'stdout')
    c_stderr_p = ctypes.c_void_p.in_dll(libc, 'stderr')
except ValueError:  # pragma: no cover
    # libc.stdout is has a funny name on OS X
    c_stdout_p = ctypes.c_void_p.in_dll(libc, '__stdoutp')  # pragma: no cover
    c_stderr_p = ctypes.c_void_p.in_dll(libc, '__stderrp')  # pragma: no cover

_default_encoding = getattr(sys.stdin, 'encoding', None) or 'utf8'
if _default_encoding.lower() == 'ascii':
    # don't respect ascii
    _default_encoding = 'utf8'  # pragma: no cover


def dup2(a, b, timeout=3):
    """Like os.dup2, but retry on EBUSY"""
    dup_err = None
    # give FDs 3 seconds to not be busy anymore
    for i in range(int(10 * timeout)):
        try:
            return os.dup2(a, b)
        except OSError as e:   # pragma nocover
            dup_err = e
            if e.errno == errno.EBUSY:
                time.sleep(0.1)
            else:
                raise
    if dup_err:   # pragma nocover
        raise dup_err


class Wurlitzer(object):  # pragma: no cover
    """Class for Capturing Process-level FD output via dup2

    Typically used via `wurlitzer.capture`
    """
    flush_interval = 0.2

    def __init__(self, stdout=None, stderr=None, encoding=_default_encoding):
        """
        Parameters
        ----------
        stdout: stream or None
            The stream for forwarding stdout.
        stderr = stream or None
            The stream for forwarding stderr.
        encoding: str or None
            The encoding to use, if streams should be interpreted as text.
        """
        self._stdout = stdout
        self._stderr = stderr
        self.encoding = encoding
        self._save_fds = {}
        self._real_fds = {}
        self._handlers = {}
        self._handlers['stderr'] = self._handle_stderr
        self._handlers['stdout'] = self._handle_stdout

    def _setup_pipe(self, name):
        real_fd = getattr(sys, '__%s__' % name).fileno()
        save_fd = os.dup(real_fd)
        self._save_fds[name] = save_fd

        pipe_out, pipe_in = os.pipe()
        dup2(pipe_in, real_fd)
        os.close(pipe_in)
        self._real_fds[name] = real_fd

        # make pipe_out non-blocking
        flags = fcntl(pipe_out, F_GETFL)
        fcntl(pipe_out, F_SETFL, flags | os.O_NONBLOCK)
        return pipe_out

    def _decode(self, data):
        """Decode data, if any

        Called before passing to stdout/stderr streams
        """
        if self.encoding:
            data = data.decode(self.encoding, 'replace')
        return data

    def _handle_stdout(self, data):
        if self._stdout:
            self._stdout.write(self._decode(data))

    def _handle_stderr(self, data):
        if self._stderr:
            self._stderr.write(self._decode(data))

    def _setup_handle(self):
        """Setup handle for output, if any"""
        self.handle = (self._stdout, self._stderr)

    def _finish_handle(self):
        """Finish handle, if anything should be done when it's all wrapped up.
        """
        pass

    def _flush(self):
        """flush sys.stdout/err and low-level FDs"""
        if self._stdout and sys.stdout:
            sys.stdout.flush()
        if self._stderr and sys.stderr:
            sys.stderr.flush()

        libc.fflush(c_stdout_p)
        libc.fflush(c_stderr_p)

    def __enter__(self):
        # flush anything out before starting
        self._flush()
        # setup handle
        self._setup_handle()
        self._control_r, self._control_w = os.pipe()

        # create pipe for stdout
        pipes = [self._control_r]
        names = {self._control_r: 'control'}
        if self._stdout:
            pipe = self._setup_pipe('stdout')
            pipes.append(pipe)
            names[pipe] = 'stdout'
        if self._stderr:
            pipe = self._setup_pipe('stderr')
            pipes.append(pipe)
            names[pipe] = 'stderr'

        # flush pipes in a background thread to avoid blocking
        # the reader thread when the buffer is full
        flush_queue = Queue()

        def flush_main():
            while True:
                msg = flush_queue.get()
                if msg == 'stop':
                    return
                self._flush()

        flush_thread = threading.Thread(target=flush_main)
        flush_thread.daemon = True
        flush_thread.start()

        def forwarder():
            """Forward bytes on a pipe to stream messages"""
            draining = False
            flush_interval = 0
            poller = selectors.DefaultSelector()

            for pipe_ in pipes:
                poller.register(pipe_, selectors.EVENT_READ)

            while pipes:
                events = poller.select(flush_interval)
                if events:
                    # found something to read, don't block select until
                    # we run out of things to read
                    flush_interval = 0
                else:
                    # nothing to read
                    if draining:
                        # if we are draining and there's nothing to read, stop
                        break
                    else:
                        # nothing to read, get ready to wait.
                        # flush the streams in case there's something waiting
                        # to be written.
                        flush_queue.put('flush')
                        flush_interval = self.flush_interval
                        continue

                for selector_key, flags in events:
                    fd = selector_key.fd
                    if fd == self._control_r:
                        draining = True
                        pipes.remove(self._control_r)
                        poller.unregister(self._control_r)
                        os.close(self._control_r)
                        continue
                    name = names[fd]
                    data = os.read(fd, 1024)
                    if not data:
                        # pipe closed, stop polling it
                        pipes.remove(fd)
                        poller.unregister(fd)
                        os.close(fd)
                    else:
                        handler = getattr(self, '_handle_%s' % name)
                        handler(data)
                if not pipes:
                    # pipes closed, we are done
                    break
            # stop flush thread
            flush_queue.put('stop')
            flush_thread.join()
            # cleanup pipes
            [os.close(pipe) for pipe in pipes]

        self.thread = threading.Thread(target=forwarder)
        self.thread.daemon = True
        self.thread.start()

        return self.handle

    def __exit__(self, exc_type, exc_value, traceback):
        # flush before exiting
        self._flush()

        # signal output is complete on control pipe
        os.write(self._control_w, b'\1')
        self.thread.join()
        os.close(self._control_w)

        # restore original state
        for name, real_fd in self._real_fds.items():
            save_fd = self._save_fds[name]
            dup2(save_fd, real_fd)
            os.close(save_fd)
        # finalize handle
        self._finish_handle()


@contextmanager
def pipes(stdout=PIPE, stderr=PIPE, encoding=_default_encoding):  # pragma: no cover  # noqa
    """Capture C-level stdout/stderr in a context manager.
    The return value for the context manager is (stdout, stderr).
    Examples
    --------
    >>> with capture() as (stdout, stderr):
    ...     printf("C-level stdout")
    ... output = stdout.read()
    """
    stdout_pipe = stderr_pipe = False
    # setup stdout
    if stdout == PIPE:
        stdout_r, stdout_w = os.pipe()
        stdout_w = os.fdopen(stdout_w, 'wb')
        if encoding:
            stdout_r = io.open(stdout_r, 'r', encoding=encoding)
        else:
            stdout_r = os.fdopen(stdout_r, 'rb')
        stdout_pipe = True
    else:
        stdout_r = stdout_w = stdout
    # setup stderr
    if stderr == STDOUT:
        stderr_r = None
        stderr_w = stdout_w
    elif stderr == PIPE:
        stderr_r, stderr_w = os.pipe()
        stderr_w = os.fdopen(stderr_w, 'wb')
        if encoding:
            stderr_r = io.open(stderr_r, 'r', encoding=encoding)
        else:
            stderr_r = os.fdopen(stderr_r, 'rb')
        stderr_pipe = True
    else:
        stderr_r = stderr_w = stderr
    if stdout_pipe or stderr_pipe:
        capture_encoding = None
    else:
        capture_encoding = encoding
    w = Wurlitzer(stdout=stdout_w, stderr=stderr_w, encoding=capture_encoding)
    try:
        with w:
            yield stdout_r, stderr_r
    finally:
        # close pipes
        if stdout_pipe:
            stdout_w.close()
        if stderr_pipe:
            stderr_w.close()
