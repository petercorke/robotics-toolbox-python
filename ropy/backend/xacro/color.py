# Copyright (c) 2015, Bielefeld University
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Bielefeld University
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior
#       written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from __future__ import print_function
import sys

# bold colors
_ansi = {'red': 91, 'yellow': 93}


def is_tty(stream):  # taken from catkin_tools/common.py  # pragma: no cover
    """Returns True if the given stream is a tty, else False"""
    return hasattr(stream, 'isatty') and stream.isatty()


def colorize(msg, color, file=sys.stderr, alt_text=None):  # pragma: no cover
    if color and is_tty(file):
        return '\033[%dm%s\033[0m' % (_ansi[color], msg)
    elif alt_text:
        return '%s%s' % (alt_text, msg)
    else:
        return msg


def message(msg, *args, **kwargs):  # pragma: no cover
    file = kwargs.get('file', sys.stderr)
    alt_text = kwargs.get('alt_text', None)
    color = kwargs.get('color', None)
    print(colorize(msg, color, file, alt_text), *args, file=file)


def warning(*args, **kwargs):  # pragma: no cover
    defaults = dict(file=sys.stderr, alt_text='warning: ', color='yellow')
    defaults.update(kwargs)
    message(*args, **defaults)


def error(*args, **kwargs):  # pragma: no cover
    defaults = dict(file=sys.stderr, alt_text='error: ', color='red')
    defaults.update(kwargs)
    message(*args, **defaults)
