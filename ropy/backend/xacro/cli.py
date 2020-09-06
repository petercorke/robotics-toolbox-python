# Copyright (c) 2015, Open Source Robotics Foundation, Inc.
# Copyright (c) 2013, Willow Garage, Inc.
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
#     * Neither the name of the Open Source Robotics Foundation, Inc.
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

# Authors: Stuart Glaser, William Woodall, Robert Haschke
# Maintainer: Morgan Quigley <morgan@osrfoundation.org>

import textwrap
from optparse import OptionParser, IndentedHelpFormatter
from .color import colorize, message


class ColoredOptionParser(OptionParser):   # pragma: no cover
    def error(self, message):
        msg = colorize(message, 'red')
        OptionParser.error(self, msg)


_original_wrap = textwrap.wrap


def wrap_with_newlines(text, width, **kwargs):  # pragma: no cover
    result = []
    for paragraph in text.split('\n'):
        result.extend(_original_wrap(paragraph, width, **kwargs))
    return result


class IndentedHelpFormatterWithNL(IndentedHelpFormatter):  # pragma: no cover
    def __init__(self, *args, **kwargs):
        IndentedHelpFormatter.__init__(self, *args, **kwargs)

    def format_option(self, text):
        textwrap.wrap, old = wrap_with_newlines, textwrap.wrap
        result = IndentedHelpFormatter.format_option(self, text)
        textwrap.wrap = old
        return result


def process_args(argv, require_input=True):   # pragma: no cover
    parser = ColoredOptionParser(usage="usage: %prog [options] <input>",
                                 formatter=IndentedHelpFormatterWithNL())
    parser.add_option("-o", dest="output", metavar="FILE",
                      help="write output to FILE instead of stdout")
    parser.add_option("--deps", action="store_true", dest="just_deps",
                      help="print file dependencies")
    parser.add_option(
        "--inorder", "-i", action="store_true", dest="in_order",
        help="processing in read order (default, can be omitted)")

    # verbosity options
    parser.add_option("-q", action="store_const", dest="verbosity", const=0,
                      help="quiet operation, suppressing warnings")
    parser.add_option("-v", action="count", dest="verbosity",
                      help="increase verbosity")
    parser.add_option("--verbosity", metavar='level', dest="verbosity",
                      type='int',
                      help=textwrap.dedent("""\
                      set verbosity level
                      0: quiet, suppressing warnings
                      1: default, showing warnings and error locations
                      2: show stack trace
                      3: log property definitions and usage on top level
                      4: log property definitions and usage on all levels"""))

    mappings = {}
    filtered_args = argv

    parser.set_defaults(just_deps=False, just_includes=False, verbosity=1)
    (options, pos_args) = parser.parse_args(filtered_args)
    if options.in_order:
        message(
            "xacro: in-order processing became default in ROS Melodic. "
            "You can drop the option.")
    options.in_order = True

    if len(pos_args) != 1:
        if require_input:
            parser.error("expected exactly one input file as argument")
        else:
            pos_args = [None]

    options.mappings = mappings
    return options, pos_args[0]
