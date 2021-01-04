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

from __future__ import print_function, division

import ast
import glob
import math
import os
import re
import sys
import xml.dom.minidom

from copy import deepcopy
from .color import error, warning
from .xmlutils import opt_attrs, reqd_attrs, first_child_element, \
    next_sibling_element, replace_node


_basestr = str
unicode = str
encoding = {}

# Dictionary of substitution args
substitution_args_context = {}


# Stack of currently processed files
filestack = []

# The top level directory
tld = ''


def push_file(filename):
    """
    Push a new filename to the filestack.
    Instead of directly modifying filestack, a deep-copy is created and
    modified,
    while the old filestack is returned.
    This allows to store the filestack that was active when a macro or
    property is defined
    """
    global filestack
    oldstack = filestack
    filestack = deepcopy(filestack)
    filestack.append(filename)
    return oldstack


def restore_filestack(oldstack):
    global filestack
    filestack = oldstack


def abs_filename_spec(filename_spec):
    """
    Prepend the dirname of the currently processed file
    if filename_spec is not yet absolute
    """
    if not os.path.isabs(filename_spec):
        parent_filename = filestack[-1]
        basedir = os.path.dirname(parent_filename) if parent_filename else '.'
        return os.path.join(basedir, filename_spec)
    return filename_spec  # pragma: no cover


class YamlDictWrapper(dict):  # pragma: no cover
    """Wrapper class providing dotted access to dict items"""
    def __getattr__(self, item):
        try:
            result = super(YamlDictWrapper, self).__getitem__(item)
            return YamlDictWrapper(result) \
                if isinstance(result, dict) else result
        except KeyError:
            raise XacroException("No such key: '{}'".format(item))

    __getitem__ = __getattr__


def load_yaml(filename):  # pragma: no cover
    try:
        import yaml
    except Exception:
        raise XacroException("yaml support not available; install python-yaml")

    filename = abs_filename_spec(filename)
    f = open(filename)
    oldstack = push_file(filename)
    try:
        return YamlDictWrapper(yaml.safe_load(f))
    finally:
        f.close()
        restore_filestack(oldstack)
        global all_includes
        all_includes.append(filename)


# global symbols dictionary
# taking simple security measures to forbid access to __builtins__
# only the very few symbols explicitly listed are allowed
# for discussion, see:
# http://nedbatchelder.com/blog/201206/eval_really_is_dangerous.html
global_symbols = {
    '__builtins__': {
        k: __builtins__[k] for k in [
            'list', 'dict', 'map', 'len', 'str', 'float', 'int',
            'True', 'False', 'min', 'max', 'round']}}

# also define all math symbols and functions
global_symbols.update(math.__dict__)
# expose load_yaml and abs_filename
global_symbols.update(
    dict(load_yaml=load_yaml, abs_filename=abs_filename_spec))


class XacroException(Exception):
    """
    XacroException allows to wrap another exception (exc) and to augment
    its error message: prefixing with msg and suffixing with suffix.
    str(e) finally prints: msg str(exc) suffix
    """

    def __init__(self, msg=None, suffix=None, exc=None, macro=None):
        super(XacroException, self).__init__(msg)
        self.suffix = suffix
        self.exc = exc
        self.macros = [] if macro is None else [macro]

    def __str__(self):  # pragma: no cover
        items = [super(XacroException, self).__str__(), self.exc, self.suffix]
        return ' '.join(
            [s for s in [unicode(e) for e in items] if s not in ['', 'None']])


verbosity = 1


def check_attrs(tag, required, optional):
    """
    Helper routine to fetch required and optional attributes
    and complain about any additional attributes.
    :param tag (xml.dom.Element): DOM element node
    :param required [str]: list of required attributes
    :param optional [str]: list of optional attributes
    """
    result = reqd_attrs(tag, required)
    result.extend(opt_attrs(tag, optional))
    allowed = required + optional
    extra = [
        a for a in tag.attributes.keys()
        if a not in allowed and not a.startswith("xmlns:")]
    if extra:  # pragma: no cover
        warning(
            "%s: unknown attribute(s): %s" % (tag.nodeName, ', '.join(extra)))
        if verbosity > 0:
            print_location(filestack)
    return result


class Macro(object):
    def __init__(self):
        self.body = None  # original xml.dom.Node
        self.params = []  # parsed parameter names
        self.defaultmap = {}  # default parameter values
        self.history = []  # definition history


def eval_extension(s):  # pragma: no cover

    if s == '$(cwd)':
        return os.getcwd()

    if s.startswith('$(find'):
        return tld
        # return '..'

    if s.startswith('$(arg'):
        s = s.replace('$(arg ', '')
        s = s[:-1]
        return substitution_args_context['arg'][s]

    try:
        from roslaunch.substitution_args import resolve_args, ArgException
        from rospkg.common import ResourceNotFound
        return resolve_args(
            s, context=substitution_args_context, resolve_anon=False)
    except ImportError:

        return ''
        # raise XacroException("substitution args not supported: ", exc=e)
    except ArgException as e:
        raise XacroException("Undefined substitution argument", exc=e)
    except ResourceNotFound as e:
        raise XacroException("resource not found:", exc=e)


class Table(object):
    def __init__(self, parent=None):
        self.parent = parent
        self.table = {}
        self.unevaluated = set()  # set of unevaluated variables
        # list of currently resolved vars (to resolve recursive definitions)
        self.recursive = []
        # the following variables are for debugging / checking only
        self.depth = self.parent.depth + 1 if self.parent else 0

    @staticmethod
    def _eval_literal(value):
        if isinstance(value, _basestr):
            # remove single quotes from escaped string
            if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
                return value[1:-1]
            # Try to evaluate as number literal or boolean.
            # This is needed to handle numbers in property definitions as
            # numbers, not strings.
            # python3 ignores/drops underscores in number literals
            # (due to PEP515).
            # Here, we want to handle literals with underscores as plain
            # strings.
            if '_' in value:
                return value
            # order of types is important!
            for f in [int, float, lambda x: get_boolean_value(x, None)]:
                try:
                    return f(value)
                except Exception:
                    pass
        return value

    def _resolve_(self, key):
        # lazy evaluation
        if key in self.unevaluated:
            if key in self.recursive:
                raise XacroException("recursive variable definition: %s" %
                                     " -> ".join(self.recursive + [key]))
            self.recursive.append(key)
            self.table[key] = self._eval_literal(
                eval_text(self.table[key], self))
            self.unevaluated.remove(key)
            self.recursive.remove(key)

        # return evaluated result
        value = self.table[key]
        if (verbosity > 2 and self.parent is None) or verbosity > 3:   # pragma: no cover # noqa
            print("{indent}use {key}: {value} ({loc})".format(
                indent=self.depth * ' ', key=key, value=value,
                loc=filestack[-1]), file=sys.stderr)
        return value

    def __getitem__(self, key):
        if key in self.table:
            return self._resolve_(key)
        elif self.parent:
            return self.parent[key]
        else:
            raise KeyError(key)

    def _setitem(self, key, value, unevaluated):
        if key in global_symbols:
            warning("redefining global property: %s" % key)
            print_location(filestack)

        value = self._eval_literal(value)
        self.table[key] = value
        if unevaluated and isinstance(value, _basestr):
            # literal evaluation failed: re-evaluate lazily at first access
            self.unevaluated.add(key)
        elif key in self.unevaluated:   # pragma: no cover
            # all other types cannot be evaluated
            self.unevaluated.remove(key)
        if (verbosity > 2 and self.parent is None) or verbosity > 3:   # pragma: no cover # noqa
            print("{indent}set {key}: {value} ({loc})".format(
                indent=self.depth * ' ', key=key, value=value,
                loc=filestack[-1]), file=sys.stderr)

    def __setitem__(self, key, value):
        self._setitem(key, value, unevaluated=True)

    def __contains__(self, key):
        return \
            key in self.table or \
            (self.parent and key in self.parent)

    def __str__(self):    # pragma: no cover
        s = unicode(self.table)
        if isinstance(self.parent, Table):
            s += "\n  parent: "
            s += unicode(self.parent)
        return s

    def root(self):
        p = self
        while p.parent:
            p = p.parent
        return p


class NameSpace(object):
    # dot access (namespace.property) is forwarded to getitem()
    def __getattr__(self, item):
        return self.__getitem__(item)


class PropertyNameSpace(Table, NameSpace):
    def __init__(self, parent=None):
        super(PropertyNameSpace, self).__init__(parent)


class MacroNameSpace(dict, NameSpace):
    def __init__(self, *args, **kwargs):
        super(MacroNameSpace, self).__init__(*args, **kwargs)


class QuickLexer(object):
    def __init__(self, *args, **kwargs):
        if args:
            # copy attributes + variables from other instance
            other = args[0]
            self.__dict__.update(other.__dict__)
        else:
            self.res = []
            for k, v in kwargs.items():
                self.__setattr__(k, len(self.res))
                self.res.append(re.compile(v))
        self.str = ""
        self.top = None

    def lex(self, str):
        self.str = str
        self.top = None
        self.next()

    def peek(self):
        return self.top

    def next(self):
        result = self.top
        self.top = None
        if not self.str:  # empty string
            return result
        for i in range(len(self.res)):
            m = self.res[i].match(self.str)
            if m:
                self.top = (i, m.group(0))
                self.str = self.str[m.end():]
                return result
        raise XacroException('invalid expression: ' + self.str)


all_includes = []
include_no_matches_msg = \
    """Include tag's filename spec \"{}\" matched "\" no files."""


def get_include_files(filename_spec, symbols):
    try:
        filename_spec = abs_filename_spec(eval_text(filename_spec, symbols))
    except XacroException as e:    # pragma: no cover
        if e.exc and isinstance(e.exc, NameError) and symbols is None:
            raise XacroException(
                'variable filename is supported with in-order option only')
        else:
            raise

    if re.search('[*[?]+', filename_spec):    # pragma: no cover
        # Globbing behaviour
        filenames = sorted(glob.glob(filename_spec))
        if len(filenames) == 0:
            warning(include_no_matches_msg.format(filename_spec))
    else:
        # Default behaviour
        filenames = [filename_spec]

    for filename in filenames:
        global all_includes
        all_includes.append(filename)
        yield filename


def import_xml_namespaces(parent, attributes):
    """import all namespace declarations into parent"""
    for name, value in attributes.items():
        if name.startswith('xmlns:'):
            oldAttr = parent.getAttributeNode(name)
            if oldAttr and oldAttr.value != value:
                warning("inconsistent namespace redefinitions for {name}:"   # pragma: no cover # noqa
                        "\n old: {old}\n new: {new} ({new_file})".format(
                            name=name, old=oldAttr.value, new=value,
                            new_file=filestack[-1]))
            else:
                parent.setAttribute(name, value)


def process_include(elt, macros, symbols, func):
    included = []
    filename_spec, namespace_spec, optional = check_attrs(
        elt, ['filename'], ['ns', 'optional'])
    if namespace_spec:
        try:
            namespace_spec = eval_text(namespace_spec, symbols)
            macros[namespace_spec] = ns_macros = MacroNameSpace()
            symbols[namespace_spec] = ns_symbols = PropertyNameSpace()
            macros = ns_macros
            symbols = ns_symbols
        except TypeError:   # pragma: no cover
            raise XacroException(
                'namespaces are supported with in-order option only')

    optional = get_boolean_value(optional, None)

    if first_child_element(elt):
        warning("Child elements of a <xacro:include> tag are ignored")   # pragma: no cover # noqa
        if verbosity > 0:   # pragma: no cover
            print_location(filestack)

    for filename in get_include_files(filename_spec, symbols):
        try:
            # extend filestack
            oldstack = push_file(filename)
            include = parse(None, filename).documentElement

            # recursive call to func
            func(include, macros, symbols)
            included.append(include)
            import_xml_namespaces(elt.parentNode, include.attributes)
        except XacroException as e:
            if e.exc and isinstance(e.exc, IOError) and optional is True:
                continue
            else:
                raise
        finally:
            # restore filestack
            restore_filestack(oldstack)

    remove_previous_comments(elt)
    # replace the include tag with the nodes of the included file(s)
    replace_node(elt, by=included, content_only=True)


def is_valid_name(name):
    """
    Checks whether name is a valid property or macro identifier.
    With python-based evaluation, we need to avoid name clashes with python
    keywords.
    """
    # Resulting AST of simple identifier is <Module [<Expr <Name "foo">>]>
    try:
        root = ast.parse(name)

        if isinstance(root, ast.Module) and \
           len(root.body) == 1 and isinstance(root.body[0], ast.Expr) and \
           isinstance(
               root.body[0].value, ast.Name) and root.body[0].value.id == name:
            return True
    except SyntaxError:
        pass

    return False


re_macro_arg = re.compile(r'''\s*([^\s:=]+?):?=(\^\|?)?((?:(?:'[^']*')?[^\s'"]*?)*)(?:\s+|$)(.*)''')    # noqa
#                           space   param    :=   ^|   <--      default      -->   space    rest        # noqa


def parse_macro_arg(s):
    """
    parse the first param spec from a macro parameter string s
    accepting the following syntax: <param>[:=|=][^|]<default>
    :param s: param spec string
    :return: param, (forward, default), rest-of-string
             forward will be either param or None (depending on whether
             ^ was specified)
             default will be the default string or None
             If there is no default spec at all, the middle pair will be
             replaced by None
    """
    m = re_macro_arg.match(s)
    if m:
        # there is a default value specified for param
        param, forward, default, rest = m.groups()
        if not default:
            default = None
        return param, (param if forward else None, default), rest
    else:
        # there is no default specified at all
        result = s.lstrip().split(None, 1)
        return result[0], None, result[1] if len(result) > 1 else ''


def grab_macro(elt, macros):
    assert(elt.tagName == 'xacro:macro')
    remove_previous_comments(elt)

    name, params = check_attrs(elt, ['name'], ['params'])
    if name == 'call':
        raise XacroException("Invalid use of macro name 'call'")
    if name.find('.') != -1:   # pragma: no cover
        raise XacroException(
            "macro names must not contain '.' (reserved for "
            "namespaces): %s" % name)
    if name.startswith('xacro:'):
        warning("macro names must not contain prefix 'xacro:': %s" % name)
        name = name[6:]  # drop 'xacro:' prefix

    # fetch existing or create new macro definition
    macro = macros.get(name, Macro())
    # append current filestack to history
    macro.history.append(filestack)
    macro.body = elt

    # parse params and their defaults
    macro.params = []
    macro.defaultmap = {}
    while params:
        param, value, params = parse_macro_arg(params)
        macro.params.append(param)
        if value is not None:
            macro.defaultmap[param] = value  # parameter with default

    macros[name] = macro
    replace_node(elt, by=None)


def grab_property(elt, table):
    assert(elt.tagName == 'xacro:property')
    remove_previous_comments(elt)

    name, value, default, scope = check_attrs(
        elt, ['name'], ['value', 'default', 'scope'])
    if not is_valid_name(name):
        raise XacroException(
            'Property names must be valid python identifiers: ' + name)
    if value is not None and default is not None:   # pragma: no cover
        raise XacroException(
            'Property cannot define both a default and a value: ' + name)

    if default is not None:
        if scope is not None:   # pragma: no cover
            warning(
                "%s: default property value can only be defined "
                "on local scope" % name)
        if name not in table:
            value = default
        else:   # pragma: no cover
            replace_node(elt, by=None)
            return

    if value is None:
        name = '**' + name
        value = elt  # debug

    replace_node(elt, by=None)

    if scope and scope == 'global':
        target_table = table.root()
        unevaluated = False
    elif scope and scope == 'parent':
        if table.parent:
            target_table = table.parent
            unevaluated = False
        else:   # pragma: no cover
            warning("%s: no parent scope at global scope " % name)
            return  # cannot store the value, no reason to evaluate it
    else:
        target_table = table
        unevaluated = True

    if not unevaluated and isinstance(value, _basestr):
        value = eval_text(value, table)

    target_table._setitem(name, value, unevaluated=unevaluated)


LEXER = QuickLexer(
    # multiple $ in a row, followed by { or (
    DOLLAR_DOLLAR_BRACE=r"^\$\$+(\{|\()",
    EXPR=r"^\$\{[^\}]*\}",        # stuff starting with ${
    EXTENSION=r"^\$\([^\)]*\)",   # stuff starting with $(
    # any text w/o $ or  $ following any chars except {($  or  single $
    TEXT=r"[^$]+|\$[^{($]+|\$$")


# evaluate text and return typed value
def eval_text(text, symbols):
    def handle_expr(s):
        try:
            return eval(eval_text(s, symbols), global_symbols, symbols)
        except Exception as e:
            # re-raise as XacroException to add more context
            raise XacroException(exc=e,
                                 suffix=os.linesep
                                 + "when evaluating expression '%s'" % s)

    def handle_extension(s):   # pragma: no cover
        return eval_extension("$(%s)" % eval_text(s, symbols))

    results = []
    lex = QuickLexer(LEXER)
    lex.lex(text)
    while lex.peek():
        id = lex.peek()[0]
        if id == lex.EXPR:
            results.append(handle_expr(lex.next()[1][2:-1]))
        elif id == lex.EXTENSION:   # pragma: no cover
            results.append(handle_extension(lex.next()[1][2:-1]))
        elif id == lex.TEXT:
            results.append(lex.next()[1])
        elif id == lex.DOLLAR_DOLLAR_BRACE:
            results.append(lex.next()[1][1:])
    # return single element as is, i.e. typed
    if len(results) == 1:
        return results[0]
    # otherwise join elements to a string
    else:
        return ''.join(map(unicode, results))


def eval_default_arg(forward_variable, default, symbols, macro):
    if forward_variable is None:
        return eval_text(default, symbols)
    try:
        return symbols[forward_variable]
    except KeyError:   # pragma: no cover
        if default is not None:
            return eval_text(default, symbols)
        else:
            raise XacroException(
                "Undefined property to forward: "
                + forward_variable, macro=macro)


def handle_dynamic_macro_call(node, macros, symbols):
    name, = reqd_attrs(node, ['macro'])
    if not name:   # pragma: no cover
        raise XacroException("xacro:call is missing the 'macro' attribute")
    name = unicode(eval_text(name, symbols))

    # remove 'macro' attribute and rename tag with resolved macro name
    node.removeAttribute('macro')
    node.tagName = 'xacro:' + name
    # forward to handle_macro_call
    handle_macro_call(node, macros, symbols)
    return True


def resolve_macro(fullname, macros):
    # split name into namespaces and real name
    namespaces = fullname.split('.')
    name = namespaces.pop(-1)

    def _resolve(namespaces, name, macros):
        # traverse namespaces to actual macros dict
        for ns in namespaces:
            macros = macros[ns]
        return macros[name]

    # try fullname and (namespaces, name) in this order
    try:
        return _resolve([], fullname, macros)
    except KeyError:
        if namespaces:
            return _resolve(namespaces, name, macros)
        else:
            raise


def handle_macro_call(node, macros, symbols):
    if node.tagName == 'xacro:call':
        return handle_dynamic_macro_call(node, macros, symbols)
    elif not node.tagName.startswith('xacro:'):
        return False  # no macro

    name = node.tagName[6:]  # drop 'xacro:' prefix
    try:
        m = resolve_macro(name, macros)
        body = m.body.cloneNode(deep=True)

    except KeyError:
        raise XacroException("unknown macro name: %s" % node.tagName)

    # Expand the macro
    scoped = Table(symbols)  # new local name space for macro evaluation
    params = m.params[:]  # deep copy macro's params list
    for name, value in node.attributes.items():
        if name not in params:   # pragma: no cover
            raise XacroException(
                "Invalid parameter \"%s\"" % unicode(name), macro=m)
        params.remove(name)
        scoped._setitem(name, eval_text(value, symbols), unevaluated=False)
        node.setAttribute(name, "")  # suppress second evaluation in eval_all()

    # Evaluate block parameters in node
    eval_all(node, macros, symbols)

    # Fetch block parameters, in order
    block = first_child_element(node)
    for param in params[:]:
        if param[0] == '*':
            if not block:   # pragma: no cover
                raise XacroException("Not enough blocks", macro=m)
            params.remove(param)
            scoped[param] = block
            block = next_sibling_element(block)

    if block is not None:   # pragma: no cover
        raise XacroException("Unused block \"%s\"" % block.tagName, macro=m)

    # Try to load defaults for any remaining non-block parameters
    for param in params[:]:
        # block parameters are not supported for defaults
        if param[0] == '*':   # pragma: no cover
            continue

        # get default
        name, default = m.defaultmap.get(param, (None, None))
        if name is not None or default is not None:
            scoped._setitem(
                param, eval_default_arg(
                    name, default, symbols, m), unevaluated=False)
            params.remove(param)

    if params:
        raise XacroException(
            "Undefined parameters [%s]" % ",".join(params), macro=m)

    try:
        eval_all(body, macros, scoped)
    except Exception as e:   # pragma: no cover
        # fill in macro call history for nice error reporting
        if hasattr(e, 'macros'):
            e.macros.append(m)
        else:
            e.macros = [m]
        raise

    # Replaces the macro node with the expansion
    remove_previous_comments(node)
    replace_node(node, by=body, content_only=True)
    return True


def get_boolean_value(value, condition):
    """
    Return a boolean value that corresponds to the given Xacro condition value.
    Values "true", "1" and "1.0" are supposed to be True.
    Values "false", "0" and "0.0" are supposed to be False.
    All other values raise an exception.
    :param value: The value to be evaluated. The value has to already be
        evaluated by Xacro.
    :param condition: The original condition text in the XML.
    :return: The corresponding boolean value, or a Python expression that,
        converted to boolean, corresponds to it.
    :raises ValueError: If the condition value is incorrect.
    """
    try:
        if isinstance(value, _basestr):
            if value == 'true' or value == 'True':
                return True
            elif value == 'false' or value == 'False':
                return False
            else:
                return bool(int(value))
        else:
            return bool(value)
    except Exception:
        raise XacroException(
            "Xacro conditional \"%s\" evaluated to \"%s\", "
            "which is not a boolean expression." % (condition, value))


_empty_text_node = xml.dom.minidom.getDOMImplementation().createDocument(
    None, "dummy", None).createTextNode('\n\n')


def remove_previous_comments(node):
    """remove consecutive comments in front of the xacro-specific node"""
    next = node.nextSibling
    previous = node.previousSibling
    while previous:
        if previous.nodeType == xml.dom.Node.TEXT_NODE and \
                previous.data.isspace() and previous.data.count('\n') <= 1:
            # skip a single empty text node (max 1 newline)
            previous = previous.previousSibling

        if previous and previous.nodeType == xml.dom.Node.COMMENT_NODE:
            comment = previous
            previous = previous.previousSibling
            node.parentNode.removeChild(comment)
        else:
            # insert empty text node to stop removing of comments in future
            # calls actually this moves the singleton instance to the new
            # location
            if next and _empty_text_node != next:
                node.parentNode.insertBefore(_empty_text_node, next)
            return


def eval_all(node, macros, symbols):
    """
    Recursively evaluate node, expanding macros, replacing properties,
    and evaluating expressions
    """
    # evaluate the attributes
    for name, value in node.attributes.items():
        if name.startswith('xacro:'):  # remove xacro:* attributes
            node.removeAttribute(name)
        else:
            result = unicode(eval_text(value, symbols))
            node.setAttribute(name, result)

    # remove xacro namespace definition
    try:
        node.removeAttribute('xmlns:xacro')
    except xml.dom.NotFoundErr:
        pass

    node = node.firstChild
    while node:
        next = node.nextSibling
        if node.nodeType == xml.dom.Node.ELEMENT_NODE:
            if node.tagName == 'xacro:insert_block':
                name, = check_attrs(node, ['name'], [])

                if ("**" + name) in symbols:
                    # Multi-block
                    block = symbols['**' + name]
                    content_only = True
                elif ("*" + name) in symbols:
                    # Single block
                    block = symbols['*' + name]
                    content_only = False
                else:   # pragma: no cover
                    raise XacroException("Undefined block \"%s\"" % name)

                # cloning block allows to insert the same block multiple times
                block = block.cloneNode(deep=True)
                # recursively evaluate block
                eval_all(block, macros, symbols)
                replace_node(node, by=block, content_only=content_only)

            elif node.tagName == 'xacro:include':
                process_include(node, macros, symbols, eval_all)

            elif node.tagName == 'xacro:property':
                grab_property(node, symbols)

            elif node.tagName == 'xacro:macro':
                grab_macro(node, macros)

            elif node.tagName == 'xacro:arg':
                name, default = check_attrs(node, ['name', 'default'], [])
                if name not in substitution_args_context['arg']:
                    substitution_args_context['arg'][name] = eval_text(
                        default, symbols)

                remove_previous_comments(node)
                replace_node(node, by=None)

            elif node.tagName == 'xacro:element':
                name = eval_text(
                    *reqd_attrs(node, ['xacro:name']), symbols=symbols)
                if not name:   # pragma: no cover
                    raise XacroException("xacro:element: empty name")

                node.removeAttribute('xacro:name')
                node.nodeName = node.tagName = name
                continue  # re-process the node with new tagName

            elif node.tagName == 'xacro:attribute':
                name, value = [eval_text(a, symbols) for a in reqd_attrs(
                    node, ['name', 'value'])]
                if not name:   # pragma: no cover
                    raise XacroException("xacro:attribute: empty name")

                node.parentNode.setAttribute(name, value)
                replace_node(node, by=None)

            elif node.tagName in ['xacro:if', 'xacro:unless']:
                remove_previous_comments(node)
                cond, = check_attrs(node, ['value'], [])
                keep = get_boolean_value(eval_text(cond, symbols), cond)
                if node.tagName in ['unless', 'xacro:unless']:
                    keep = not keep

                if keep:
                    eval_all(node, macros, symbols)
                    replace_node(node, by=node, content_only=True)
                else:
                    replace_node(node, by=None)

            elif handle_macro_call(node, macros, symbols):
                # handle_macro_call does all the work of expanding the macro
                pass

            else:
                eval_all(node, macros, symbols)

        # TODO: Also evaluate content of COMMENT_NODEs?
        elif node.nodeType == xml.dom.Node.TEXT_NODE:
            node.data = unicode(eval_text(node.data, symbols))

        node = next


def parse(inp, filename=None):
    """
    Parse input or filename into a DOM tree.
    If inp is None, open filename and load from there.
    Otherwise, parse inp, either as string or file object.
    If inp is already a DOM tree, this function is a noop.
    :return:xml.dom.minidom.Document
    :raise: xml.parsers.expat.ExpatError
    """

    f = None
    if inp is None:
        try:
            inp = f = open(filename)
        except IOError as e:
            # do not report currently processed file as "in file ..."
            filestack.pop()
            raise XacroException(e.strerror + ": " + e.filename, exc=e)

    try:
        if isinstance(inp, _basestr):
            return xml.dom.minidom.parseString(inp)
        elif hasattr(inp, 'read'):
            return xml.dom.minidom.parse(inp)
        return inp   # pragma: no cover

    finally:
        if f:
            f.close()


def process_doc(doc, mappings=None, **kwargs):
    global verbosity
    verbosity = kwargs.get('verbosity', verbosity)

    # set substitution args
    substitution_args_context['arg'] = {} if mappings is None else mappings

    # if not yet defined: initialize filestack
    if not filestack:
        restore_filestack([None])

    macros = {}
    symbols = Table()

    # apply xacro:targetNamespace as global xmlns (if defined)
    targetNS = doc.documentElement.getAttribute('xacro:targetNamespace')
    if targetNS:
        doc.documentElement.removeAttribute('xacro:targetNamespace')
        doc.documentElement.setAttribute('xmlns', targetNS)

    eval_all(doc.documentElement, macros, symbols)

    # reset substitution args
    substitution_args_context['arg'] = {}


def open_output(output_filename):
    if output_filename is None:   # pragma: no cover
        return sys.stdout
    else:   # pragma: no cover
        dir_name = os.path.dirname(output_filename)
        if dir_name:
            try:
                os.makedirs(dir_name)
            except os.error:
                # errors occur when dir_name exists or creation failed
                # ignore error here; opening of file will fail if directory
                # is still missing
                pass

        try:
            return open(output_filename, 'w')
        except IOError as e:
            raise XacroException("Failed to open output:", exc=e)


def print_location(filestack, err=None, file=sys.stderr):
    macros = getattr(err, 'macros', []) if err else []
    msg = 'when instantiating macro:'
    for m in macros:   # pragma: no cover
        name = m.body.getAttribute('name')
        location = '(%s)' % m.history[-1][-1]
        print(msg, name, location, file=file)
        msg = 'instantiated from:'

    msg = 'in file:' if macros else 'when processing file:'
    for f in reversed(filestack):
        if f is None:
            f = 'string'
        print(msg, f, file=file)
        msg = 'included from:'


def process_file(input_file_name, **kwargs):   # pragma: no cover
    """main processing pipeline"""
    # initialize file stack for error-reporting
    restore_filestack([input_file_name])
    # parse the document into a xml.dom tree
    doc = parse(None, input_file_name)
    # perform macro replacement
    process_doc(doc, **kwargs)

    # add xacro auto-generated banner
    banner = [xml.dom.minidom.Comment(c) for c in
              [" %s " % ('=' * 83),
               " |    This document was autogenerated by xacro from %-30s | "
               % input_file_name,
               " |    EDITING THIS FILE BY HAND IS NOT RECOMMENDED  %-30s | "
               % "",
               " %s " % ('=' * 83)]]
    first = doc.firstChild
    for comment in banner:
        doc.insertBefore(comment, first)

    return doc


def main(filename, tld_other=None):   # pragma: no cover
    opts = {
        'output': None,
        'just_deps': False,
        'in_order': True,
        'verbosity': 1,
        'just_includes': False,
        'mappings': {}
    }

    global tld

    if tld_other is None:
        tld = '..'
    else:
        tld = tld_other

    try:
        # open and process file
        doc = process_file(filename, **opts)
        # open the output file
        out = open_output(opts["output"])

    except Exception as e:
        msg = unicode(e)
        if not msg:
            msg = repr(e)
        error(msg)
        if verbosity > 0:
            print_location(filestack, e)
        if verbosity > 1:
            print(file=sys.stderr)  # add empty separator line before error
            raise  # create stack trace
        else:
            sys.exit(2)  # gracefully exit with error condition

    # special output mode
    if opts["just_deps"]:
        out.write(" ".join(set(all_includes)))
        print()
        return

    # write output
    # out.write(doc.toprettyxml(indent='  ', **encoding))
    # print()
    # only close output file, but not stdout
    if opts["output"]:
        out.close()

    return doc.toprettyxml(indent='  ', **encoding)
