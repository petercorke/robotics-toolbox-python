#! /usr/bin/env python
# -*- coding: utf-8 -*-

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
# Maintainer: Robert Haschke <rhaschke@techfak.uni-bielefeld.de>

from __future__ import print_function

import ast
from contextlib import contextmanager
import os.path
import re
import subprocess
import sys
import unittest
from roboticstoolbox.tools import xacro
from roboticstoolbox.tools.xacro.cli import process_args
import xml.dom
from xml.dom.minidom import parseString

try:
    from cStringIO import StringIO  # Python 2.x
except ImportError:
    from io import StringIO  # Python 3.x

# regex to match whitespace
whitespace = re.compile(r'\s+')


def text_values_match(a, b):
    # generic comparison
    if whitespace.sub(' ', a).strip() == whitespace.sub(' ', b).strip():
        return True

    try:  # special handling of dicts: ignore order
        a_dict = ast.literal_eval(a)
        b_dict = ast.literal_eval(b)
        if (
              isinstance(a_dict, dict)
              and isinstance(b_dict, dict)
              and a_dict == b_dict):
            return True
    except Exception:  # Attribute values aren't dicts
        pass

    # on failure, try to split a and b at whitespace and compare snippets
    def match_splits(a_, b_):   # pragma: no cover
        if len(a_) != len(b_):
            return False
        for a, b in zip(a_, b_):
            if a == b:
                continue
            try:  # compare numeric values only up to some accuracy
                if abs(float(a) - float(b)) > 1.0e-9:
                    return False
            except ValueError:  # values aren't numeric and not identical
                return False
        return True

    return match_splits(a.split(), b.split())


def all_attributes_match(a, b):
    if len(a.attributes) != len(b.attributes):   # pragma: no cover
        print('Different number of attributes')
        return False
    a_atts = a.attributes.items()
    b_atts = b.attributes.items()
    a_atts.sort()
    b_atts.sort()

    for a, b in zip(a_atts, b_atts):
        if a[0] != b[0]:   # pragma: no cover
            print('Different attribute names: %s and %s' % (a[0], b[0]))
            return False
        if not text_values_match(a[1], b[1]):   # pragma: no cover
            print('Different attribute values: %s and %s' % (a[1], b[1]))
            return False
    return True


def text_matches(a, b):
    if text_values_match(a, b):
        return True
    print("Different text values: '%s' and '%s'" % (a, b))
    return False


def nodes_match(a, b, ignore_nodes):
    if not a and not b:
        return True
    if not a or not b:    # pragma: no cover
        return False

    if a.nodeType != b.nodeType:    # pragma: no cover
        print('Different node types: %s and %s' % (a, b))
        return False

    # compare text-valued nodes
    if a.nodeType in [xml.dom.Node.TEXT_NODE,
                      xml.dom.Node.CDATA_SECTION_NODE,
                      xml.dom.Node.COMMENT_NODE]:
        return text_matches(a.data, b.data)

    # ignore all other nodes except ELEMENTs
    if a.nodeType != xml.dom.Node.ELEMENT_NODE:    # pragma: no cover
        return True

    # compare ELEMENT nodes
    if a.nodeName != b.nodeName:    # pragma: no cover
        print('Different element names: %s and %s' % (a.nodeName, b.nodeName))
        return False

    if not all_attributes_match(a, b):    # pragma: no cover
        return False

    a = a.firstChild
    b = b.firstChild
    while a or b:
        # ignore whitespace-only text nodes
        # we could have several text nodes in a row, due to replacements
        while (a and
               ((a.nodeType in ignore_nodes) or
                (
                  a.nodeType == xml.dom.Node.TEXT_NODE
                  and whitespace.sub('', a.data) == ""))):
            a = a.nextSibling
        while (b and
               ((b.nodeType in ignore_nodes) or
                (
                  b.nodeType == xml.dom.Node.TEXT_NODE
                  and whitespace.sub('', b.data) == ""))):
            b = b.nextSibling

        if not nodes_match(a, b, ignore_nodes):    # pragma: no cover
            return False

        if a:
            a = a.nextSibling
        if b:
            b = b.nextSibling

    return True


def xml_matches(a, b, ignore_nodes=[]):
    if isinstance(a, str):
        return xml_matches(parseString(a).documentElement, b, ignore_nodes)
    if isinstance(b, str):
        return xml_matches(a, parseString(b).documentElement, ignore_nodes)
    if a.nodeType == xml.dom.Node.DOCUMENT_NODE:
        return xml_matches(a.documentElement, b, ignore_nodes)
    if b.nodeType == xml.dom.Node.DOCUMENT_NODE:
        return xml_matches(a, b.documentElement, ignore_nodes)

    if not nodes_match(a, b, ignore_nodes):    # pragma: no cover
        print('Match failed:')
        a.writexml(sys.stdout)
        print()
        print('=' * 78)
        b.writexml(sys.stdout)
        print()
        return False
    return True


# capture output going to file=sys.stdout | sys.stderr
@contextmanager
def capture_stderr(function, *args, **kwargs):
    # temporarily replace sys.stderr with StringIO()
    old, sys.stderr = sys.stderr, StringIO()
    result = function(*args, **kwargs)
    sys.stderr.seek(0)
    yield (result, sys.stderr.read())
    sys.stderr = old  # restore sys.stderr


class TestMatchXML(unittest.TestCase):
    def test_normalize_whitespace_text(self):
        self.assertTrue(text_matches("", " \t\n\r"))

    def test_normalize_whitespace_trim(self):
        self.assertTrue(text_matches(" foo bar ", "foo \t\n\r bar"))

    def test_match_similar_numbers(self):
        self.assertTrue(text_matches("0.123456789", "0.123456788"))

    def test_mismatch_different_numbers(self):
        self.assertFalse(text_matches("0.123456789", "0.1234567879"))

    def test_match_unordered_dicts(self):
        self.assertTrue(
          text_matches("{'a': 1, 'b': 2, 'c': 3}", "{'c': 3, 'b': 2, 'a': 1}"))

    def test_mismatch_different_dicts(self):
        self.assertFalse(
          text_matches("{'a': 1, 'b': 2, 'c': 3}", "{'c': 3, 'b': 2, 'a': 0}"))

    def test_empty_node_vs_whitespace(self):
        self.assertTrue(xml_matches('''<foo/>''', '''<foo> \t\n\r </foo>'''))

    def test_whitespace_vs_empty_node(self):
        self.assertTrue(xml_matches('''<foo> \t\n\r </foo>''', '''<foo/>'''))

    def test_normalize_whitespace_nested(self):
        self.assertTrue(xml_matches(
          '''<a><b/></a>''', '''<a>\n<b> </b> </a>'''))

    def test_ignore_comments(self):
        self.assertTrue(xml_matches(
          '''<a><b/><!-- foo --> <!-- bar --></a>''',
          '''<a><b/></a>''', [xml.dom.Node.COMMENT_NODE]))


class TestXacroFunctions(unittest.TestCase):
    def test_is_valid_name(self):
        self.assertTrue(xacro.is_valid_name("_valid_name_123"))
        # syntactically correct keyword
        self.assertFalse(xacro.is_valid_name('pass'))
        self.assertFalse(xacro.is_valid_name('foo '))     # trailing whitespace
        self.assertFalse(xacro.is_valid_name(' foo'))     # leading whitespace
        self.assertFalse(xacro.is_valid_name('1234'))     # number
        self.assertFalse(xacro.is_valid_name('1234abc'))  # number and letters
        self.assertFalse(xacro.is_valid_name(''))         # empty string
        self.assertFalse(xacro.is_valid_name('   '))      # whitespace only
        self.assertFalse(xacro.is_valid_name('foo bar'))  # several tokens
        self.assertFalse(xacro.is_valid_name('no-dashed-names-for-you'))
        # dot separates fields
        self.assertFalse(xacro.is_valid_name('invalid.too'))

    def test_resolve_macro(self):
        # define three nested macro dicts with the same macro names (keys)
        content = {'simple': 'simple'}
        ns2 = dict({k: v + '2' for k, v in content.items()})
        ns1 = dict({k: v + '1' for k, v in content.items()})
        ns1.update(ns2=ns2)
        macros = dict(content)
        macros.update(ns1=ns1)

        self.assertEqual(xacro.resolve_macro('simple', macros), 'simple')
        self.assertEqual(xacro.resolve_macro('ns1.simple', macros), 'simple1')
        self.assertEqual(xacro.resolve_macro(
          'ns1.ns2.simple', macros), 'simple2')

    def check_macro_arg(self, s, param, forward, default, rest):
        p, v, r = xacro.parse_macro_arg(s)
        self.assertEqual(
          p, param, msg="'{0}' != '{1}' parsing {2}".format(p, param, s))
        if forward or default:
            self.assertTrue(v is not None)
            self.assertEqual(
              v[0], forward, msg="'{0}' != '{1}' parsing {2}".format(
                v[0], forward, s))
            self.assertEqual(
              v[1], default, msg="'{0}' != '{1}' parsing {2}".format(
                v[1], default, s))
        else:
            self.assertTrue(v is None)
        self.assertEqual(
          r, rest, msg="'{0}' != '{1}' parsing {2}".format(r, rest, s))

    def test_parse_macro_arg(self):
        for forward in ['', '^', '^|']:
            defaults = ['', "f('some string','some other')", "f('a b')"]
            if forward == '^':
                defaults = ['']
            for default in defaults:
                seps = ['=', ':='] if forward or default else ['']
                for sep in seps:
                    for rest in ['', ' ', ' bar', ' bar=42']:
                        s = 'foo{0}{1}{2}{3}'.format(
                          sep, forward, default, rest)
                        self.check_macro_arg(
                          s, 'foo', 'foo' if forward else None,
                          default if default else None,
                          rest.lstrip())

    def test_parse_macro_whitespace(self):
        for ws in ['  ', ' \t ', ' \n ']:
            self.check_macro_arg(
              ws + 'foo' + ws + 'bar=42' + ws,
              'foo', None, None, 'bar=42' + ws)


# base class providing some convenience functions
class TestXacroBase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestXacroBase, self).__init__(*args, **kwargs)
        self.ignore_nodes = []

    def assert_matches(self, a, b):
        self.assertTrue(xml_matches(a, b, self.ignore_nodes))

    def quick_xacro(self, xml, cli=None, **kwargs):
        args = {}
        if cli:  # pragma nocover
            opts, _ = process_args(cli, require_input=False)
            args.update(vars(opts))  # initialize with cli args
        args.update(kwargs)  # explicit function args have highest priority

        doc = xacro.parse(xml)
        xacro.process_doc(doc, **args)
        return doc

    def run_xacro(self, input_path, *args):   # pragma nocover
        args = list(args)
        subprocess.call(['xacro', input_path] + args)


# class to match XML docs while ignoring any comments
class TestXacroCommentsIgnored(TestXacroBase):
    def __init__(self, *args, **kwargs):
        super(TestXacroCommentsIgnored, self).__init__(*args, **kwargs)
        self.ignore_nodes = [xml.dom.Node.COMMENT_NODE]

    def test_pr2(self):
        # run xacro on the pr2 tree snapshot
        test_dir = os.path.abspath(os.path.dirname(__file__))
        print(test_dir)
        pr2_xacro_path = os.path.join(
          test_dir, 'robots', 'pr2', 'pr2.urdf.xacro')
        pr2_golden_parse_path = os.path.join(
          test_dir, 'robots', 'pr2', 'pr2_1.11.4.xml')
        self.assert_matches(
            xml.dom.minidom.parse(pr2_golden_parse_path),
            self.quick_xacro(open(pr2_xacro_path)))


# standard test class (including the test from TestXacroCommentsIgnored)
class TestXacro(TestXacroCommentsIgnored):
    def __init__(self, *args, **kwargs):
        super(TestXacroCommentsIgnored, self).__init__(*args, **kwargs)
        self.ignore_nodes = []

    def test_invalid_property_name(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
        <xacro:property name="invalid.name"/></a>'''
        self.assertRaises(xacro.XacroException, self.quick_xacro, src)

    def test_dynamic_macro_names(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="foo"><a>foo</a></xacro:macro>
  <xacro:macro name="bar"><b>bar</b></xacro:macro>
  <xacro:property name="var" value="%s"/>
  <xacro:call macro="${var}"/></a>'''
        res = '''<a>%s</a>'''
        self.assert_matches(self.quick_xacro(src % "foo"), res % "<a>foo</a>")
        self.assert_matches(self.quick_xacro(src % "bar"), res % "<b>bar</b>")

    def test_dynamic_macro_name_clash(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="foo"><a name="foo"/></xacro:macro>
  <xacro:macro name="call"><a name="bar"/></xacro:macro>
  <xacro:call macro="foo"/></a>'''
        self.assertRaises(xacro.XacroException, self.quick_xacro, src)

    def test_dynamic_macro_undefined(self):
        self.assertRaises(xacro.XacroException,
                          self.quick_xacro,
                          '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
                          <xacro:call macro="foo"/></a>''')

    def test_macro_undefined(self):
        self.assertRaises(
          xacro.XacroException,
          self.quick_xacro,
          '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
          <xacro:undefined><foo/><bar/></xacro:undefined></a>''')

    def test_xacro_element(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="foo" params="name"><xacro:element xacro:name="${name}"/></xacro:macro>
  <xacro:foo name="A"/>
  <xacro:foo name="B"/>
</a>'''  # noqa
        res = '''<a><A/><B/></a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_xacro_attribute(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="foo" params="name value">
  <tag><xacro:attribute name="${name}" value="${value}"/></tag>
  </xacro:macro>
  <xacro:foo name="A" value="foo"/>
  <xacro:foo name="B" value="bar"/>
</a>'''
        res = '''<a><tag A="foo"/><tag B="bar"/></a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_inorder_processing(self):
        src = '''
<xml xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="foo" value="1.0"/>
  <xacro:macro name="m" params="foo"><a foo="${foo}"/></xacro:macro>
  <xacro:m foo="1 ${foo}"/>
  <!-- now redefining the property and macro -->
  <xacro:property name="foo" value="2.0"/>
  <xacro:macro name="m" params="foo"><b bar="${foo}"/></xacro:macro>
  <xacro:m foo="2 ${foo}"/>
</xml>'''
        expected = '''
<xml>
  <a foo="1 1.0"/>
  <b bar="2 2.0"/>
</xml>
'''
        self.assert_matches(self.quick_xacro(src), expected)

    def test_should_replace_before_macroexpand(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:macro name="inner" params="*the_block">
  <in_the_inner><xacro:insert_block name="the_block" /></in_the_inner>
</xacro:macro>
<xacro:macro name="outer" params="*the_block">
  <in_the_outer><xacro:inner><xacro:insert_block name="the_block" /></xacro:inner></in_the_outer>
</xacro:macro>
<xacro:outer><woot /></xacro:outer></a>'''  # noqa
        res = '''<a><in_the_outer><in_the_inner><woot /></in_the_inner></in_the_outer></a>'''  # noqa
        self.assert_matches(self.quick_xacro(src), res)

    def test_evaluate_macro_params_before_body(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="foo" params="lst">${lst[-1]}</xacro:macro>
  <xacro:foo lst="${[1,2,3]}"/></a>'''
        self.assert_matches(self.quick_xacro(src), '''<a>3</a>''')

    def test_macro_params_escaped_string(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
    <xacro:macro name="foo" params="a='1 -2' c=3"><bar a="${a}" c="${c}"/></xacro:macro>
    <xacro:foo/></a>'''  # noqa
        self.assert_matches(self.quick_xacro(src), '''<a><bar a="1 -2" c="3"/></a>''')   # noqa

    def test_property_replacement(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="foo" value="42" />
  <the_foo result="${foo}" />
</a>'''
        res = '''<a><the_foo result="42"/></a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_property_scope_parent(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="foo" params="factor">
  <xacro:property name="foo" value="${21*factor}" scope="parent"/>
  </xacro:macro>
  <xacro:foo factor="2"/><a foo="${foo}"/></a>'''
        self.assert_matches(self.quick_xacro(src), '''<a><a foo="42"/></a>''')

    def test_property_scope_global(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="foo" params="factor">
    <xacro:macro name="bar">
      <xacro:property name="foo" value="${21*factor}" scope="global"/>
    </xacro:macro>
    <xacro:bar/>
  </xacro:macro>
  <xacro:foo factor="2"/><a foo="${foo}"/></a>'''
        self.assert_matches(self.quick_xacro(src), '''<a><a foo="42"/></a>''')

    def test_math_ignores_spaces(self):
        src = '''<a><f v="${0.9 / 2 - 0.2}" /></a>'''
        self.assert_matches(self.quick_xacro(src), '''<a><f v="0.25" /></a>''')

    # def test_substitution_args_find(self):
    #     self.assert_matches(self.quick_xacro('''<a><f v="$(find xacro)/test/test_xacro.py" /></a>'''),   # noqa
    #             '''<a><f v="''' + os.path.abspath((__file__).replace(".pyc",".py") + '''" /></a>'''))   # noqa

    # def test_substitution_args_arg(self):
    #     res = '''<a><f v="my_arg" /></a>'''
    #     self.assert_matches(self.quick_xacro('''<a><f v="$(arg sub_arg)" /></a>''', cli=['sub_arg:=my_arg']), res)   # noqa

    def test_escaping_dollar_braces(self):
        src = '''<a b="$${foo}" c="$$${foo}" d="text $${foo}" e="text $$${foo}" f="$$(pwd)" />'''   # noqa
        res = '''<a b="${foo}" c="$${foo}" d="text ${foo}" e="text $${foo}" f="$(pwd)" />'''   # noqa
        self.assert_matches(self.quick_xacro(src), res)

    def test_just_a_dollar_sign(self):
        src = '''<a b="$" c="text $" d="text $ text"/>'''
        self.assert_matches(self.quick_xacro(src), src)

    def test_multiple_insert_blocks(self):
        self.assert_matches(self.quick_xacro('''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:macro name="foo" params="*block">
  <xacro:insert_block name="block" />
  <xacro:insert_block name="block" />
</xacro:macro>
<xacro:foo>
  <a_block />
</xacro:foo>
</a>'''), '''<a>
  <a_block />
  <a_block />
</a>''')

    def test_multiple_blocks(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:macro name="foo" params="*block{A} *block{B}">
  <xacro:insert_block name="block1" />
  <xacro:insert_block name="block2" />
</xacro:macro>
<xacro:foo>
  <block1/>
  <block2/>
</xacro:foo>
</a>'''
        res = '''<a>
<block{A}/>
<block{B}/>
</a>'''
        # test both, reversal and non-reversal of block order
        for d in [dict(A='1', B='2'), dict(A='2', B='1')]:
            self.assert_matches(
              self.quick_xacro(src.format(**d)), res.format(**d))

    def test_integer_stays_integer(self):
        self.assert_matches(self.quick_xacro('''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:macro name="m" params="num">
  <test number="${num}" />
</xacro:macro>
<xacro:m num="100" />
</a>'''), '''
<a>
  <test number="100" />
</a>''')

    def test_insert_block_property(self):
        self.assert_matches(self.quick_xacro('''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:macro name="bar">bar</xacro:macro>
<xacro:property name="val" value="2" />
<xacro:property name="some_block">
  <some_block attr="${val}"><xacro:bar/></some_block>
</xacro:property>
<foo>
  <xacro:insert_block name="some_block" />
</foo>
</a>'''), '''
<a>
<foo><some_block attr="2">bar</some_block></foo>
</a>''')

    # def test_include(self):
    #     src = '''<a xmlns:xacro="http://www.ros.org/xacro"><xacro:include filename="include1.xml"/></a>'''   # noqa
    #     self.assert_matches(self.quick_xacro(src), '''<a><inc1/></a>''')

    # def test_include_glob(self):
    #     src = '''<a xmlns:xacro="http://www.ros.org/xacro"><xacro:include filename="include{glob}.xml"/></a>'''  # noqa
    #     res = '<a><inc1/><inc2/></a>'
    #     for pattern in ['*', '?', '[1-2]']:
    #         self.assert_matches(self.quick_xacro(src.format(glob=pattern)), res)  # noqa

    def test_include_nonexistent(self):
        self.assertRaises(xacro.XacroException,
                          self.quick_xacro, '''<a xmlns:xacro="http://www.ros.org/xacro">
                             <xacro:include filename="include-nada.xml" /></a>''')  # noqa

    def test_include_deprecated(self):
        # <include> tags with some non-trivial content should not issue the deprecation warning  # noqa
        src = '''<a><include filename="nada"><tag/></include></a>'''
        with capture_stderr(self.quick_xacro, src) as (result, output):
            self.assert_matches(result, src)
            self.assertEqual(output, '')

    def test_include_from_variable(self):
        doc = '''<a xmlns:xacro="http://www.ros.org/xacro">
        <xacro:property name="file" value="roboticstoolbox/tools/xacro/tests/include1.xml"/>
        <xacro:include filename="${file}" /></a>'''  # noqa
        self.assert_matches(self.quick_xacro(doc), '''<a><inc1/></a>''')

    def test_include_with_namespace(self):
        src = '''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="var" value="main"/>
  <xacro:include filename="roboticstoolbox/tools/xacro/tests/include1.xacro" ns="A"/>
  <xacro:include filename="roboticstoolbox/tools/xacro/tests/include2.xacro" ns="B"/>
  <xacro:A.foo/><xacro:B.foo/>
  <main var="${var}" A="${2*A.var}" B="${B.var+1}"/>
</a>'''  # noqa
        res = '''
<a>
    <inc1/><inc2/><main var="main" A="2" B="3"/>
</a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_boolean_if_statement(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:if value="false">
    <a />
  </xacro:if>
  <xacro:if value="true">
    <b />
  </xacro:if>
</robot>'''), '''
<robot>
    <b />
</robot>''')

    def test_invalid_if_statement(self):
        self.assertRaises(xacro.XacroException,
                          self.quick_xacro,
                          '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
                          <xacro:if value="nonsense"><foo/></xacro:if></a>''')

    def test_integer_if_statement(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:if value="${0*42}">
    <a />
  </xacro:if>
  <xacro:if value="0">
    <b />
  </xacro:if>
  <xacro:if value="${0}">
    <c />
  </xacro:if>
  <xacro:if value="${1*2+3}">
    <d />
  </xacro:if>
</robot>'''), '''
<robot>
    <d />
</robot>''')

    def test_float_if_statement(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:if value="${3*0.0}">
    <a />
  </xacro:if>
  <xacro:if value="${3*0.1}">
    <b />
  </xacro:if>
</robot>'''), '''
<robot>
    <b />
</robot>''')

    def test_property_if_statement(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="condT" value="${True}"/>
  <xacro:property name="condF" value="${False}"/>
  <xacro:if value="${condF}"><a /></xacro:if>
  <xacro:if value="${condT}"><b /></xacro:if>
  <xacro:if value="${True}"><c /></xacro:if>
</robot>'''), '''
<robot>
    <b /><c />
</robot>''')

    def test_consecutive_if(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:if value="1"><xacro:if value="0"><a>bar</a></xacro:if></xacro:if>
</a>'''), '''<a/>''')

    def test_equality_expression_in_if_statement(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="var" value="useit"/>
  <xacro:if value="${var == 'useit'}"><foo>bar</foo></xacro:if>
  <xacro:if value="${'use' in var}"><bar>foo</bar></xacro:if>
</a>'''), '''
<a>
  <foo>bar</foo>
  <bar>foo</bar>
</a>''')

    def test_no_evaluation(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="xyz" value="5 -2"/>
  <foo>${xyz}</foo>
</a>'''), '''
<a>
  <foo>5 -2</foo>
</a>''')

    def test_math_expressions(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <foo function="${1. + sin(pi)}"/>
</a>'''), '''
<a>
  <foo function="1.0"/>
</a>''')

    def test_consider_non_elements_if(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:if value="1"><!-- comment --> text <b>bar</b></xacro:if>
</a>'''), '''
<a><!-- comment --> text <b>bar</b></a>''')

    def test_consider_non_elements_block(self):
        self.assert_matches(
            self.quick_xacro('''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:macro name="foo" params="*block">
  <!-- comment -->
  foo
  <xacro:insert_block name="block" />
</xacro:macro>
<xacro:foo>
  <!-- ignored comment -->
  ignored text
  <a_block />
</xacro:foo>
</a>'''), '''
<a>
  <!-- comment -->
  foo
  <a_block />
</a>''')

    def test_ignore_xacro_comments(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- A -->

  <!-- ignore multiline comments before any xacro tag -->
  <!-- ignored -->
  <xacro:property name="foo" value="1"/>
  <!-- ignored -->
  <xacro:if value="1"><!-- B --></xacro:if>
  <!-- ignored -->
  <xacro:macro name="foo"><!-- C --></xacro:macro>
  <!-- ignored -->
  <xacro:foo/>
</a>'''), '''
<a><!-- A --><!-- B --><!-- C --></a>''')

    def test_recursive_evaluation(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="a" value=" 42 "/>
  <xacro:property name="a2" value="${ 2 * a }"/>
  <a doubled="${a2}"/>
</robot>'''), '''
<robot>
  <a doubled="84"/>
</robot>''')

    def test_recursive_evaluation_wrong_order(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="a2" value="${2*a}"/>
  <xacro:property name="a" value="42"/>
  <a doubled="${a2}"/>
</robot>'''), '''
<robot>
  <a doubled="84"/>
</robot>''')

    def test_recursive_definition(self):
        self.assertRaises(xacro.XacroException,
                          self.quick_xacro, '''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="a" value="${a2}"/>
  <xacro:property name="a2" value="${2*a}"/>
  <a doubled="${a2}"/>
</robot>''')

    def test_multiple_recursive_evaluation(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="a" value="1"/>
  <xacro:property name="b" value="2"/>
  <xacro:property name="c" value="3"/>
  <xacro:property name="product" value="${a*b*c}"/>
  <answer product="${product}"/>
</robot>'''), '''
<robot>
  <answer product="6"/>
</robot>''')

    def test_multiple_definition_and_evaluation(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="a" value="42"/>
  <xacro:property name="b" value="${a}"/>
  <xacro:property name="b" value="${-a}"/>
  <xacro:property name="b" value="${a}"/>
  <answer b="${b} ${b} ${b}"/>
</robot>'''), '''
<robot>
  <answer b="42 42 42"/>
</robot>''')

    def test_transitive_evaluation(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="a" value="42"/>
  <xacro:property name="b" value="${a}"/>
  <xacro:property name="c" value="${b}"/>
  <xacro:property name="d" value="${c}"/>
  <answer d="${d}"/>
</robot>'''), '''
<robot>
  <answer d="42"/>
</robot>''')

    def test_multi_tree_evaluation(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="a" value="42"/>
  <xacro:property name="b" value="2.1"/>
  <xacro:property name="c" value="${a}"/>
  <xacro:property name="d" value="${b}"/>
  <xacro:property name="f" value="${c*d}"/>
  <answer f="${f}"/>
</robot>'''), '''
<robot>
  <answer f="88.2"/>
</robot>''')

    def test_from_issue(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="x" value="42"/>
  <xacro:property name="wheel_width" value="${x}"/>
  <link name="my_link">
    <origin xyz="0 0 ${wheel_width/2}"/>
  </link>
</robot>'''), '''
<robot>
  <link name="my_link">
    <origin xyz="0 0 21.0"/>
  </link>
</robot>''')

    def test_recursive_bad_math(self):
        self.assertRaises(xacro.XacroException, self.quick_xacro, '''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="x" value="0"/>
  <tag badness="${1/x}"/>
</robot>''')

    def test_default_param(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="fixed_link" params="parent_link:=base_link child_link *joint_pose">
    <link name="${child_link}"/>
    <joint name="${child_link}_joint" type="fixed">
      <xacro:insert_block name="joint_pose" />
      <parent link="${parent_link}"/>
      <child link="${child_link}" />
    </joint>
  </xacro:macro>
  <xacro:fixed_link child_link="foo">
    <origin xyz="0 0 0" rpy="0 0 0" />
  </xacro:fixed_link >
</robot>'''), '''
<robot>
  <link name="foo"/>
  <joint name="foo_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="base_link"/>
    <child link="foo"/>
  </joint>
</robot>''')  # noqa

    def test_default_param_override(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="fixed_link" params="parent_link:=base_link child_link *joint_pose">
    <link name="${child_link}"/>
    <joint name="${child_link}_joint" type="fixed">
      <xacro:insert_block name="joint_pose" />
      <parent link="${parent_link}"/>
      <child link="${child_link}" />
    </joint>
  </xacro:macro>
  <xacro:fixed_link child_link="foo" parent_link="bar">
    <origin xyz="0 0 0" rpy="0 0 0" />
  </xacro:fixed_link >
</robot>'''), '''
<robot>
  <link name="foo"/>
  <joint name="foo_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="bar"/>
    <child link="foo"/>
  </joint>
</robot>''')  # noqa

    def test_param_missing(self):
        self.assertRaises(xacro.XacroException, self.quick_xacro, '''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:macro name="fixed_link" params="parent_link child_link *joint_pose">
    <link name="${child_link}"/>
    <joint name="${child_link}_joint" type="fixed">
      <xacro:insert_block name="joint_pose" />
      <parent link="${parent_link}"/>
      <child link="${child_link}" />
    </joint>
  </xacro:macro>
  <xacro:fixed_link child_link="foo">
    <origin xyz="0 0 0" rpy="0 0 0" />
  </xacro:fixed_link >
</robot>''')

    def test_default_arg(self):
        self.assert_matches(self.quick_xacro('''
<robot xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:arg name="foo" default="2"/>
  <link name="my_link">
    <origin xyz="0 0 0"/>
  </link>
</robot>
'''), '''
<robot>
  <link name="my_link">
    <origin xyz="0 0 0"/>
  </link>
</robot>''')

#     def test_default_arg_override(self):
#         self.assert_matches(self.quick_xacro('''
# <robot xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <xacro:arg name="foo" default="2"/>
#   <link name="my_link">
#     <origin xyz="0 0 $(arg foo)"/>
#   </link>
# </robot>
# ''', ['foo:=4']), '''
# <robot>
#   <link name="my_link">
#     <origin xyz="0 0 4"/>
#   </link>
# </robot>''')

#     def test_default_arg_missing(self):
#         self.assertRaises(Exception, self.quick_xacro, '''
# <a xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <a arg="$(arg foo)"/>
# </a>
# ''')

    def test_default_arg_empty(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:arg name="foo" default=""/></a>'''), '''<a/>''')

    def test_iterable_literals_plain(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="list" value="[0, 1+1, 2]"/>
  <xacro:property name="tuple" value="(0,1+1,2)"/>
  <xacro:property name="dict" value="{'a':0, 'b':1+1, 'c':2}"/>
  <a list="${list}" tuple="${tuple}" dict="${dict}"/>
</a>'''), '''
<a>
  <a list="[0, 1+1, 2]" tuple="(0,1+1,2)" dict="{'a':0, 'b':1+1, 'c':2}"/>
</a>''')

    def test_iterable_literals_eval(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="list" value="${[0, 1+1, 2]}"/>
  <xacro:property name="tuple" value="${(0,1+1,2)}"/>
  <xacro:property name="dic" value="${dict(a=0, b=1+1, c=2)}"/>
  <a list="${list}" tuple="${tuple}" dict="${dic}"/>
</a>'''), '''
<a>
  <a list="[0, 2, 2]" tuple="(0, 2, 2)" dict="{'a': 0, 'c': 2, 'b': 2}"/>
</a>''')

    def test_literals_eval(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="f" value="1.23"/>
  <xacro:property name="i" value="123"/>
  <xacro:property name="s" value="1_2_3"/>
  float=${f+1} int=${i+1} string=${s}
</a>'''), '''
<a>
  float=2.23 int=124 string=1_2_3
</a>''')

    def test_enforce_xacro_ns(self):
        self.assert_matches(self.quick_xacro('''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <arg name="foo" value="bar"/>
  <include filename="foo"/>
</a>''', xacro_ns=False), '''
<a>
  <arg name="foo" value="bar"/>
  <include filename="foo"/>
</a>''')

#     def test_issue_68_numeric_arg(self):
#         # If a property is assigned from a substitution arg, then this properties' value was  # noqa
#         # no longer converted to a python type, so that e.g. 0.5 remained u'0.5'.  # noqa
#         # If this property is then used in a numerical expression an exception is thrown.  # noqa
#         self.assert_matches(self.quick_xacro('''
# <a xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <xacro:arg name="foo" default="0.5"/>
#   <xacro:property name="prop" value="$(arg foo)" />
#   <a prop="${prop-0.3}"/>
# </a>
# '''), '''
# <a>
#   <a prop="0.2"/>
# </a>''')

#     def test_transitive_arg_evaluation(self):
#         self.assert_matches(self.quick_xacro('''
# <a xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <xacro:arg name="foo" default="0.5"/>
#   <xacro:arg name="bar" default="$(arg foo)"/>
#   <xacro:property name="prop" value="$(arg bar)" />
#   <a prop="${prop-0.3}"/>
# </a>
# '''), '''
# <a>
#   <a prop="0.2"/>
# </a>''')

    def test_macro_name_with_colon(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
        <xacro:macro name="xacro:my_macro"><foo/></xacro:macro>
        <xacro:my_macro/>
        </a>'''
        res = '''<a><foo/></a>'''
        with capture_stderr(self.quick_xacro, src) as (result, output):
            self.assert_matches(result, res)
            self.assertTrue(
              "macro names must not contain prefix 'xacro:'" in output)

    def test_overwrite_globals(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
        <xacro:property name="pi"  value="3.14"/></a>'''
        with capture_stderr(self.quick_xacro, src) as (result, output):
            self.assert_matches(result, '<a/>')
            self.assertTrue(output)

    def test_no_double_evaluation(self):
        src = '''
<a xmlns:xacro="http://www.ros.org/xacro">
  <xacro:macro name="foo" params="a b:=${a} c:=$${a}"> a=${a} b=${b} c=${c} </xacro:macro>
  <xacro:property name="a" value="1"/>
  <xacro:property name="d" value="$${a}"/>
  <d d="${d}"><xacro:foo a="2"/></d>
</a>'''  # noqa
        res = '''<a><d d="${a}"> a=2 b=1 c=${a} </d></a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_property_forwarding(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
        <xacro:property name="arg" value="42"/>
        <xacro:macro name="foo" params="arg:=^%s">${arg}</xacro:macro>
        <xacro:foo/>
        </a>'''
        res = '''<a>%s</a>'''
        self.assert_matches(self.quick_xacro(src % ''), res % '42')
        self.assert_matches(self.quick_xacro(src % '|'), res % '42')
        self.assert_matches(self.quick_xacro(src % '|6'), res % '42')

    # def test_extension_in_expression(self):
    #     src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">${2*'$(arg var)'}</a>'''  # noqa
    #     res = '''<a>%s</a>'''
    #     self.assert_matches(self.quick_xacro(src, ['var:=xacro']), res % (2 * 'xacro'))  # noqa

    # def test_expression_in_extension(self):
    #     src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">$(arg ${'v'+'ar'})</a>'''  # noqa
    #     res = '''<a>%s</a>'''
    #     self.assert_matches(self.quick_xacro(src, ['var:=xacro']), res % 'xacro')  # noqa

    def test_target_namespace(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro" xacro:targetNamespace="http://www.ros.org"/>'''  # noqa
        res = '''<a xmlns="http://www.ros.org"/>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_target_namespace_only_from_root(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro"><b xacro:targetNamespace="http://www.ros.org"/></a>'''  # noqa
        res = '''<a><b/></a>'''
        self.assert_matches(self.quick_xacro(src), res)

    # def test_include_lazy(self):
    #     doc = ('''<a xmlns:xacro="http://www.ros.org/xacro">
    #     <xacro:if value="false"><xacro:include filename="non-existent"/></xacro:if></a>''')  # noqa
    #     self.assert_matches(self.quick_xacro(doc), '''<a/>''')

#     def test_issue_63_fixed_with_inorder_processing(self):
#         self.assert_matches(self.quick_xacro('''
# <a xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <xacro:arg name="has_stuff" default="false"/>
#   <xacro:if value="$(arg has_stuff)">
#     <xacro:include file="$(find nonexistent_package)/stuff.urdf" />
#   </xacro:if>
# </a>'''), '<a/>')

    # def test_include_from_macro(self):
    #     src = '''
    # <a xmlns:xacro="http://www.ros.org/xacro">
    #   <xacro:macro name="foo" params="file:=include1.xml"><xacro:include filename="${file}"/></xacro:macro>  # noqa
    #   <xacro:foo/>
    #   <xacro:foo file="${abs_filename('include1.xml')}"/>
    #   <xacro:include filename="subdir/foo.xacro"/>
    #   <xacro:foo file="$(cwd)/subdir/include1.xml"/>
    # </a>'''
    #     res = '''<a><inc1/><inc1/><subdir_inc1/><subdir_inc1/></a>'''
    #     self.assert_matches(self.quick_xacro(src), res)

#     def test_yaml_support(self):
#         src = '''
# <a xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <xacro:property name="settings" value="${load_yaml('settings.yaml')}"/>
#   <xacro:property name="type" value="$(arg type)"/>
#   <xacro:include filename="${settings['arms'][type]['file']}"/>
#   <xacro:call macro="${settings['arms'][type]['macro']}"/>
# </a>'''
#         res = '''<a><{tag}/></a>'''
#         for i in ['inc1', 'inc2']:
#             self.assert_matches(self.quick_xacro(src, cli=['type:=%s' % i]),
#                                 res.format(tag=i))

#     def test_yaml_support_dotted(self):
#         src = '''
# <a xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <xacro:property name="settings" value="${load_yaml('settings.yaml')}"/>
#   <xacro:property name="type" value="$(arg type)"/>
#   <xacro:include filename="${settings.arms[type].file}"/>
#   <xacro:call macro="${settings.arms[type].macro}"/>
# </a>'''
#         res = '''<a><{tag}/></a>'''
#         for i in ['inc1', 'inc2']:
#             self.assert_matches(self.quick_xacro(src, cli=['type:=%s' % i]),
#                                 res.format(tag=i))

#     def test_yaml_support_dotted_key_error(self):
#         src = '''
# <a xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <xacro:property name="settings" value="${load_yaml('settings.yaml')}"/>
#   <xacro:property name="bar" value="${settings.baz}"/>
#   ${bar}
# </a>'''
#         self.assertRaises(xacro.XacroException, self.quick_xacro, src)

#     def test_yaml_support_dotted_arith(self):
#         src = '''
# <a xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <xacro:property name="settings" value="${load_yaml('settings.yaml')}"/>
#   <xacro:property name="bar" value="${settings.arms.inc2.props.port + 1}"/>
#   ${bar}
# </a>'''
#         res = '''<a>4243</a>'''
#         self.assert_matches(self.quick_xacro(src), res)

#     def test_yaml_support_key_in_dict(self):
#         src = '''
# <a xmlns:xacro="http://www.ros.org/wiki/xacro">
#   <xacro:property name="settings" value="${load_yaml('settings.yaml')}"/>
#   ${'arms' in settings} ${'baz' in settings}
# </a>'''
#         res = '''<a>True False</a>'''
#         self.assert_matches(self.quick_xacro(src), res)

    def test_xacro_exist_required(self):
        src = '''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="non-existent.xacro"/>
</a>'''
        self.assertRaises(xacro.XacroException, self.quick_xacro, src)

    def test_xacro_exist_optional(self):
        src = '''
<a xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:include filename="non-existent.xacro" optional="True"/>
</a>'''
        res = '''<a></a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_macro_default_param_evaluation_order(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:macro name="foo" params="arg:=${2*foo}">
    <xacro:property name="foo" value="-"/>
    <f val="${arg}"/>
</xacro:macro>
<xacro:property name="foo" value="${3*7}"/>
<xacro:foo/>
<xacro:property name="foo" value="*"/>
<xacro:foo/>
</a>'''
        res = '''<a>
<f val="42"/><f val="**"/></a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_default_property(self):
        src = '''
        <a xmlns:xacro="http://www.ros.org/xacro">
            <xacro:property name="prop" default="false"/>
            <xacro:unless value="${prop}">
                <foo/>
                <xacro:property name="prop" value="true"/>
            </xacro:unless>

            <!-- second foo should be ignored -->
            <xacro:unless value="${prop}">
                <foo/>
                <xacro:property name="prop" value="true"/>
            </xacro:unless>
        </a>'''
        res = '''<a><foo/></a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_unicode_literal_parsing(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">🍔 </a>'''
        self.assert_matches(self.quick_xacro(src), '''<a>🍔 </a>''')

    def test_unicode_property(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:property name="burger" value="🍔"/>
${burger}</a>'''
        res = '''<a>🍔</a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_unicode_property_attribute(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:property name="burger" value="🍔"/>
<b c="${burger}"/></a>'''
        res = '''<a><b c="🍔"/></a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_unicode_property_block(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:property name="burger">
🍔
</xacro:property>
<xacro:insert_block name="burger"/></a>'''
        res = '''<a>🍔</a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_unicode_conditional(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:property name="burger" value="🍔"/>
<xacro:if value="${burger == u'🍔'}">
🍟
</xacro:if>
</a>'''
        res = '''<a>🍟</a>'''
        self.assert_matches(self.quick_xacro(src), res)

    def test_unicode_macro(self):
        src = '''<a xmlns:xacro="http://www.ros.org/wiki/xacro">
<xacro:macro name="burger" params="how_many">
${u'🍔' * how_many}
</xacro:macro>
<xacro:burger how_many="4"/>
</a>'''
        res = '''<a>🍔🍔🍔🍔</a>'''
        self.assert_matches(self.quick_xacro(src), res)

    # def test_unicode_file(self):
    #     # run the full xacro processing pipeline on a file with
    #     # unicode characters in it and make sure the output is correct
    #     test_dir = os.path.abspath(os.path.dirname(__file__))
    #     input_path = os.path.join(test_dir, 'emoji.xacro')
    #     tmp_dir_name = tempfile.mkdtemp()  # create directory we can trash
    #     output_path = os.path.join(tmp_dir_name, "out.xml")
    #     self.run_xacro(input_path, '-o', output_path)
    #     self.assertTrue(os.path.isfile(output_path))
    #     self.assert_matches(xml.dom.minidom.parse(output_path), '''<robot>🍔</robot>''')  # noqa
    #     shutil.rmtree(tmp_dir_name)  # clean up after ourselves

    def test_invalid_syntax(self):
        self.assertRaises(xacro.XacroException, self.quick_xacro, '<a>a${</a>')
        self.assertRaises(xacro.XacroException, self.quick_xacro, '<a>${b</a>')
        self.assertRaises(
          xacro.XacroException, self.quick_xacro, '<a>${{}}</a>')
        self.assertRaises(xacro.XacroException, self.quick_xacro, '<a>a$(</a>')
        self.assertRaises(xacro.XacroException, self.quick_xacro, '<a>$(b</a>')


if __name__ == '__main__':  # pragma nocover
    unittest.main()
