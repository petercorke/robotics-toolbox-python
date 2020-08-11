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

import xml.dom.minidom


def first_child_element(elt):
    c = elt.firstChild
    while c and c.nodeType != xml.dom.Node.ELEMENT_NODE:
        c = c.nextSibling
    return c


def next_sibling_element(node):
    c = node.nextSibling
    while c and c.nodeType != xml.dom.Node.ELEMENT_NODE:
        c = c.nextSibling
    return c


def replace_node(node, by, content_only=False):
    parent = node.parentNode

    if by is not None:
        if not isinstance(by, list):
            by = [by]

        # insert new content before node
        for doc in by:
            if content_only:
                c = doc.firstChild
                while c:
                    n = c.nextSibling
                    parent.insertBefore(c, node)
                    c = n
            else:
                parent.insertBefore(doc, node)

    # remove node
    parent.removeChild(node)


def attribute(tag, a):
    """
    Helper function to fetch a single attribute value from tag
    :param tag (xml.dom.Element): DOM element node
    :param a (str): attribute name
    :return: attribute value if present, otherwise None
    """
    if tag.hasAttribute(a):
        # getAttribute returns empty string for non-existent attributes,
        # which makes it impossible to distinguish with empty values
        return tag.getAttribute(a)
    else:
        return None


def opt_attrs(tag, attrs):
    """
    Helper routine for fetching optional tag attributes
    :param tag (xml.dom.Element): DOM element node
    :param attrs [str]: list of attributes to fetch
    """
    return [attribute(tag, a) for a in attrs]


def reqd_attrs(tag, attrs):
    """
    Helper routine for fetching required tag attributes
    :param tag (xml.dom.Element): DOM element node
    :param attrs [str]: list of attributes to fetch
    :raise RuntimeError: if required attribute is missing
    """
    result = opt_attrs(tag, attrs)
    for (res, name) in zip(result, attrs):
        if res is None:
            raise RuntimeError("%s: missing attribute '%s'" % (tag.nodeName, name))   # pragma: no cover # noqa
    return result


# Better pretty printing of xml
# Taken from
# http://ronrothman.com/public/leftbraned/xml-dom-minidom-toprettyxml-and-silly-whitespace/ # noqa
def fixed_writexml(self, writer, indent="", addindent="", newl=""):   # pragma: no cover # noqa
    # indent = current indentation
    # addindent = indentation to add to higher levels
    # newl = newline string
    writer.write(indent + "<" + self.tagName)

    attrs = self._get_attributes()
    a_names = sorted(attrs.keys())

    for a_name in a_names:
        writer.write(" %s=\"" % a_name)
        xml.dom.minidom._write_data(writer, attrs[a_name].value)
        writer.write("\"")
    if self.childNodes:
        if len(self.childNodes) == 1 \
           and self.childNodes[0].nodeType == xml.dom.minidom.Node.TEXT_NODE:
            writer.write(">")
            self.childNodes[0].writexml(writer, "", "", "")
            writer.write("</%s>%s" % (self.tagName, newl))
            return
        writer.write(">%s" % newl)
        for node in self.childNodes:
            # skip whitespace-only text nodes
            if node.nodeType == xml.dom.minidom.Node.TEXT_NODE and \
                    (not node.data or node.data.isspace()):
                continue
            node.writexml(writer, indent + addindent, addindent, newl)
        writer.write("%s</%s>%s" % (indent, self.tagName, newl))
    else:
        writer.write("/>%s" % newl)


# replace minidom's function with ours
xml.dom.minidom.Element.writexml = fixed_writexml
