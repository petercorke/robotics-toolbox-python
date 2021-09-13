from docutils import nodes
#from docutils.parsers.rst import Directive


def block_name(name, rawtext, text, lineno, inliner, options={}, content=[]):

    # name: The local name of the interpreted role, the role name actually used in the document.
    # rawtext: A string containing the enitre interpreted text input, including the role and markup. Return it as a problematic node linked to a system message if a problem is encountered.
    # text: The interpreted text content.
    # lineno: The line number where the interpreted text begins.
    # inliner: The docutils.parsers.rst.states.Inliner object that called role_fn. It contains the several attributes useful for error reporting and document tree access.
    # options: A dictionary of directive options for customization (from the "role" directive), to be interpreted by the role function. Used for additional attributes for the generated elements and other functionality.
    # content: A list of strings, the directive content for customization (from the "role" directive). To be interpreted by the role function.
    html_node = nodes.raw(text='<p style="border:10px; background-color:#000000; padding: 1em; color: white; font-size: 30px; font-weight: bold;">' + text + '</p>', format='html')
    return [html_node], []


def setup(app):
    app.add_role("blockname", block_name)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }