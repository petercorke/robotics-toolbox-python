from docutils import nodes

# from docutils.parsers.rst import Directive


def block_name(name, rawtext, text, lineno, inliner, options={}, content=[]):

    # name: The local name of the interpreted role, the role name actually used in the document.
    # rawtext: A string containing the enitre interpreted text input, including the role and markup. Return it as a problematic node linked to a system message if a problem is encountered.
    # text: The interpreted text content.
    # lineno: The line number where the interpreted text begins.
    # inliner: The docutils.parsers.rst.states.Inliner object that called role_fn. It contains the several attributes useful for error reporting and document tree access.
    # options: A dictionary of directive options for customization (from the "role" directive), to be interpreted by the role function. Used for additional attributes for the generated elements and other functionality.
    # content: A list of strings, the directive content for customization (from the "role" directive). To be interpreted by the role function.

    html = """
    <table width="100%">
    	<col style="width:75%">
        <col style="width:5%;">
	    <col style="width:20%">
    <tr>
    <td>
    <p style="border:10px; background-color:#000000; padding: 1em; color: white; font-size: 30px; font-weight: bold;">
    {0}
    </p>
    </td>
    <td></td>
    <td>
    <img src="{1}" width=80 height=80 style="border:2px solid black; border-radius:10px; padding: 4px">
    </td>
    </tr>
    </table>
    """

    # this is the path to the icons within the github repo
    path = (
        "https://github.com/petercorke/robotics-toolbox-python/raw/master/roboticstoolbox/blocks/Icons/"
        + text.lower()
        + ".png"
    )
    html_node = nodes.raw(text=html.format(text, path), format="html")
    return [html_node], []


def setup(app):
    app.add_role("blockname", block_name)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
