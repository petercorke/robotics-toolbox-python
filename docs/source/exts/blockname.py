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

    # icons are copied from src/roboticstoolbox/blocks/Icons into _static/
    # by html_static_path in conf.py, and served from there at build time --
    # avoids depending on a live network fetch (and a hardcoded branch/path)
    # for every docs build. The relative prefix accounts for how deeply
    # nested the current page is, matching Sphinx's own pathto() semantics
    # for _static references, so this keeps working if these pages ever
    # move into a subdirectory.
    env = inliner.document.settings.env
    depth = env.docname.count("/")
    prefix = "../" * depth
    path = f"{prefix}_static/{text.lower()}.png"
    html_node = nodes.raw(text=html.format(text, path), format="html")
    return [html_node], []


def summary_without_banner(app, what, name, obj, options, lines):
    """
    Give autosummary the description of a block, not its banner

    Autosummary uses the first paragraph of the docstring of a class as its one
    line summary.  For a block that is the ``:blockname:`` banner, which would
    appear in a table as the block's name in capitals.  So when autosummary asks,
    drop the banner and use the description that follows.  Autodoc passes the
    options of its directive, and autosummary does not, which is how the two are
    told apart: the pages of the blocks keep their banners.
    """
    if (
        what == "class"
        and options.members is None
        and lines
        and lines[0].lstrip().startswith(":blockname:")
    ):
        del lines[0]
        while lines and not lines[0].strip():
            del lines[0]


def no_sidebar_entries(app, doctree):
    """
    Keep the members of the blocks out of the sidebar

    Each block is a class, and gets an entry in the table of contents, which the
    sidebar shows.  Its ``__init__`` would get one too, and make the block look
    as if it could be opened when there is nothing below it worth a link.  So the
    members of the classes on these pages are marked as not wanting an entry, and
    the blocks are the last level.  This has to run before the table of contents
    is built from the page.
    """
    if app.env.docname.startswith("blocks-"):
        from sphinx import addnodes

        for node in doctree.findall(addnodes.desc):
            if node.get("objtype") != "class":
                node["no-contents-entry"] = node["nocontentsentry"] = True


def setup(app):
    app.connect("autodoc-process-docstring", summary_without_banner)
    app.connect("doctree-read", no_sidebar_entries, priority=100)
    app.add_role("blockname", block_name)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
