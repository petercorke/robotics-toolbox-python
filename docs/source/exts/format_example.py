import re

"""
This extension looks for examples under the `.. runblock:: pycon` directive and indents the
code underneath.

For example:

.. runblock:: pycon
>>> import roboticstoolbox as rtb
>>> panda = rtb.models.Panda().ets()
>>> solver = rtb.IK_NR(pinv=True)
>>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
>>> solver.solve(panda, Tep)

becomes

.. runblock:: pycon

    >>> import roboticstoolbox as rtb
    >>> panda = rtb.models.Panda().ets()
    >>> solver = rtb.IK_NR(pinv=True)
    >>> Tep = panda.fkine([0, -0.3, 0, -2.2, 0, 2, 0.7854])
    >>> solver.solve(panda, Tep)

"""


def process_doc(app, what, name, obj, options, lines):

    run_directive_re = r"( )*(\.\. runblock:: pycon)"
    python_re = r"( )*(>>>)"

    for i, line in enumerate(lines):

        if re.match(run_directive_re, line):
            # We have matched "   .. runblock:: pycon"

            # Insert a blank line
            lines.insert(i + 1, "")

            searching = True
            j = 2

            while searching:
                try:
                    if re.match(python_re, lines[i + j]):
                        # We have matched "   >>>"
                        lines[i + j] = "    " + lines[i + j]
                    else:
                        # We have reached the end of the example
                        searching = False

                    j += 1
                except IndexError:
                    # End of the docstring has been reached
                    return


def setup(app):
    app.connect("autodoc-process-docstring", process_doc)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
