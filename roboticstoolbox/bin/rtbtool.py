#!/usr/bin/env python3

# a simple Robotics Toolbox "shell", runs Python3 and loads in NumPy, RTB, SMTB
#
# Run it from the shell
#  % rtb.py
#
# or setup an alias
#
#  alias rtb=PATH/rtb.py   # sh/bash
#  alias rtb PATH/rtb.py   # csh/tcsh
#
# % rtb

# import stuff
from pygments.token import Token
from IPython.terminal.prompts import Prompts
from IPython.terminal.prompts import ClassicPrompts
from traitlets.config import Config
import IPython
import argparse
from math import pi  # lgtm [py/unused-import]
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt  # lgtm [py/unused-import]
from roboticstoolbox import *  # lgtm [py/unused-import]
from spatialmath import *  # lgtm [py/polluting-import]
from spatialgeometry import *  # lgtm [py/polluting-import]
import spatialmath.base as smb 
from spatialmath.base import sym
import matplotlib as mpl
from pathlib import Path
import sys
from importlib.metadata import version

try:
    from colored import fg, bg, attr

    _colored = True
    # print('using colored output')
except ImportError:
    # print('colored not found')
    _colored = False

def main():
    # setup defaults
    np.set_printoptions(
        linewidth=120,
        formatter={"float": lambda x: f"{0:8.4g}" if abs(x) < 1e-10 else f"{x:8.4g}"},
    )

    parser = argparse.ArgumentParser("Robotics Toolbox shell")
    parser.add_argument("script", default=None, nargs="?", help="specify script to run")
    parser.add_argument("--backend", "-b", default=None, help="specify Matplotlib backend")
    parser.add_argument(
        "--color",
        "-c",
        default="neutral",
        help="specify terminal color scheme (neutral, lightbg, nocolor, linux), linux is for dark mode",
    )
    parser.add_argument("--confirmexit", "-x", default=False,
        help="confirm exit")
    parser.add_argument("--prompt", "-p", default="(rtb) >>> ",
        help="input prompt")
    parser.add_argument(
        "--resultprefix",
        "-r",
        default=None,
        help="execution result prefix, include {} for execution count number",
    )
    parser.add_argument(
        "--showassign",
        "-a",
        default=False,
        action="store_true",
        help="do not display the result of assignments",
    )
    parser.add_argument(
        "--book", default=False, action="store_true",
        help="use defaults as per RVC book"
    )
    parser.add_argument(
        "--vision", default=False, action="store_true",
        help="import vision toolbox (MVTB)"
    )
    parser.add_argument(
        "--ansi",
        default=False,
        action="store_true",
        help="use ANSImatrix to display matrices",
    )
    parser.add_argument(
        "--examples",
        "-e",
        default=False,
        action="store_true",
        help="change working directory to shipped examples",
    )
    parser.add_argument(
        "--swift",
        "-s",
        default=False,
        action="store_true",
        help="use Swift as default backend",
    )
    args = parser.parse_args()

    # TODO more options
    # color scheme, light/dark
    # silent startup

    sys.argv = [sys.argv[0]]

    if args.book:
        # set book options
        args.resultprefix = ""
        args.prompt = ">>> "
        args.showassign = True
        args.ansi = False
        args.examples = True

    # set default backend for Robot.plot
    if args.swift:
        Robot.default_backend = "swift"

    # set matrix printing mode for spatialmath
    SE3._ansimatrix = args.ansi

    # set default matplotlib backend
    if args.backend is not None:
        print(f"Using matplotlb backend {args.backend}")
        mpl.use(args.backend)

    # load some robot models
    puma = models.DH.Puma560()
    panda = models.DH.Panda()

    # build the banner, import * packages and their versions
    versions = f"(RTB=={version('roboticstoolbox-python')}, SMTB=={version('spatialmath-python')}"
    imports = ["roboticstoolbox", "spatialmath"]
    if args.vision:
        versions += f", MVTB=={version('machinevision-toolbox-python')}"
        imports.append("machinevisiontoolbox")

    versions += ")"
    imports = "\n".join([f"    from {x} import *" for x in imports])

    # banner template
    # https://patorjk.com/software/taag/#p=display&f=Cybermedium&t=Robotics%20Toolbox%0A
    banner = f"""\
    ____ ____ ___  ____ ___ _ ____ ____    ___ ____ ____ _    ___  ____ _  _
    |__/ |  | |__] |  |  |  | |    [__      |  |  | |  | |    |__] |  |  \/
    |  \ |__| |__] |__|  |  | |___ ___]     |  |__| |__| |___ |__] |__| _/\_

    for Python {versions}

{imports}
    import numpy as np
    import scipy as sp

    func/object?       - show brief help
    help(func/object)  - show detailed help
    func/object??      - show source code

    """

    print(fg("yellow") + banner + attr(0))

    if args.showassign:
        print(
            fg("red")
            + """Results of assignments will be displayed, use trailing ; to suppress

    """,
            attr(0),
        )

    # drop into IPython
    class MyPrompt(Prompts):
        def in_prompt_tokens(self, cli=None):
            return [(Token.Prompt, args.prompt)]

        def out_prompt_tokens(self, cli=None):
            if args.resultprefix is None:
                # traditional behaviour
                return [
                    (Token.OutPrompt, "Out["),
                    (Token.OutPromptNum, str(self.shell.execution_count)),
                    (Token.OutPrompt, "]: "),
                ]
            else:
                return [
                    (Token.Prompt, args.resultprefix.format(self.shell.execution_count))
                ]

    # set configuration options, there are lots, see
    # https://ipython.readthedocs.io/en/stable/config/options/terminal.html
    c = Config()
    c.InteractiveShellEmbed.colors = args.color
    c.InteractiveShell.confirm_exit = args.confirmexit
    # c.InteractiveShell.prompts_class = ClassicPrompts
    c.InteractiveShell.prompts_class = MyPrompt
    if args.showassign:
        c.InteractiveShell.ast_node_interactivity = "last_expr_or_assign"

    # set precision, same as %precision
    c.PlainTextFormatter.float_precision = "%.3f"

    # set up a script to be executed by IPython when we get there
    code = None
    if args.script is not None:
        path = Path(args.script)
        if not path.exists():
            raise ValueError(f"script does not exist: {args.script}")
        code = path.open("r").readlines()
    if code is None:
        code = [
            "%precision %.3g",
            "plt.ion()",
        ]

    else:
        code.append("plt.ion()")
    if args.vision:
        code.append("from machinevisiontoolbox import *")
    c.InteractiveShellApp.exec_lines = code
    IPython.start_ipython(config=c, user_ns=globals())



if __name__ == "__main__":
    main()
