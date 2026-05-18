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

# imports for use by IPython and user
import math
from math import pi  # lgtm [py/unused-import]
import numpy as np
from scipy import linalg, optimize
import matplotlib.pyplot as plt  # lgtm [py/unused-import]

from spatialmath import *  # lgtm [py/polluting-import]
from spatialmath.base import *
import spatialmath.base as smb 
from spatialmath.base import sym

from spatialgeometry import *  # lgtm [py/polluting-import]

from roboticstoolbox import *  # lgtm [py/unused-import]
# load some robot models
puma = models.DH.Puma560()
panda = models.DH.Panda()

def parse_arguments():
    parser = argparse.ArgumentParser("Robotics Toolbox shell")
    parser.add_argument("script", default=None, nargs="?", help="specify script to run")
    parser.add_argument(
        "--backend", "-B", default=None, help="specify graphics backend"
    )
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
        "-b",
        "--no-banner",
        dest="banner",
        default=True,
        action="store_false",
        help="suppress startup banner",
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
    args, rest = parser.parse_known_args()

    # remove the arguments we've just parsed from sys.argv so that IPython can have a
    # go at them later
    sys.argv = [sys.argv[0]] + rest

    # TODO more options
    # color scheme, light/dark
    # silent startup

    if args.script is not None:
        args.banner = False

    return args

def make_banner():
    # banner template
    # https://patorjk.com/software/taag/#p=display&f=Cybermedium&t=Robotics%20Toolbox%0A

    banner = f"""\
    ____ ____ ___  ____ ___ _ ____ ____    ___ ____ ____ _    ___  ____ _  _
    |__/ |  | |__] |  |  |  | |    [__      |  |  | |  | |    |__] |  |  \/
    |  \ |__| |__] |__|  |  | |___ ___]     |  |__| |__| |___ |__] |__| _/\_

    for Python"""
    
    versions = []
    versions.append(f"RTB=={version('roboticstoolbox-python')}")
    versions.append(f"SMTB=={version('spatialmath-python')}")
    versions.append(f"SG=={version('spatialmath-python')}")
    versions.append(f"NumPy=={version('numpy')}")
    versions.append(f"SciPy=={version('scipy')}")
    versions.append(f"Matplotlib=={version('matplotlib')}")

    # create banner
    banner += " (" + ", ".join(versions) + ")"
    banner += r"""

    import math
    import numpy as np
    from scipy import linalg, optimize
    import matplotlib.pyplot as plt
    from spatialmath import *
    from spatialmath.base import *
    from spatialmath.base import sym
    from roboticstoolbox import *"

    # useful variables
    from math import pi
    puma = models.DH.Puma560()
    panda = models.DH.Panda()

    func/object?       - show brief help
    help(func/object)  - show detailed help
    func/object??      - show source code

    """

    print(fg("yellow") + banner + attr(0))

def startup():
    plt.ion()

def main():

    args = parse_arguments()


    # setup defaults
    np.set_printoptions(
        linewidth=120,
        formatter={"float": lambda x: f"{0:8.4g}" if abs(x) < 1e-10 else f"{x:8.4g}"},
    )

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

    # build the banner, import * packages and their versions


    if args.banner:
        banner = make_banner()
        print(banner)

    if args.showassign and args.banner:
        print(
            fg("red")
            + "Results of assignments will be displayed, use trailing ; to suppress"
            + attr(0)
            + "\n"
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
    c.TerminalIPythonApp.force_interact = False
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
            "startup()",
            "%precision %.3g",
        ]
    else:
        code.append("plt.ion()")


    c.InteractiveShellApp.exec_lines = code
    IPython.start_ipython(config=c, user_ns=globals())


if __name__ == "__main__":
    main()
