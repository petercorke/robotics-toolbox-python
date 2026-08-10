#!/usr/bin/env python3
"""
Interactive Robotics Toolbox shell — starts an IPython session with NumPy,
RTB, and SpatialMath pre-imported.

Usage::

    $ rtbtool
    $ rtbtool myscript.py
"""

# import stuff
import argparse
import pathlib
import shlex
import sys
import os
from importlib.metadata import version

from roboticstoolbox.bin._bintools import LineWrapRawTextDefaultsHelpFormatter

try:
    from colored import fg, bg, attr

    _colored = True
    # print('using colored output')
except ImportError:
    # print('colored not found')
    _colored = False
    fg = lambda *args, **kwargs: ""
    bg = lambda *args, **kwargs: ""
    attr = lambda *args, **kwargs: ""

# imports for use by IPython and user
import math
from math import pi  # lgtm [py/unused-import]
import numpy as np
from scipy import linalg, optimize
import matplotlib.pyplot as plt  # lgtm [py/unused-import]
import matplotlib as mpl

from spatialmath import *  # lgtm [py/polluting-import]
from spatialmath.base import *
import spatialmath.base as smb
from spatialmath.base import sym

from spatialgeometry import *  # lgtm [py/polluting-import]

from roboticstoolbox import *  # lgtm [py/unused-import]

_OPTIONS_ENVVAR = "RTB_OPTIONS"


def env_arguments(parser):
    """Return command-line style options from the environment.

    :param parser: argument parser used for error reporting
    :type parser: :class:`argparse.ArgumentParser`
    :return: tokenised environment arguments
    :rtype: list[str]
    """
    options = os.environ.get(_OPTIONS_ENVVAR)
    if not options:
        return []

    try:
        return shlex.split(options)
    except ValueError as exc:
        parser.error(f"invalid {_OPTIONS_ENVVAR}: {exc}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Robotics Toolbox shell",
        formatter_class=LineWrapRawTextDefaultsHelpFormatter,
        epilog=(
            "options can be set via the environment variable RTB_OPTIONS, "
            "for example:\n\n"
            "    $ export RTB_OPTIONS=\"--backend TkAgg --prompt 'rtb> ' "
            '--reload --showassign"\n'
        ),
    )
    parser.add_argument("script", default=None, nargs="?", help="specify script to run")
    parser.add_argument(
        "--backend", "-B", default=None, help="specify graphics backend"
    )
    parser.add_argument(
        "--theme",
        "-t",
        default="neutral",
        help="specify terminal color theme (neutral, lightbg, nocolor, linux), linux is for dark mode",
    )
    parser.add_argument(
        "--confirmexit",
        "-x",
        default=False,
        action="store_true",
        help="confirm exit",
    )
    parser.add_argument("--prompt", "-P", default="(rtb) >>> ", help="input prompt")
    parser.add_argument(
        "--resultprefix",
        "-R",
        default=None,
        help="execution result prefix, include {} for execution count number",
    )
    parser.add_argument(
        "--reload",
        default=False,
        action="store_true",
        help="enable autoreload of any imported modules, same as IPython's builtin %%autoreload 2",
    )
    parser.add_argument(
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
        help="display the result of assignments",
    )
    parser.add_argument(
        "--book",
        default=False,
        action="store_true",
        help="use defaults as per RVC book",
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
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="non-interactive environment smoke test: print package versions, "
        "exercise one real numeric code path per package, exit 0/1 "
        "instead of starting an interactive shell",
    )

    argv = env_arguments(parser) + sys.argv[1:]
    args, rest = parser.parse_known_args(argv)

    if args.script is not None:
        args.banner = False

    return args, rest


def get_versions() -> list[str]:
    """Package version strings shown in the banner and by --test."""
    return [
        f"RTB=={version('roboticstoolbox-python')}",
        f"SMTB=={version('spatialmath-python')}",
        f"SG=={version('spatialgeometry')}",
        f"NumPy=={version('numpy')}",
        f"SciPy=={version('scipy')}",
        f"Matplotlib=={version('matplotlib')}",
    ]


def make_banner():
    # banner template
    # https://patorjk.com/software/taag/#p=display&f=Cybermedium&t=Robotics%20Toolbox%0A

    banner = r"""\
    ____ ____ ___  ____ ___ _ ____ ____    ___ ____ ____ _    ___  ____ _  _
    |__/ |  | |__] |  |  |  | |    [__      |  |  | |  | |    |__] |  |  \/
    |  \ |__| |__] |__|  |  | |___ ___]     |  |__| |__| |___ |__] |__| _/\_

    for Python"""

    # create banner
    banner += " (" + ", ".join(get_versions()) + ")"
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

    return banner


def examples_path():
    # Repository layout: <root>/src/roboticstoolbox/bin/rtbtool.py
    return pathlib.Path(__file__).resolve().parents[3] / "examples"


def startup():
    plt.ion()


def run_smoke_test() -> bool:
    """Non-interactive environment sanity check, used by --test.

    Not a substitute for the pytest suite -- a fast, human- or script-run
    "did this environment actually come together correctly" check: real
    versions, confirmation the compiled extensions loaded, and one real
    numeric result compared against a known-correct value. The last part
    matters specifically because a compiled extension built against the
    wrong NumPy ABI can load successfully and still compute garbage --
    checking that it merely *imported* wouldn't catch that.
    """
    print(", ".join(get_versions()))

    from roboticstoolbox.ets.fknm import _C_AVAILABLE as fknm_c
    from roboticstoolbox.robot.frne import _C_AVAILABLE as frne_c

    panda = models.DH.Panda()
    T = panda.fkine(panda.qr).A
    expected = np.array(
        [
            [9.9500416528e-01, 0.0000000000e00, 9.9833416647e-02, 4.8400688203e-01],
            [0.0000000000e00, -1.0000000000e00, -1.2032944640e-16, -6.8775459668e-17],
            [9.9833416647e-02, 1.2490009027e-16, -9.9500416528e-01, 4.1302777713e-01],
            [0.0000000000e00, 0.0000000000e00, 0.0000000000e00, 1.0000000000e00],
        ]
    )

    checks = [
        ("fknm compiled extension loaded", fknm_c),
        ("frne compiled extension loaded", frne_c),
        (
            "Panda.fkine(qr) matches expected (1e-9)",
            bool(np.allclose(T, expected, atol=1e-9)),
        ),
    ]

    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    n_passed = sum(1 for _, passed in checks if passed)
    print(f"rtbtool --test: {n_passed}/{len(checks)} checks passed")
    return n_passed == len(checks)


def main():
    args, ipython_args = parse_arguments()

    if args.test:
        sys.exit(0 if run_smoke_test() else 1)

    try:
        import IPython
        from IPython.terminal.prompts import Prompts
        from pygments.token import Token
        from traitlets.config import Config
    except ImportError as e:
        sys.exit(
            f"rtbtool requires IPython and pygments, which are not "
            f"installed ({e}).\nInstall them with:\n\n"
            "    pip install roboticstoolbox-python[tool]\n"
        )

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

    if args.examples:
        path = examples_path()
        if path.exists() and path.is_dir():
            print(f"Changing working directory to {path}")
            os.chdir(path)
        else:
            print(f"Examples directory not found: {path}")

    # load some robot models after argument handling to avoid import-time side effects
    puma = models.DH.Puma560()
    panda = models.DH.Panda()

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
        print(fg("yellow") + banner + attr(0))

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
    c.InteractiveShellEmbed.colors = args.theme
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
        path = pathlib.Path(args.script)
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

    if args.reload:
        code = ["%load_ext autoreload", "%autoreload 2"] + code

    c.InteractiveShellApp.exec_lines = code
    namespace = {k: v for k, v in globals().items() if not k.startswith("__")}
    namespace.update({"puma": puma, "panda": panda})

    # clear argv so IPython doesn't try to reparse arguments we've already consumed
    sys.argv = sys.argv[:1]
    IPython.start_ipython(config=c, user_ns=namespace, argv=ipython_args)


if __name__ == "__main__":
    main()
