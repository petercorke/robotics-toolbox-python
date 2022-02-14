#!/usr/bin/env python

"""
@author: Jesse Haviland
"""

from roboticstoolbox.robot.Link import Link, Link2


class ELink(Link):
    def __init__(self, **kwargs):
        import warnings

        warnings.warn("ELink is deprecated, use Link instead", FutureWarning)
        super().__init__(**kwargs)


class ELink2(Link2):
    def __init__(self, **kwargs):
        import warnings

        warnings.warn("ELink2 is deprecated, use Link instead", FutureWarning)
        super().__init__(**kwargs)
