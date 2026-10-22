#!/usr/bin/env python

"""
@author: Jesse Haviland
"""

from __future__ import annotations

from typing import Any

from roboticstoolbox.robot.Link import Link, Link2


class ELink(Link):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import warnings

        warnings.warn("ELink is deprecated, use Link instead", FutureWarning)
        super().__init__(*args, **kwargs)


class ELink2(Link2):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        import warnings

        warnings.warn("ELink2 is deprecated, use Link2 instead", FutureWarning)
        super().__init__(*args, **kwargs)
