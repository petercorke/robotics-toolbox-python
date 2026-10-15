"""
Window titles of the Matplotlib backends
"""

from __future__ import annotations

from typing import Any


def set_window_title(fig: Any, title: str) -> bool:
    """
    Set the title of the window of a Matplotlib figure, if it has one

    :param fig: the figure
    :param title: the window title
    :return: True if the title was set

    A figure on a headless backend, or one that is not managed by pyplot, has no
    window and the title is not set.
    """
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)
        return True
    return False
