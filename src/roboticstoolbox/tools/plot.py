from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np


try:
    # spatialmath.base.set_window_title, once released (SMTB PR #263)
    from spatialmath.base import set_window_title
except ImportError:  # pragma: no cover

    def set_window_title(fig: Any, title: str) -> bool:
        """
        Set the title of the window of a Matplotlib figure, if it has one

        :param fig: the figure
        :param title: the window title
        :return: True if the title was set

        A figure on a headless backend, or one that is not managed by pyplot,
        has no window and the title is not set.  This is the one place in RTB
        that sets a window title.

        .. note:: A copy of ``spatialmath.base.set_window_title``, used until
            the minimum version of Spatial Math has it.
        """
        manager = getattr(fig.canvas, "manager", None)
        if manager is not None and hasattr(manager, "set_window_title"):
            manager.set_window_title(title)
            return True
        return False


def apply_block(block: bool | None) -> None:
    """
    Apply the ``block`` argument of a plot method: whether to show the figure and wait

    :param block: ``None`` does nothing, ``False`` shows the figures and returns at
        once, ``True`` shows them and waits until the windows are closed

    This is the one place that turns ``block`` into a Matplotlib call, so that every
    plot method in the toolbox treats it the same way:

    =========  ===========================  ====================================
    ``block``  call                         use
    =========  ===========================  ====================================
    ``None``   none                         leave showing the figure to the
                                            environment, eg. a notebook
    ``False``  ``plt.show(block=False)``    show the figure, carry on
    ``True``   ``plt.show(block=True)``     show the figure, wait until closed
    =========  ===========================  ====================================

    ``None`` is not Matplotlib's ``block=None``, which shows the figure and decides
    for itself whether to wait.  Notebook backends (``inline``, ``ipympl``) ignore
    the argument.  See the RVC ecosystem ``notebooks.md``.

    .. versionadded:: 1.5.0
    """
    if block is not None:
        plt.show(block=block)


def xplot(
    x: Any,
    y: Any = None,
    wrist: bool = False,
    unwrap: bool = False,
    block: bool = False,
    labels: Any = None,
    loc: Any = None,
    grid: bool = True,
    stack: bool = False,
    **kwargs: Any,
) -> Any:
    r"""
    Plot trajectory data

    :param x: trajectory, one row per timestep
    :type x: ndarray(m,n)
    :param t: time vector, optional
    :type t: numpy ndarray, shape=(M,)
    :param wrist: distinguish arm and wrist joints with line styles
    :type wrist: bool
    :param unwrap: unwrap joint angles so that they smoothly increase or
        decrease when they pass through :math:`\pm \pi`
    :type unwrap: bool
    :param block: ``None`` leaves showing the figure to the environment (no
        ``plt.show`` call), ``False`` shows it and returns at once, ``True`` shows
        it and waits until it is closed; ignored by notebook backends,
        defaults to False
    :param labels: legend labels
    :type labels: list of str, or single string with space separated labels
    :param kwargs: options passed to pyplot.plot
    :param loc: legend location as per pyplot.legend
    :type loc: str

    This is a convenience function to plot trajectories, where each row represents one time step.

    - ``xplot(q)`` plots the joint angles versus row number.  If N==6 a
      conventional 6-axis robot is assumed, and the first three joints are
      shown as solid lines, the last three joints (wrist) are shown as dashed
      lines. A legend is also displayed.

    - ``xplot(t, q)`` as above but displays the joint angle trajectory versus
      time given the time vector T (Mx1).

    Example::

        >>> qplot(q, x, labels='x y z')

    :seealso: :func:`jtraj`, :func:`numpy.unwrap`

    .. versionchanged:: 1.5.0
        ``block=None`` does not call ``plt.show``, as documented. It used to call
        ``plt.show(block=None)``, which waits in a script. The default, ``False``, is unchanged.
    """
    if y is None:
        q = x
        t = np.arange(0, q.shape[0])
    else:
        t = x
        q = y

    if q.ndim == 1:
        # if 1D, make it Nx1
        q = q[:, np.newaxis]

    if t.ndim != 1 or q.shape[0] != t.shape[0]:
        raise ValueError("dimensions of arguments are not consistent")

    if unwrap:
        q = np.unwrap(q, axis=0)

    n = q.shape[1]

    if labels is None:
        labels = [f"q{i}" for i in range(n)]
    elif isinstance(labels, str):
        labels = labels.split(" ")
    elif not isinstance(labels, (tuple, list)):
        raise TypeError("wrong type for labels")

    fig, ax = plt.subplots()

    if stack:
        for i in range(n):
            ax = plt.subplot(n, 1, i + 1)

            plt.plot(t, q[:, i], **kwargs)

            plt.grid(grid)
            ax.set_ylabel(labels[i])
            ax.set_xlim(t[0], t[-1])

        ax.set_xlabel("Time (s)")

    else:
        if n == 6 and wrist:
            plt.plot(t, q[:, 0:3], **kwargs)
            plt.plot(t, q[:, 3:6], "--", **kwargs)
        else:
            plt.plot(t, q, **kwargs)

        ax.legend(labels, loc=loc)

        plt.grid(grid)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint coordinates (rad,m)")
        # ax.set_xlim(t[0], t[-1])  fails with RVC3 Sec 3.3.3

    apply_block(block)

    return fig.get_axes()
