"""
The registry of graphical backends: the names a backend can be asked for by, and
how to make each one.

``robot.plot(q, backend="swift")`` looks the name up here.  A backend that ships
with the Toolbox is an entry in the table below; a backend from anywhere else is
added with :func:`register_backend`.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from roboticstoolbox.backends.Connector import Connector


@dataclass(frozen=True)
class _Entry:
    #: ``"module:Class"``, imported only when the backend is chosen, or a callable
    #: that returns a backend
    target: str | Callable[[], Connector]
    #: the pip extra that provides the backend's package, or None
    extra: str | None = None


_registry: dict[str, _Entry] = {}


def normalise_backend_name(name: str) -> str:
    """Backend names are not case sensitive, and surrounding spaces do not count."""
    if not isinstance(name, str):
        raise TypeError(f"a backend name is a string, not {type(name).__name__}")
    return name.strip().lower()


def register_backend(
    name: str,
    target: str | Callable[[], Connector],
    *,
    replace: bool = False,
    extra: str | None = None,
) -> None:
    """
    Make a graphical backend available by name

    ``register_backend("mine", "my_package.module:MyBackend")`` lets
    ``robot.plot(q, backend="mine")`` use it.  The registration lasts for this
    process, so call it before ``plot()``.  It is how a backend can be developed,
    tested and used outside the Toolbox, and how one that cannot live in it (a
    licence, a GPU) can be kept as a package of its own.

    :param name: name to give the backend, which is not case sensitive
    :param target: either a string ``"module:Class"``, which is imported only when
        the backend is chosen, so registering it costs nothing and the package need
        not be installed; or a callable, such as a class or a function, that returns
        a backend each time it is called
    :param replace: replace the backend if the name is already registered
    :param extra: the pip extra of ``roboticstoolbox-python`` that installs the
        backend's package, used to say so if the package is missing
    :raises ValueError: the name is empty, ``target`` is a string that is not of the
        form ``"module:Class"``, or the name is registered and ``replace`` is False
    :raises TypeError: ``name`` is not a string, or ``target`` is neither a string
        nor callable

    .. seealso:: :func:`load_backend`, :func:`registered_backends`

    .. versionadded:: 1.5.1
    """
    key = normalise_backend_name(name)
    if not key:
        raise ValueError("a backend needs a name")

    if isinstance(target, str):
        module, _, attribute = target.partition(":")
        if not module or not attribute or ":" in attribute:
            raise ValueError(
                f"the target of backend {name!r} is {target!r}, which is not of "
                'the form "module:Class"'
            )
    elif not callable(target):
        raise TypeError(
            f"the target of backend {name!r} is a string such as "
            f'"module:Class" or a callable, not {type(target).__name__}'
        )

    if key in _registry and not replace:
        raise ValueError(
            f"backend {key!r} is already registered, use replace=True to replace it"
        )
    _registry[key] = _Entry(target, extra)


def registered_backends() -> list[str]:
    """
    The names of the registered graphical backends

    :returns: the names, in alphabetical order

    .. seealso:: :func:`register_backend`

    .. versionadded:: 1.5.1
    """
    return sorted(_registry)


def load_backend(name: str) -> Connector:
    """
    Make a graphical backend from its name

    The name is not case sensitive.  A backend given as ``"module:Class"`` is
    imported now, not when it was registered.

    :param name: a registered backend, such as ``"swift"``, ``"pyplot"`` or
        ``"pyplot2"``, see :func:`registered_backends`
    :returns: a backend, ready for ``launch()``
    :raises ValueError: ``name`` is not registered
    :raises ModuleNotFoundError: the package that provides the backend is not
        installed, such as ``swift-sim`` for ``"swift"``; the message names the pip
        extra if there is one

    .. seealso:: :func:`register_backend`
    """
    key = normalise_backend_name(name)
    try:
        entry = _registry[key]
    except KeyError:
        raise ValueError(
            f"Unknown graphical backend: {name!r}. "
            f"The registered backends are: {', '.join(registered_backends())}"
        ) from None

    if callable(entry.target):
        return entry.target()

    module_name, _, attribute = entry.target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as err:
        if entry.extra is None:
            raise
        raise ModuleNotFoundError(
            f"the {key!r} backend needs a package that is not installed ({err}): "
            f"pip install roboticstoolbox-python[{entry.extra}]",
            name=err.name,
        ) from err
    return getattr(module, attribute)()


# the backends that come with the Toolbox
register_backend("swift", "roboticstoolbox.backends.swift:Swift", extra="swift")
register_backend("pyplot", "roboticstoolbox.backends.PyPlot:PyPlot")
register_backend("pyplot2", "roboticstoolbox.backends.PyPlot:PyPlot2")
