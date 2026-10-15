from __future__ import annotations

from typing import Any

_params = {
    "unicode": True,
}


def rtb_set_param(param: Any, value: Any) -> None:
    _params[param] = value


def rtb_get_param(param: Any) -> Any:
    return _params[param]
