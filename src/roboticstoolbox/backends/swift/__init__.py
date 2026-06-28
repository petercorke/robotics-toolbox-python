from swift.SwiftRoute import SwiftServer, SwiftSocket, start_servers
from swift.SwiftElement import (
    SwiftElement,
    Slider,
    Select,
    Checkbox,
    Radio,
    Button,
    Label,
)
from swift.Swift import Swift as _SwiftBase


class Swift(_SwiftBase):
    """Swift backend with RTB capability flags."""

    supports_teach: bool = False
    supports_ellipse: bool = False


__all__ = [
    "Swift",
    "Slider",
    "SwiftElement",
    "Label",
    "Select",
    "Button",
    "Checkbox",
    "Radio",
    "SwiftServer",
    "SwiftSocket",
    "start_servers",
]
