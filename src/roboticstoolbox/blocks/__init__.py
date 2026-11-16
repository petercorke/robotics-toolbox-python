import importlib.metadata
import re

#: the oldest bdsim these blocks work with: the first with the block parameter registry
MIN_BDSIM = (1, 5, 0)


def _check_bdsim_version() -> None:
    """
    Check that the installed bdsim is new enough for these blocks

    :raises ImportError: if bdsim is missing or older than :data:`MIN_BDSIM`

    These blocks are an extension of bdsim, which this package does not require.
    An older bdsim would otherwise fail later with an obscure error from deep
    inside bdsim, so say what is wrong here instead.
    """
    need = ".".join(str(v) for v in MIN_BDSIM)
    try:
        have = importlib.metadata.version("bdsim")
    except importlib.metadata.PackageNotFoundError:
        raise ImportError(
            f"roboticstoolbox.blocks needs bdsim {need} or later, which is not "
            "installed: pip install 'roboticstoolbox-python[blocks]'"
        ) from None
    # the leading digits of each of the first three parts, so 1.5.0rc1 is (1, 5, 0)
    parts = tuple(
        int(m.group()) if (m := re.match(r"\d+", p)) else 0 for p in have.split(".")[:3]
    )
    if parts < MIN_BDSIM:
        raise ImportError(
            f"roboticstoolbox.blocks needs bdsim {need} or later, but {have} is "
            "installed: pip install --upgrade bdsim"
        )


_check_bdsim_version()

from .arm import *
from .mobile import *
from .uav import *
from .spatial import *

url = "https://petercorke.github.io/robotics-toolbox-python/" + __package__
