from roboticstoolbox.backends.Connector import Connector


def load_backend(name: str) -> Connector:
    """
    Instantiate a graphical backend by name.

    Parameters
    ----------
    name
        Backend identifier: ``'swift'``, ``'pyplot'``, or ``'pyplot2'``.

    Returns
    -------
    Connector
        An instantiated backend ready for ``.launch()``.

    Raises
    ------
    ModuleNotFoundError
        If the requested backend's package is not installed (e.g. ``swift``
        when ``swift-sim`` is not present).
    ValueError
        If ``name`` is not a known backend.
    """
    if name == "swift":
        from roboticstoolbox.backends.swift import Swift  # optional dep
    
    if name == "isaacsim":
        from roboticstoolbox.backends.IsaacSim.isaacsim_backend import IsaacSim 
        
        return IsaacSim() 

    elif name == "pyplot":
        from roboticstoolbox.backends.PyPlot import PyPlot

        return PyPlot()
    elif name == "pyplot2":
        from roboticstoolbox.backends.PyPlot import PyPlot2

        return PyPlot2()
    else:
        raise ValueError(f"Unknown graphical backend: {name!r}")
