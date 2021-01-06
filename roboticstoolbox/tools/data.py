from pathlib import Path
import sys

def loadmat(filename):
    """
    Load toolbox mat format data file

    :param filename: relative pathname of datafile
    :type filename: str
    :raises ValueError: File does not exist
    :return: contents of mat data file
    :rtype: dict

    Reads a MATLAB format *mat* file which can contain multiple variables, in 
    a binary or ASCII format.  Returns a dict where the keys are the variable
    names and the values are NumPy arrays.

    .. note::
        - Uses SciPy ``io.loadmat`` to do the work.
        - If the filename has no path component, eg. ``map1.mat`, it will be 
          first be looked for in the folder ``roboticstoolbox/data``.
    
    :seealso: :func:`path_to_datafile`
    """
    from scipy.io import loadmat

    return loaddata(filename, loadmat, squeeze_me=True, struct_as_record=False)

def loaddata(filename, handler, **kwargs):
    """
    Load toolbox data file

    :param filename: relative pathname of datafile
    :type filename: str
    :param handler: function to read data
    :type handler: callable
    :raises ValueError: File does not exist
    :return: data object

    Resolves the relative pathname to an absolute name and then invokes the
    data reading function::

        handler(abs_file_name, **kwargs)
    
    .. note:: If the filename has no path component, eg. ``foo.dat``, it will 
        first be looked for in the folder ``roboticstoolbox/data``.

    :seealso: :func:`path_to_datafile`
    """
    path = path_to_datafile(filename)
    return handler(path, **kwargs)

def path_to_datafile(filename):
    """
    Get absolute path to datafile

    :param filename: pathname of datafile
    :type filename: str
    :raises FileNotFoundError: File does not exist
    :return: Absolute path
    :rtype: str

    If ``filename`` contains no path specification eg. ``map1.mat`` it will
    first attempt to locate the file within the ``roboticstoolbox/data``
    folder and if found, return that absolute path.

    Otherwise, ``~`` is expanded, the path made absolute, resolving symlinks
    and the file's existence is tested.

    Example::

        loadmat('map1.mat')        # read ...roboticstoolbox/data/map1.mat
        loadmat('foo.dat')         # read ./foo.dat
        loadmat('~/data/foo.dat')  # read ~/data/foo.dat
    """

    filename = Path(filename)

    if filename.parent == Path():
        # just a filename, no path, assume it is in roboticstoolbox/data

        p = Path(__file__).resolve().parent.parent / 'data' / filename
        if p.exists():
            return str(p.resolve())

    p = filename.expanduser()
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"File '{p}' does not exist")
    return str(p)

if __name__ == "__main__":

    a = loadmat("map1.mat")
    print(a)
    a = loadmat("roboticstoolbox/data/map1.mat")
    print(a)
    a = loadmat("roboticstoolbox/data/../data/map1.mat")
    print(a)
    a = loadmat("~/code/robotics-toolbox-python/roboticstoolbox/data/map1.mat")
    print(a)
