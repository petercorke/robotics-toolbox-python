from pathlib import Path
import sys
import importlib


def rtb_load_matfile(filename):
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

    return rtb_load_data(filename, loadmat, squeeze_me=True, struct_as_record=False)

def rtb_load_jsonfile(filename):
    """
    Load toolbox JSON format data file

    :param filename: relative pathname of datafile
    :type filename: str
    :raises ValueError: File does not exist
    :return: contents of JSON data file
    :rtype: dict

    Reads a JSON format file which can contain multiple variables and return
    a dict where the keys are the variable
    names and the values are NumPy arrays.

    .. note::
        - If the filename has no path component, eg. ``map1.mat`, it will be 
          first be looked for in the folder ``roboticstoolbox/data``.
    
    :seealso: :func:`path_to_datafile`
    """
    import json

    return rtb_load_data(filename, lambda f: json.load(open(f, 'r')))

def rtb_load_data(filename, handler, **kwargs):
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

    For example::

        data = rtb_load_data('data/queensland.json', lambda f: json.load(open(f, 'r')))

    
    .. note:: If the filename has no path component, eg. ``foo.dat``, it will 
        first be looked for in the folder ``roboticstoolbox/data``.

    :seealso: :func:`path_to_datafile`
    """
    path = rtb_path_to_datafile(filename)
    return handler(path, **kwargs)

def rtb_path_to_datafile(*filename, local=True):
    """
    Get absolute path to datafile

    :param filename: pathname of datafile
    :type filename: str
    :param local: search for file locally first, default True
    :type local: bool
    :raises FileNotFoundError: File does not exist
    :return: Absolute path
    :rtype: Path

    The positional arguments are joined, like ``os.path.join``.

    If ``local`` is True then ``~`` is expanded and if the file exists, the
    path is made absolute, and symlinks resolved.

    Otherwise, the file is sought within the ``rtbdata`` package and if found,
    return that absolute path.

    Example::

        loadmat('data/map1.mat')   # read rtbdata/data/map1.mat
        loadmat('foo.dat')         # read ./foo.dat
        loadmat('~/foo.dat')       # read $HOME/foo.dat
    """

    filename = Path(*filename)

    if local:
        # check if file is in user's local filesystem

        p = filename.expanduser()
        p = p.resolve()
        if p.exists():
            return p

    # otherwise, look for it in rtbdata

    rtbdata = importlib.import_module("rtbdata")
    root = Path(rtbdata.__path__[0])
    
    path = root / filename
    if path.exists():
        return path.resolve()
    else:
        raise ValueError(f"file {filename} not found locally or in rtbdata")

if __name__ == "__main__":

    a = rtb_loadmat("map1.mat")
    print(a)
    a = rtb_loadmat("data/map1.mat")
    print(a)

