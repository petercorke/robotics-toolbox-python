from pathlib import Path

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

    .. note::  Uses SciPy ``io.loadmat`` to do the work.
    """
    from scipy.io import loadmat

    path = _path_datafile(filename)
    return loaddata(path, loadmat, squeeze_me=True, struct_as_record=False)

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
    """
    path = _path_datafile(filename)
    return handler(path, **kwargs)

def _path_datafile(filename):
    """
    Get absolute path to toolbox datafile

    :param filename: relative pathname of datafile
    :type filename: str
    :raises ValueError: File does not exist
    :return: Absolute path relative to *roboticstoolbox* folder
    :rtype: str

    Get the absolute path to a file specified relative to the toolbox root
    folder *roboticstoolbox*.
    """

    p = Path(__file__).parent / Path(filename)
    p = p.resolve()
    if not p.exists():
        raise ValueError(f"File '{p}' does not exist")
    print("_path_datafile", __file__, filename, p)
    return str(p)

if __name__ == "__main__":

    a = loadmat("../data/map1.mat")
    print(a)