# Robotics Toolbox for Python - data files

[![PyPI version](https://badge.fury.io/py/rtb-data.svg)](https://badge.fury.io/py/rtb-data)
[![Anaconda version](https://anaconda.org/conda-forge/rtb-data/badges/version.svg)](https://anaconda.org/conda-forge/rtb-datan)

<table style="border:0px">
<tr style="border:0px">
<td style="border:0px">
<img src="https://github.com/petercorke/robotics-toolbox-python/raw/master/docs/figs/RTBDataLogo.png" width="200"></td>
<td style="border:0px">
This package includes large data files associated with the Robotics Toolbox for Python (RTB-P).
</td>
</tr>
</table>


## Rationale

The data files are provided as a separate package to work around disk space limitations on PyPI.  Including these data with the RTB code adds nearly 200MB to every release, which will blow the PyPI limit quite quickly.  
Since the data doesn't change very much, it's mostly robot models and a few data sets, it makes sense for it to be a standalone package.

## Package contents

| Folder | Purpose                        |
| ------ | ------------------------------ |
| data   | miscellaneous STL files and data sets |
| meshes | STL mesh models for DH robots         |
| xacro  | URDF/xacro models for URDF robots     |      

## Accessing data within the package

The Toolbox function `path_to_datafile(file)` will return an absolute
`Path` to `file` which is relative to the root of the data package.  For example

```
loadmat("data/map1.mat")   # read rtbdata/data/map1.mat
loadmat("foo.dat")         # read ./foo.dat
loadmat("~/foo.dat")       # read $HOME/foo.dat
```

A matching local file takes precendence over a file in the data package.

## Installing the package

You don't need to explicitly install this package, it happens automatically when you when you install RTB-P

```
pip install roboticstoolbox-python
```
since it is a dependency.
