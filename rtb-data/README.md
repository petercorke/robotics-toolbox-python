# Robotics Toolbox for Python - data files

This package includes large data files associated with the Robotics Toolbox for Python (RTB-P).

## Rationale

The data files are provided as a separate package to work around disk space limitations on PyPI.  Including these data with the RTB code adds nearly 200MB to every release, which will blow the PyPI limit quite quickly.  
Since the data doesn't change very much, it's mostly robot models and a few data sets, it makes sense for it to be a standalone package.

## Package contents

| Folder | Purpose                        |
| ------ | ------------------------------ |
| data   | miscellaneous STL files and data sets |
| meshes | STL mesh models for DH robots         |
| xacro  | URDF/xacro models for URDF robots     |      

## Installing the package

You don't need to explicitly install this package, it happens automatically when you when you install RTB-P

```
pip install roboticstoolbox-python
```
since it is a dependency.
