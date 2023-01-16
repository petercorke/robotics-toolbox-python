from setuptools import setup, Extension
import os
import numpy

extra_folders = [
    "roboticstoolbox/core",
]


def package_files(directory):
    paths = []
    for (pathhere, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", pathhere, filename))
    return paths


extra_files = []
for extra_folder in extra_folders:
    extra_files += package_files(extra_folder)

frne = Extension(
    "roboticstoolbox.frne",
    sources=[
        "./roboticstoolbox/core/vmath.c",
        "./roboticstoolbox/core/ne.c",
        "./roboticstoolbox/core/frne.c",
    ],
    include_dirs=["./roboticstoolbox/core/"],
)

fknm = Extension(
    "roboticstoolbox.fknm",
    sources=[
        "./roboticstoolbox/core/methods.cpp",
        "./roboticstoolbox/core/ik.cpp",
        "./roboticstoolbox/core/linalg.cpp",
        "./roboticstoolbox/core/fknm.cpp",
    ],
    include_dirs=["./roboticstoolbox/core/", numpy.get_include()],
)

setup(
    ext_modules=[frne, fknm],
    package_data={"roboticstoolbox": extra_files},
)
