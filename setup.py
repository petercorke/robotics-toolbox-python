from setuptools import setup, find_packages, Extension
import os

# fmt: off
import pip
pip.main(['install', 'numpy>=1.17.4'])
import numpy
# fmt: on

here = os.path.abspath(os.path.dirname(__file__))

req = [
    "numpy>=1.17.4",
    "spatialmath-python~=1.0.0",
    "spatialgeometry~=1.0.0",
    "pgraph-python",
    "scipy",
    "matplotlib",
    "ansitable",
    "swift-sim~=1.0.0",
    "rtb-data",
    "progress",
]

collision_req = ["pybullet"]

vp_req = ["vpython", "numpy-stl", "imageio", "imageio-ffmpeg"]

dev_req = ["pytest", "pytest-cov", "flake8", "pyyaml", "sympy"]

docs_req = [
    "sphinx",
    "sphinx_rtd_theme",
    "sphinx-autorun",
]

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# list all data folders here, to ensure they get packaged

extra_folders = [
    # 'roboticstoolbox/models/URDF/xacro',
    # 'roboticstoolbox/models/DH/meshes',
    # 'roboticstoolbox/data',
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
    "frne",
    sources=[
        "./roboticstoolbox/core/vmath.c",
        "./roboticstoolbox/core/ne.c",
        "./roboticstoolbox/core/frne.c",
    ],
    include_dirs=["./roboticstoolbox/core/"],
)

# eig = "./roboticstoolbox/core/Eigen"
fknm = Extension(
    "fknm",
    sources=[
        "./roboticstoolbox/core/methods.cpp",
        "./roboticstoolbox/core/ik.cpp",
        "./roboticstoolbox/core/linalg.cpp",
        "./roboticstoolbox/core/fknm.cpp",
    ],
    include_dirs=["./roboticstoolbox/core/", numpy.get_include()],
    # define_macros=[("EIGEN_USE_MKL_ALL", "1")],
    # extra_compile_args=["-Werror"],
    # extra_compile_args=["-fopenmp"],
    # extra_compile_args=["-Ofast"],
    # extra_link_args=["-lgomp"],
    # extra_compile_args=["-I/opt/intel/oneapi/mkl/2022.0.2/include"],
    # extra_link_args=[
    #     "-L/opt/intel/oneapi/mkl/2022.0.2/lib/intel64 -lmkl_rt -Wl,--no-as-needed -lpthread -lm -ldl"
    # ],
)

setup(
    name="roboticstoolbox-python",
    version="1.0.2",
    description="A Python library for robotic education and research",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/petercorke/robotics-toolbox-python",
    author="Jesse Haviland and Peter Corke",
    license="MIT",
    classifiers=[
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        # Pick your license as you wish (should match "license" above)
        "License :: OSI Approved :: MIT License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.6",
    project_urls={
        "Documentation": "https://petercorke.github.io/roboticstoolbox-python",
        "Source": "https://github.com/petercorke/roboticstoolbox-python",
        "Tracker": "https://github.com/petercorke/roboticstoolbox-python/issues",
        "Coverage": "https://codecov.io/gh/petercorke/roboticstoolbox-python",
    },
    # cmdclass={"build_ext": build_ext_subclass},
    ext_modules=[frne, fknm],
    keywords="python robotics robotics-toolbox kinematics dynamics"
    " motion-planning trajectory-generation jacobian hessian"
    " control simulation robot-manipulator mobile-robot",
    packages=find_packages(exclude=["tests", "notebooks"]),
    package_data={"roboticstoolbox": extra_files},
    scripts=[
        "roboticstoolbox/bin/rtbtool",
    ],
    entry_points={
        "console_scripts": [
            "eigdemo=roboticstoolbox.examples.eigdemo:main",
            "tripleangledemo=roboticstoolbox.examples.tripleangledemo:main",
            "twistdemo=roboticstoolbox.examples.twistdemo:main",
        ]
    },
    install_requires=req,
    extras_require={
        "collision": collision_req,
        "dev": dev_req,
        "docs": docs_req,
        "vpython": vp_req,
    },
)
