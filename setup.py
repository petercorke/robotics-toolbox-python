from setuptools import setup, find_packages, Extension
from os import path
import os

here = path.abspath(path.dirname(__file__))

req = [
    'numpy',
    'spatialmath-python>=0.8.2',
    'scipy',
    'matplotlib'
]

vp_req = [
    'vpython',
    'numpy-stl'
]

dev_req = [
    'pytest',
    'pytest-cov',
    'flake8',
    'pyyaml',
]

docs_req = [
    'sphinx',
    'sphinx_rtd_theme',
    'sphinx_markdown_tables'
]

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the release/version string
with open(path.join(here, 'RELEASE'), encoding='utf-8') as f:
    release = f.read()


def package_files(directory):
    paths = []
    for (pathhere, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', pathhere, filename))
    return paths


extra_files = package_files('roboticstoolbox/models/xacro')

frne = Extension(
        'frne',
        sources=[
            './roboticstoolbox/core/vmath.c',
            './roboticstoolbox/core/ne.c',
            './roboticstoolbox/core/frne.c'])

setup(
    name='roboticstoolbox',

    version=release,

    description='A Python library for robot control',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/petercorke/robotics-toolbox-python',

    author='Jesse Haviland',

    license='MIT',

    python_requires='>=3.5',

    ext_modules=[frne],

    keywords='robotics vision arm kinematics ros',

    packages=find_packages(exclude=["tests", "examples"]),

    package_data={'roboticstoolbox': extra_files},

    include_package_data=True,

    install_requires=req,

    extras_require={
        'dev': dev_req,
        'docs': docs_req,
        'vpython': vp_req
    }
)
