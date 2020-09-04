from setuptools import setup, find_packages, Extension
from os import path
import os

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get the release/version string
with open(path.join(here, 'RELEASE'), encoding='utf-8') as f:
    release = f.read()


def package_files(directory):
    paths = []
    for (pathhere, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', pathhere, filename))
    return paths


extra_files = package_files('ropy/models/xacro')

frne = Extension(
        'frne',
        sources=[
            './ropy/core/vmath.c',
            './ropy/core/ne.c',
            './ropy/core/frne.c'])

setup(
    name='ropy',

    version=release,

    description='A Python library for robot control',

    long_description=long_description,

    long_description_content_type='text/markdown',

    url='https://github.com/jhavl/ropy',

    author='Jesse Haviland',

    license='MIT',

    python_requires='>=3.5',

    ext_modules=[frne],

    keywords='robotics vision arm kinematics ros',

    packages=find_packages(exclude=["tests", "examples"]),

    package_data={'ropy': extra_files},

    include_package_data=True,

    install_requires=[
        'numpy',
        'spatialmath-python>=0.7.1',
        'scipy',
        'matplotlib']

)
