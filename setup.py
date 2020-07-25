from setuptools import setup, find_packages, Extension
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

frne = Extension(
        'frne',
        sources=[
            './ropy/core/vmath.c',
            './ropy/core/ne.c',
            './ropy/core/frne.c'])

setup(
    name='ropy',

    version='0.2.1',

    description='A Python library for robot control',

    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/jhavl/ropy',

    author='Jesse Haviland',

    license='MIT',

    python_requires='>=3.2',

    ext_modules=[frne],

    keywords='robotics vision arm kinematics ros',

    packages=find_packages(),

    install_requires=['numpy', 'transforms3d']

)
