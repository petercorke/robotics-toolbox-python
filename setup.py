from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

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
    
    keywords='robotics vision arm kinematics ros',

    packages=find_packages(),

    install_requires=['numpy', 'transforms3d']

)
