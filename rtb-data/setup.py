from setuptools import setup, find_packages

import os

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

release = "1.0.1"

# list all data folders here, to ensure they get packaged

data_folders = [
    "rtbdata",
]

# https://stackoverflow.com/questions/18725137/how-to-obtain-arguments-passed-to-setup-py-from-pip-with-install-option
# but get an error


def package_files(directory):
    paths = []
    for (pathhere, _, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", pathhere, filename))
    return paths


extra_files = []
for data_folder in data_folders:
    extra_files += package_files(data_folder)

print(extra_files)
print(find_packages(exclude=["test_*", "TODO*"]))

setup(
    name="rtb-data",
    version=release,
    # This is a one-line description or tagline of what your project does. This
    # corresponds to the "Summary" metadata field:
    description="Data files for the Robotics Toolbox for Python.",  # TODO
    long_description=long_description,
    long_description_content_type="text/markdown",
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
        "Programming Language :: Python :: 3 :: Only",
    ],
    project_urls={
        "Source": "https://github.com/petercorke/roboticstoolbox-python",
    },
    url="https://github.com/petercorke/roboticstoolbox-python",
    author="Peter Corke",
    author_email="rvc@petercorke.com",  # TODO
    keywords="python robotics",
    # license='MIT',
    package_data={"rtbdata": extra_files},
    # include_package_data=True,
    # data_files = [('mvtbimages', ["../image-data/monalisa.png", "../image-data/street.png"]),],
    packages=find_packages(exclude=["test_*", "TODO*"]),
)
