# Contributing to the Robotics Toolbox

## Reporting issues

When reporting issues please include as much detail as possible about your
operating system, roboticstoolbox version and python version. Whenever possible, please
also include a brief, self-contained code example that demonstrates the problem.

## Contributing code

Thanks for your interest in contributing code!

We welcome all kinds of contributions including:

+ New features
+ Bug and issue fixes
+ Cleaning, adding or adding to documentation and docstrings
+ Adding or fixing Python types


Keep in mind the following when making your contribution:

+ Keep pull requests to a **single** feature/bug fix. This makes it much easier to review and merge. If you wish to contribure multiple different fixes or features, that means you should make multiple pull requests.

+ For API changes, propose the API change in the discussions first before opening a pull request.

+ Code additions should be formatted using [black](https://pypi.org/project/black/). Our configuration for black can be found in the [pyproject.toml](https://github.com/petercorke/robotics-toolbox-python/blob/master/pyproject.toml) file under the heading `[tool.black]`. Avoid reformatting code using other formatters.

+ Code addition should be linted using [flake8](https://pypi.org/project/flake8/). Our configuration for black can be found in the [pyproject.toml](https://github.com/petercorke/robotics-toolbox-python/blob/master/pyproject.toml) file under the heading `[tool.flake8]`.

+ Any code addition needs to be covered by unit tests and not break existing tests. Our unit tests live in `robotics-toolbox-python/tests/`. You can install the dev dependencies using the command `pip install -e '.[dev,docs]'`. You can run the test suite using the command `pytest --cov=roboticstoolbox/ --cov-report term-missing`. Check the output to make sure your additions have been covered by the unit tests.

+ All methods and classes need to be documented with an appropriate docstring. See our [style guide](https://github.com/petercorke/robotics-toolbox-python/wiki/Documentation-Style-Guide) for documentation. Keep the ordering and formatting as described by the style guide.

+ New additions should be typed appropriately. See our typing [style guide]().
