.FORCE:

BLUE=\033[0;34m
BLACK=\033[0;30m
PYTHON ?= python

help:
	@echo " make dist - build source and wheel distributions"
	@echo " make wheel - build wheel distribution"
	@echo " make upload - upload to PyPI"
	@echo " make clean - remove build artifacts"
	@echo " make help - this message$(BLACK)"


dist: .FORCE
	$(PYTHON) -m build

wheel: .FORCE
	$(PYTHON) -m build --wheel

upload: .FORCE
	twine upload dist/*

clean: .FORCE
	-rm -rf build
	-rm -r *.egg-info
	-rm -rf dist
	-rm -rf .pytest_cache
	-rm -rf .hatch

