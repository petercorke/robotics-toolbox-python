.FORCE:

BLUE=\033[0;34m
TOML_IMPORT = import sys; tomllib = __import__('tomllib' if sys.version_info >= (3, 11) else 'tomli')
BLACK=\033[0;30m

help:
	@echo "$(BLUE) make test           - run all unit tests"
	@echo " make test-notebooks  - run notebook smoke tests (requires pytest-nbmake)"
	@echo " make coverage        - run unit tests and coverage report"
	@echo " make docs            - build Sphinx documentation"
	@echo " make docupdate       - upload Sphinx documentation to GitHub pages"
	@echo " make dist            - build native wheel + sdist"
	@echo " make wheel-pyodide   - build wasm32/emscripten wheel for Pyodide/JupyterLite"
	@echo " make wheel-pyodide-check - verify wasm wheel tags from filename"
	@echo " make upload          - upload to PyPI"
	@echo " make clean           - remove dist and docs build files"
	@echo " make help            - this message$(BLACK)"



install-collision:
ifeq ($(shell uname -m),arm64)
	CC=/usr/bin/clang CXX=/usr/bin/clang++ \
	CFLAGS="-Wno-error -Wno-implicit-function-declaration" \
	pip install pybullet
else
	pip install pybullet
endif

test:
	python -c "$(TOML_IMPORT); tomllib.load(open('pyproject.toml','rb')); print('pyproject.toml OK')"
	pytest tests/ --ignore=tests/test_blocks.py

test-notebooks:
	MPLBACKEND=Agg pytest --nbmake \
	  $(shell find docs/notebooks -name '*.ipynb' ! -name 'Untitled*.ipynb')

coverage:
	coverage run --omit=\*/test_\* -m unittest
	coverage report

docs: .FORCE
	(cd docs; make html)

docupdate: docs
	git clone https://github.com/petercorke/roboticstoolbox-python.git --branch gh-pages --single-branch gh-pages
	cp -r docs/build/html/. gh-pages
	git add gh-pages
	git commit -m "rebuilt docs"
	git push origin gh-pages

dist: .FORCE
	$(MAKE) test
	python -m build

PYODIDE_BUILD ?= cp313-*

wheel-pyodide: .FORCE
	@echo "Building Pyodide wheel with selector $(PYODIDE_BUILD)"
	@if [ -n "$(PYODIDE_VERSION)" ]; then \
		echo "Using Pyodide runtime $(PYODIDE_VERSION)"; \
		conda run -n dev env CIBW_PLATFORM=pyodide CIBW_BUILD=$(PYODIDE_BUILD) CIBW_PYODIDE_VERSION=$(PYODIDE_VERSION) python -m cibuildwheel --output-dir dist; \
	else \
		echo "Using cibuildwheel default Pyodide runtime (set PYODIDE_VERSION=... to pin)"; \
		conda run -n dev env CIBW_PLATFORM=pyodide CIBW_BUILD=$(PYODIDE_BUILD) python -m cibuildwheel --output-dir dist; \
	fi
	@$(MAKE) wheel-pyodide-check

wheel-pyodide-check: .FORCE
	@wheel=$$(ls -1 dist/*cp313-cp313*wasm32*.whl 2>/dev/null | head -n 1); \
	if [ -z "$$wheel" ]; then \
		wheel=$$(ls -1 dist/*wasm32*.whl 2>/dev/null | head -n 1); \
	fi; \
	if [ -z "$$wheel" ]; then \
		echo "No wasm wheel found under dist/."; \
		exit 1; \
	fi; \
	name=$$(basename "$$wheel"); \
	echo "Built wheel: $$name"; \
	echo "Path: $$wheel"; \
	if ! echo "$$name" | grep -Eq "cp313-cp313-.*wasm32[.]whl$$"; then \
		echo "Wheel tag check failed: expected cp313-cp313 and wasm32 in filename."; \
		exit 1; \
	fi; \
	if ! echo "$$name" | grep -Eq "pyemscripten_[0-9]+_[0-9]+" && ! echo "$$name" | grep -Eq "pyodide_[0-9]+_[0-9]+"; then \
		echo "Wheel tag check failed: expected pyemscripten_<major>_<minor> or pyodide_<major>_<minor> tag in filename."; \
		exit 1; \
	fi; \
	echo "Wheel tags look compatible with current cp313 runtime contract."

upload: .FORCE
	$(eval VERSION := $(shell python -c "$(TOML_IMPORT); print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])"))
	@echo "Uploading version $(VERSION) to PyPI"
	twine upload dist/*
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)
	@echo "Tagged and pushed v$(VERSION)"

clean: .FORCE
	(cd docs; make clean)
	-rm -r *.egg-info
	-rm -r dist

