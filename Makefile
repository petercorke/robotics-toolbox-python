.FORCE:

BLUE=\033[0;34m
BLACK=\033[0;30m

help:
	@echo "$(BLUE) make test           - run all unit tests"
	@echo " make test-notebooks  - run notebook smoke tests (requires pytest-nbmake)"
	@echo " make coverage        - run unit tests and coverage report"
	@echo " make docs            - build Sphinx documentation"
	@echo " make docupdate       - upload Sphinx documentation to GitHub pages"
	@echo " make dist            - build native wheel + sdist"
	@echo " make wheel-pyodide   - build wasm32/emscripten wheel via pyodide-build"
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
	python -c "import tomllib, sys; tomllib.load(open('pyproject.toml','rb')); print('pyproject.toml OK')"
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

wheel-pyodide: .FORCE
	conda run -n dev pyodide build
	@echo "Pyodide wheel written to dist/"

upload: .FORCE
	twine upload dist/*

clean: .FORCE
	(cd docs; make clean)
	-rm -r *.egg-info
	-rm -r dist

