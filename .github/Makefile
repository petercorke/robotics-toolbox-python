.FORCE:

BLUE=\033[0;34m
BLACK=\033[0;30m

help:
	@echo "$(BLUE) make test - run all unit tests"
	@echo " make coverage - run unit tests and coverage report"
	@echo " make docs - build Sphinx documentation"
	@echo " make docupdate - upload Sphinx documentation to GitHub pages"
	@echo " make dist - build dist files"
	@echo " make upload - upload to PyPI"
	@echo " make clean - remove dist and docs build files"
	@echo " make help - this message$(BLACK)"

test:
	pytest

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
	python setup.py sdist

upload: .FORCE
	twine upload dist/*

clean: .FORCE
	(cd docs; make clean)
	-rm -r *.egg-info
	-rm -r dist

