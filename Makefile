.PHONY: all clean distclean

all: install test

install: dist
	pip install -e .

dist: distclean
	python setup.py sdist

test: clean_test
	tox

distclean: clean

clean: clean_installation clean_test

clean_installation:
	rm -rf src/autoCorrect.egg-info/ dist/

clean_test:
	rm -fr .tox/
