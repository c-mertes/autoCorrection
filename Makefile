.PHONY: all clean distclean optimize

all: install test

install: dist
	pip install -e .

dist: distclean
	python setup.py sdist

test: clean_test
	tox

distclean: clean

clean: clean_installation clean_test clean_opt

clean_installation:
	rm -rf src/autoCorrect.egg-info/ dist/

clean_test:
	rm -fr .tox/

optimize: run_optimize clean_opt

clean_opt:
	cd optimization; \
	rm -rf 5*; \
	rm slurm*

run_optimize: run_workers
	cd optimization; \
	./train_all.py --notest

test_optimization: run_workers
	cd optimization; \
	./train_all.py

run_workers:
	cd optimization; \
	for run in {1..5};\
	do\
		sbatch worker_script.sh;\
	done	

