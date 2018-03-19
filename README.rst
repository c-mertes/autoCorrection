========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|



.. |requires| image:: https://requires.io/github/matusevi/autonorm/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/matusevi/autonorm/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/matusevi/autonorm/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/matusevi/autonorm

.. |version| image:: https://img.shields.io/pypi/v/autonorm.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/autonorm

.. |commits-since| image:: https://img.shields.io/github/commits-since/matusevi/autonorm/v1.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/matusevi/autonorm/compare/v1.0.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/autonorm.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/autonorm

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/autonorm.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/autonorm

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/autonorm.svg
    :alt: Supported implementations
    :target: https://pypi.python.org/pypi/autonorm


.. end-badges

...

* Free software: MIT license

Install tensorflow and keras
============================

If you have problems with virtualenv, installing using conda may help: 

(Installation of conda: https://conda.io/docs/user-guide/install/index.html)

    pip uninstall virtualenv
    conda install virtualenv

Create virual environment with a name you like (here env-with-tensorflow)

    virtualenv env-with-tensorflow
    
Activate the environment

    source env-with-tensorflow/bin/activate

Install packages 

    pip install tensorflow
    
    pip install keras



Package Installation
============

::

    git clone [this repo]
    
    cd autoCorrect
    make install
    
    #later:
    #pip install autcorrect


Usage
============

::

    #in python:
    python
    import autoCorrect
    import numpy
    counts = numpy.random.negative_binomial(n = 20, p=0.2, size = (10,8))
    sf = numpy.ones((10,1))
    corrector = autoCorrect.correctors.AECorrector()
    c = corrector.correct(counts = counts, size_factors = sf)
    
    #in R:
    library(reticulate)
    autoCorrect <- import("autoCorrect")
    corrected <- autoCorrect$correctors$AECorrector(model_name, model_directory)$correct(COUNTS, SIZE_FACTORS, only_predict=FALSE)

Documentation
=============

https://i12g-gagneurweb.in.tum.de/public/docs/autocorrect/

Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
