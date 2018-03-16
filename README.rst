========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|

.. |docs| image:: https://readthedocs.org/projects/autonorm/badge/?style=flat
    :target: https://readthedocs.org/projects/autonorm
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/matusevi/autonorm.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/matusevi/autonorm

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/matusevi/autonorm?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/matusevi/autonorm

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

Installation
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
    c = autoCorrect.correctors.AECorrector().correct(counts = counts, size_factors = sf)
    
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
