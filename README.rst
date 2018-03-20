========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - package
      - | |version| |wheel| |supported-versions|
        | |commits-since|


.. |version| image:: https://img.shields.io/pypi/v/autonorm.svg
    :alt: PyPI Package latest release
    :target: https://pypi.python.org/pypi/autonorm

.. |commits-since| image:: https://img.shields.io/github/commits-since/matusevi/autonorm/v1.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/matusevi/autonorm/compare/v1.0.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/autonorm.svg
    :alt: PyPI Wheel
    :target: https://pypi.python.org/pypi/autoCorrect

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/autonorm.svg
    :alt: Supported versions
    :target: https://pypi.python.org/pypi/autoCorrect



.. end-badges



* Free software: MIT license

Activate virtual environment
==================
Together with the autoCorrect package you will get

        'tensorflow',
        'toolz',
        'keras',
        'numpy',
        'kopt',
        'scipy',
        'h5py',
        'sklearn',
        'dask',
        'pandas',
        'matplotlib'

packages automatically installed, if not present.

If you don't wannt to install these packages globally, please use virtual environment.

If you have problems with virtualenv, installing using conda may help:

(Installation of conda: https://conda.io/docs/user-guide/install/index.html)

    pip uninstall virtualenv

    conda install virtualenv

Create virual environment with a name you like (here env-with-autoCorrect)

    virtualenv env-with-autoCorrect

Activate the environment

    source env-with-autoCorrect/bin/activate




Package Installation
============

::

    pip install autoCorrect


Deactivate virtual environment
============

::

    deactivate

Usage
============

::

    #in python:
    python
    import autoCorrect
    import numpy as np
    counts = np.random.negative_binomial(n = 20, p=0.2, size = (10,8))
    sf = np.ones((10,8))
    corrector = autoCorrect.correctors.AECorrector()
    c = corrector.correct(counts = counts, size_factors = sf)

    #in R:
    library(reticulate)
    autoCorrect <- import("autoCorrect")
    corrected <- autoCorrect$correctors$AECorrector(model_name, model_directory)$correct(COUNTS, SIZE_FACTORS, only_predict=FALSE)

Documentation
=============

https://i12g-gagneurweb.in.tum.de/public/docs/autocorrect/


