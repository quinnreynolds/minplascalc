README
======

.. image:: https://github.com/quinnreynolds/minplascalc/actions/workflows/on-push.yml/badge.svg
    :target: https://github.com/quinnreynolds/minplascalc/actions/workflows/on-push.yml/badge.svg

.. image:: https://raw.githubusercontent.com/quinnreynolds/minplascalc/coverage-badge/coverage.svg?raw=true
    :target: https://raw.githubusercontent.com/quinnreynolds/minplascalc/coverage-badge/coverage.svg?raw=true


A simple set of tools in Python 3 for doing calculations of thermal plasma
compositions relevant to metallurgical problems.

*Quinn Reynolds, MINTEK Pyrometallurgy Division, 2018-present*


What is this repository for?
----------------------------

* You're here because you want to calculate plasma compositions,
  thermodynamics, and physical and radiative properties. This package will
  do most of those things for LTE plasmas, to varying degrees of fidelity.
* Version `v1.0.1 <https://github.com/quinnreynolds/minplascalc/releases/latest>`_.


Quick start
-----------

* Simply install the package with ``pip install minplascalc``.
* You should now be able to run the example scripts in the ``examples``
  directory.
* The package is still in development, so expect bugs and changes.


Documentation
-------------

A full set of documentation is available online at
`https://quinnreynolds.github.io/minplascalc/ <https://quinnreynolds.github.io/minplascalc/>`_.


Workflow for developers/contributors
------------------------------------

* First, clone the repository. You'll get the package and some test drivers.
* Since we are using `uv <https://docs.astral.sh/uv/>`_, install it
  by following `instructions on their website <https://docs.astral.sh/uv/getting-started/installation/>`_.
* Run ``uv sync --python 3.13`` to create a virtual environment at ``.venv``,
  with ``python 3.13`` and all the necessary dependencies.
* To test if the package is working, run ``uv run pytest``. All tests should pass.

Next time, you just need to activate the virtual environment with

  * (macOs and Linux) ``source .venv/bin/activate``.

  * (Windows) ``.venv\Scripts\activate``.


Before pushing to GitHub, run the following commands:

1. Update dependencies with ``just update-env``.
2. Run quality assurance checks (code linting) checks with ``just qa``.
3. Run type checks with ``just type-check``.
4. Run unit tests with ``just tests``.
5. Run unit tests with coverage and generate a badge with ``just tests-cov``.
6. Build the documentation with ``just build-docs``.


Contribution guidelines
-----------------------

* Writing tests: TBC
* Code review: TBC
* Other guidelines: TBC


Who do I talk to?
-----------------

* quinnr@mintek.co.za


References
----------

* List of references are available at `the reference section <https://quinnreynolds.github.io/minplascalc/references/_bibliography.html>`_.
* To add a reference, add a new entry to the ``docs/references/_bibliography.rst`` file.


Science behind the code
-----------------------

A brief overview of the science behind the code is available at
`the Background/Theory section <https://quinnreynolds.github.io/minplascalc/theory/Background_Theory.html>`_.
