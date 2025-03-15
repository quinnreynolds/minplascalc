README
======

.. image:: https://github.com/quinnreynolds/minplascalc/actions/workflows/on-push.yaml/badge.svg
    :target: https://github.com/quinnreynolds/minplascalc/actions

A simple set of tools in Python 3 for doing calculations of thermal plasma
compositions relevant to metallurgical problems.

*Quinn Reynolds, MINTEK Pyrometallurgy Division, 2018-present*


What is this repository for?
----------------------------

* You're here because you want to calculate plasma compositions,
  thermodynamics, and physical and radiative properties. This package will
  do most of those things for LTE plasmas, to varying degrees of fidelity.
* Version 0.7.0 alpha


Quick start
-----------

* Simply install the package with ``pip install minplascalc``.
* You should now be able to run the example scripts in the ``examples``
  directory.
* The package is still in development, so expect bugs and changes.


Documentation
-------------

A full set of documentation is available at
`https://quinnreynolds.github.io/minplascalc/ <https://quinnreynolds.github.io/minplascalc/>`_.


Workflow for developers/contributors
------------------------------------

* First, clone the repository. You'll get the package and some test drivers.
* Since we are using `uv <https://docs.astral.sh/uv/>`_, install it with
  by following `instructions on their website <https://docs.astral.sh/uv/getting-started/installation/>`_.
* Run ``uv venv --python 3.13`` to create a virtual environment at ``.venv``.
* Activate the virtual environment with
  * (macOs and Linux) ``source .venv/bin/activate``.
  * (Windows) ``.venv\Scripts\activate``.
* Run ``uv sync`` to install the necessary dependencies.
* To test if the package is working, run ``uv run pytest``. All tests should pass.

Before pushing to GitHub, run the following commands:

1. Update dependencies with ``just update-env``.
2. Run quality assurance checks (code linting) checks with ``just qa``.
3. Run type checks with ``just type-check``.
4. Run unit tests with ``just unit-tests``.
5. Build the documentation with ``just docs-build``.


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

* MI Boulos, P Fauchais, and E Pfender. Thermal Plasmas: Fundamentals and
  Applications Volume 1, *Plenum Press*, New York NY, 1994
* S Chapman and TG Cowling. The Mathematical Theory of Non-Uniform Gases
  3\ :sup:`ed`\, *Cambridge University Press*, Cambridge, United Kingdom,
  1970
* JC Stewart and KD Pyatt Jr. Lowering of Ionization Potentials in Plasmas,
  *The Astrophysical Journal*, 144, 1966, 1203-1211
* GJ Dunn and TW Eagar. Calculation of Electrical and Thermal
  Conductivities of Metallurgical Plasmas,
  *Welding Research Council*, Bulletin 357, 1990
* RS Devoto. Transport Properties of Ionized Monatomic Gases,
  *The Physics of Fluids*, 9(6), 1966, 1230-1240
* RS Devoto. Transport Coefficients of Partially Ionized Argon,
  *The Physics of Fluids*, 10(2), 1967, 354-364
* A Laricchiuta, G Colonna, D Bruno, R Celiberto, C Gorse, F Pirani, and
  M Capitelli. Classical transport collision integrals for a Lennard-Jones
  like phenomenological model potential, *Chemical Physical Letters*, 445,
  2007, 133-139
* A Kramida, Yu Ralchenko, J Reader, and NIST ASD Team. NIST Atomic Spectra
  Database (ver. 5.3) [Online],
  *National Institute of Standards and Technology*, Gaithersburg MD,
  http://physics.nist.gov/asd
* PJ Linstrom and WG Mallard (ed). NIST Chemistry WebBook, NIST
  Standard Reference Database Number 69.
  *National Institute of Standards and Technology*, Gaithersburg MD,
  http://webbook.nist.gov/chemistry/
* RD Johnson III (ed). NIST Chemistry WebBook, NIST Computational
  Chemistry Comparison and Benchmark Database Release 22.
  *National Institute of Standards and Technology*, Gaithersburg MD,
  http://cccbdb.nist.gov/
