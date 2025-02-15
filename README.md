# Minplascalc 🍚⚡

[![CI](https://github.com/pag1pag/minplascalc/actions/workflows/on-push.yml/badge.svg)](https://github.com/pag1pag/minplascalc/actions/workflows/on-push.yml/)

A simple set of tools in Python 3 for doing calculations of thermal plasma
compositions relevant to metallurgical problems.

*Quinn Reynolds, MINTEK Pyrometallurgy Division, 2018-present*

## What is this repository for?

- You're here because you want to calculate plasma compositions,
  thermodynamics, and physical and radiative properties. This package will
  do most of those things for LTE plasmas, to varying degrees of fidelity.
- Version 0.x alpha

## Quick Start

This project used [conda](https://anaconda.org/) to install all dependencies.

Clone the repository and create an isolated environment :

```
git clone https://github.com/pag1pag/minplascalc
cd minplascalc
conda env create -n minplascalc-env -f environment.yml
conda activate minplascalc-env
pip install -e .         # install in editable mode
```

Run in Python:

```python
>>> import minplascalc
```

## Workflow for developers/contributors

Clone the repository :

```
git clone https://github.com/pag1pag/minplascalc
cd minplascalc
```

For best experience create a new conda environment (e.g. minplascalc-env) with Python 3.11:

```
conda create -n minplascalc-env -c conda-forge python=3.11 -y
conda activate minplascalc-env
```

Before pushing to GitHub, run the following commands:

1. Update conda environment: `make conda-env-update`
1. Install this package in editable mode: `pip install -e .`
1. (optional) Run quality assurance checks (code linting): `make qa`
1. (optional) Run tests: `make unit-tests`
1. (optional) Run the static type checker: `make type-check`
1. (optional) Build the documentation (see [Sphinx tutorial](https://www.sphinx-doc.org/en/master/tutorial/)): `make docs-build`

If using Windows, `make` is not available by default. Either install it
([for instance with Chocolatey](https://stackoverflow.com/questions/32127524/how-to-install-and-use-make-in-windows)),
or open the [Makefile](./Makefile) and execute the lines therein manually.

## Contribution guidelines

- Writing tests: TBC
- Code review: TBC
- Other guidelines: TBC

## Who do I talk to?

- quinnr@mintek.co.za

## References

- MI Boulos, P Fauchais, and E Pfender. Thermal Plasmas: Fundamentals and
  Applications Volume 1, *Plenum Press*, New York NY, 1994
- S Chapman and TG Cowling. The Mathematical Theory of Non-Uniform Gases
  3\\ :sup:`ed`, *Cambridge University Press*, Cambridge, United Kingdom,
  1970
- JC Stewart and KD Pyatt Jr. Lowering of Ionization Potentials in Plasmas,
  *The Astrophysical Journal*, 144, 1966, 1203-1211
- GJ Dunn and TW Eagar. Calculation of Electrical and Thermal
  Conductivities of Metallurgical Plasmas,
  *Welding Research Council*, Bulletin 357, 1990
- RS Devoto. Transport Properties of Ionized Monatomic Gases,
  *The Physics of Fluids*, 9(6), 1966, 1230-1240
- RS Devoto. Transport Coefficients of Partially Ionized Argon,
  *The Physics of Fluids*, 10(2), 1967, 354-364
- A Laricchiuta, G Colonna, D Bruno, R Celiberto, C Gorse, F Pirani, and
  M Capitelli. Classical transport collision integrals for a Lennard-Jones
  like phenomenological model potential, *Chemical Physical Letters*, 445,
  2007, 133-139
- A Kramida, Yu Ralchenko, J Reader, and NIST ASD Team. NIST Atomic Spectra
  Database (ver. 5.3) [Online],
  *National Institute of Standards and Technology*, Gaithersburg MD,
  http://physics.nist.gov/asd
- PJ Linstrom and WG Mallard (ed). NIST Chemistry WebBook, NIST
  Standard Reference Database Number 69.
  *National Institute of Standards and Technology*, Gaithersburg MD,
  http://webbook.nist.gov/chemistry/
- RD Johnson III (ed). NIST Chemistry WebBook, NIST Computational
  Chemistry Comparison and Benchmark Database Release 22.
  *National Institute of Standards and Technology*, Gaithersburg MD,
  http://cccbdb.nist.gov/
