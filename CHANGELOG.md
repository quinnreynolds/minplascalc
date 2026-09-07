# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Recover log-equilibrium stalls near electronic energy cutoffs by solving
  adjacent discrete branches and validating roots against the original model.
  The hard cutoff and equilibrium tolerance are unchanged; unresolved local
  gaps raise `CutoffConvergenceError` with state and search diagnostics.
- Preserve the input temperature when equilibrium continuation fails, and
  bypass cutoff gaps at unrequested bootstrap or intermediate temperatures.
- Handle equilibrium branch selection for mixtures without electronic levels.

## [1.1.0] - 2026-09-01

### Added

- Added support for LTE mixtures without electrons ([#90](https://github.com/quinnreynolds/minplascalc/pull/90)).
- Added a development-only symbolic derivation and randomized cross-check of
  Devoto's A3--A22 collision-bracket matrices, plus regressions for the reported
  Si-C-O thermal-conductivity outliers
  ([#101](https://github.com/quinnreynolds/minplascalc/pull/101)).
- Added ascending and descending temperature-sweep regression coverage
  ([#95](https://github.com/quinnreynolds/minplascalc/pull/95)).

### Changed

- Optimised the transport and composition paths by sharing collision integrals,
  vectorising electronic and emission sums, and using analytical collision
  recursion derivatives ([#93](https://github.com/quinnreynolds/minplascalc/pull/93)).
- Consolidated equilibrium-derived state and shared transport intermediates,
  including compiled collision-pair evaluation
  ([#102](https://github.com/quinnreynolds/minplascalc/pull/102)).
- Replaced finite-difference composition and enthalpy temperature derivatives
  with analytical equilibrium tangents
  ([#103](https://github.com/quinnreynolds/minplascalc/pull/103)).
- Replaced the default LTE composition solve with the coupled log-particle
  formulation, retaining the particle-number solver as a regression oracle and
  fallback for zero conserved-element totals
  ([#104](https://github.com/quinnreynolds/minplascalc/pull/104)).

### Fixed

- Corrected the misplaced parentheses in the Devoto A11 and A16
  $\\overline Q^{(2,2)}$ coefficients. Conductivity reference values change as
  a result ([#101](https://github.com/quinnreynolds/minplascalc/pull/101)).
- Corrected electronic-level summation so qualifying levels are not skipped
  when input data are not strictly energy-sorted
  ([#93](https://github.com/quinnreynolds/minplascalc/pull/93)).

## [1.0.2] - 2025-05-09

### Added

- Added `CITATION.cff` and updated the bibliography
  ([#84](https://github.com/quinnreynolds/minplascalc/pull/84)).

### Changed

- Renamed `energylevels` to `energy_levels`
  ([#80](https://github.com/quinnreynolds/minplascalc/pull/80)).
- Moved the package to the `src` layout
  ([#81](https://github.com/quinnreynolds/minplascalc/pull/81)).
- Added Numba and optimised the Devoto `q` and `qhat` calculations
  ([#83](https://github.com/quinnreynolds/minplascalc/pull/83)).

## [1.0.1] - 2025-03-20

### Fixed

- `pip install minplascalc` was not working for version 0.7.0 and 1.0.0.
  Update the `[tool.hatch.build.targets.sdist]` section in `pyproject.toml` and move data back into the package ([#73](https://github.com/quinnreynolds/minplascalc/issues/73)).

## [1.0.0] - 2025-03-18

### Added

- NumPy docstring for every functions ([#66](https://github.com/quinnreynolds/minplascalc/issues/66)).
- Documentation workflow, build on top of NumPy docstring ([#66](https://github.com/quinnreynolds/minplascalc/issues/66)).
- Comments almost everywhere to clarify the code ([#66](https://github.com/quinnreynolds/minplascalc/issues/66)).
- Typing of functions/variable (although it is optional in Python, it helps when coding) ([#66](https://github.com/quinnreynolds/minplascalc/issues/66)).
- New Github Action `on-push.yaml`, which check the code quality, run the tests and build the documentation ([#66](https://github.com/quinnreynolds/minplascalc/issues/66)).
- Commands in the `justfile` and show how to use them in `README.rst` ([#72](https://github.com/quinnreynolds/minplascalc/issues/72)).
- New Github Action `test-cov.yaml`, which run test coverage ([#72](https://github.com/quinnreynolds/minplascalc/issues/72)).

### Changed

- Move and transform notebooks into proper Python examples (in the `./examples` folder) ([#66](https://github.com/quinnreynolds/minplascalc/issues/66)).
- Update variable names to be more easily readable ([#66](https://github.com/quinnreynolds/minplascalc/issues/66)).
- Move tests from `./test/unit` to `./tests` (and all tests are passing) ([#66](https://github.com/quinnreynolds/minplascalc/issues/66)).
- Move data from `./minplascalc/species` to `./data/species` ([#66](https://github.com/quinnreynolds/minplascalc/issues/66)).
- **Breaking change -->** Update function names to respect PEP8 convention ([#72](https://github.com/quinnreynolds/minplascalc/issues/72)).
- Update installation procedure in `README.rst` ([#72](https://github.com/quinnreynolds/minplascalc/issues/72)).
- Move reference that where in `README.rst` towards the documentation bibliography (in `./docs/references/_bibliography.rst`) ([#72](https://github.com/quinnreynolds/minplascalc/issues/72)).

### Fixed

## [0.7.0] - 2025-03-05

### Added

- Support for uv for package management ([#65](https://github.com/quinnreynolds/minplascalc/issues/65)).
- pre-commit and ruff, with minimal configuration ([#55](https://github.com/quinnreynolds/minplascalc/issues/61)).

### Changed

### Fixed

- Small errors/miscopies in Devoto1966 q matrix expression ([#61](https://github.com/quinnreynolds/minplascalc/issues/61)).
