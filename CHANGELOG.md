# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

- Renamed energylevels -> energy_levels
- Moved package to src layout

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
