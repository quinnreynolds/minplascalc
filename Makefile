PROJECT := minplascalc
CONDA := conda
CONDAFLAGS :=
COV_REPORT := html

default: qa unit-tests type-check

qa:
	pre-commit run --all-files

unit-tests:
# python -m pytest tests docs -vv --doctest-glob="*.md" --doctest-glob="*.rst"
# Coverage bugging for now in CI --> comment
	python -m pytest tests docs -vv --cov=. --cov-report=xml --doctest-glob="*.md" --doctest-glob="*.rst"

type-check:
	python -m mypy .  --exclude docs

conda-env-update:
	$(CONDA) install -y -c conda-forge conda-merge
	$(CONDA) run conda-merge environment.yml ci/environment-ci.yml > ci/combined-environment-ci.yml
	$(CONDA) env update $(CONDAFLAGS) -f ci/combined-environment-ci.yml

template-update:
	pre-commit run --all-files cruft -c .pre-commit-config-cruft.yaml

docs-build:
	cd docs && rm -fr _api && make clean && make html

# DO NOT EDIT ABOVE THIS LINE, ADD COMMANDS BELOW
