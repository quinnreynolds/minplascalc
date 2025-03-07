PROJECT := minplascalc
CONDAFLAGS :=
COV_REPORT := html

default: qa type-check unit-tests

update-env:
	uv pip install -r requirements.txt

qa:
	pre-commit run --all-files

unit-tests:
	uv run pytest tests docs -vv --doctest-glob="*.md" --doctest-glob="*.rst"

type-check:
	uv run mypy .  --exclude docs

docs-build:
	cd docs && rm -fr _api && make clean && make html

# DO NOT EDIT ABOVE THIS LINE, ADD COMMANDS BELOW
