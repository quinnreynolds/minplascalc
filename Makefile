PROJECT := minplascalc
CONDAFLAGS :=
COV_REPORT := html

default: qa unit-tests

qa:
	pre-commit run --all-files

unit-tests:
	uv run pytest tests docs -vv --doctest-glob="*.md" --doctest-glob="*.rst"

update-requirements:
	uv pip install -r requirements.txt

docs-build:
	cd docs && rm -fr _api && make clean && make html

# DO NOT EDIT ABOVE THIS LINE, ADD COMMANDS BELOW
