default: qa type-check unit-tests

update-env:
	uv lock --upgrade

qa:
	pre-commit run --all-files

unit-tests:
	uv run pytest tests docs -vv --doctest-glob="*.md" --doctest-glob="*.rst"

type-check:
	uv run mypy .  --exclude docs

docs-build:
	cd docs && rm -fr _api && make clean && make html

# DO NOT EDIT ABOVE THIS LINE, ADD COMMANDS BELOW
