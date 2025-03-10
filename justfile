default: qa type-check unit-tests

update-env:
	uv lock --upgrade

qa:
	pre-commit run --all-files

unit-tests:
	uv run pytest tests docs -vv --doctest-glob="*.md" --doctest-glob="*.rst"

# See https://mypy.readthedocs.io/en/stable/command_line.html for more information.
type-check:
	uv run mypy . --exclude docs

# See https://www.sphinx-doc.org/en/master/man/sphinx-build.html for more information.
docs-build:
	cd docs && rm -fr _api && uv run sphinx-build -M html . _build


# DO NOT EDIT ABOVE THIS LINE, ADD COMMANDS BELOW
