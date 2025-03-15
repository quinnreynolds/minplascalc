# Configure the shell to use for running commands.
# See https://just.systems/man/en/configuring-the-shell.html
# On Windows, use PowerShell instead of sh.
set windows-shell := ["powershell.exe", "-c"]
# For the other platforms, use the default shell.

default: qa type-check tests

update-env:
	uv lock --upgrade

qa:
	pre-commit run --all-files

tests:
	uv run pytest tests docs -vv --doctest-glob="*.md" --doctest-glob="*.rst"

tests-cov:
	uv run pytest tests docs -vv --doctest-glob="*.md" --doctest-glob="*.rst"  --cov=minplascalc --cov-report=term-missing
	mkdir -p coverage
	uv run coverage-badge -o coverage/coverage.svg

# See https://mypy.readthedocs.io/en/stable/command_line.html for more information.
type-check:
	uv run mypy . --exclude docs

# See https://www.sphinx-doc.org/en/master/man/sphinx-build.html for more information.
docs-build:
	cd docs && rm -fr _api && mkdir -p "source/backreferences" && uv run sphinx-build -M html . _build

docs-build-windows:
	cd docs; Ls _api -recurse -ea silentlycontinue | remove-item -r -fo; md -Force "source/backreferences"; uv run sphinx-build -M html . _build
