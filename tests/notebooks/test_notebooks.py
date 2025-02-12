"""Test file for running all notebooks in the project.

It will find all `.ipynb` files in the current directory, ignoring directories
which start with `.` and any files which match patterns found in the file `.testignore`.
"""

import pathlib

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

# Default search path is the current directory
searchpath = pathlib.Path(".")

# Read patterns from .testignore file
ignores = [line.strip() for line in open(".testignore") if line.strip()]

# Ignore hidden folders (startswith('.')) and files matching ignore patterns
notebooks = [
    notebook
    for notebook in searchpath.glob("**/*.ipynb")
    if not (
        any(parent.startswith(".") for parent in notebook.parent.parts)
        or any(notebook.match(pattern) for pattern in ignores)
    )
]

notebooks.sort()
ids = [str(n) for n in notebooks]


@pytest.mark.parametrize("notebook", notebooks, ids=ids)
def test_run_notebook(notebook):
    """Read and execute notebook.

    The method here is directly from the nbconvert docs.
    Note that there is no error handling in this file as any errors will be
    caught by pytest.
    """
    with notebook.open() as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": notebook.parent}})
