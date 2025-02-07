"""Minplascalc."""

try:
    # NOTE: the `version.py` file must not be present in the git repository
    #   as it is generated by setuptools at install time
    from .version import __version__
except ImportError:  # pragma: no cover
    # Local copy or not installed with setuptools
    __version__ = "999"

from . import mixture, parsers, species

__all__ = ["__version__", "species", "mixture", "parsers"]
