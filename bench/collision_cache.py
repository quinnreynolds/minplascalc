"""Memoise the (l, s)-independent parts of the collision integrals.

``Qij_mix`` is called 16 times per ``q()`` -- once per (l, s) pair -- and
each call re-derives the interaction-potential parameters for every species
pair.  Those parameters depend only on the species pair (and, for the
charged case, on T and the number densities), never on (l, s), so 15 of
every 16 evaluations are redundant.

This module installs ``functools.lru_cache`` wrappers on the pure helpers to
size that redundancy.  It is a measurement prototype, not a proposed patch:
a real fix would restructure ``Qij_mix`` to sweep (l, s) innermost, which
avoids the cache-invalidation question entirely.
"""

from functools import lru_cache

import minplascalc.functions_transport as ft

_ORIGINALS = {}

# Pure functions of hashable arguments (Species objects hash by identity).
_TO_CACHE = (
    "pot_parameters_ion_neut",
    "pot_parameters_neut_neut",
    "beta",
    "x0_ion_neut",
    "x0_neut_neut",
    "psiconst",
    "sum1",
    "sum2",
    "A",
    "B",
)


def patch(maxsize=8192):
    for name in _TO_CACHE:
        fn = getattr(ft, name, None)
        if fn is None:
            continue
        _ORIGINALS[name] = fn
        setattr(ft, name, lru_cache(maxsize=maxsize)(fn))

    def undo():
        for name, fn in _ORIGINALS.items():
            setattr(ft, name, fn)
        _ORIGINALS.clear()

    return undo
