"""Vectorised ``Monatomic.internal_partition_function``.

The existing implementation is

.. code:: python

    for J_i, E_i in self.energy_levels:
        if E_i < (self.ionisation_energy - dE):
            electron_partition_function += (2 * J_i + 1) * np.exp(-beta * E_i)
        else:
            break

Note the ``break``: the sum runs over the *prefix* of the level list up to
the first level at or above the cutoff.  ``energy_levels`` is **not** sorted
by energy, so this is not the same as "all levels below the cutoff" -- a
mask-based rewrite would silently change the numbers.  The version below
reproduces the prefix semantics exactly with ``argmax`` on the predicate,
then does one vectorised ``exp`` and one dot product.
"""

import numpy as np

import minplascalc.units as u
from minplascalc.species import Monatomic


def _level_arrays(self):
    """Cache (g, E) as contiguous float arrays on the species instance."""
    cached = self.__dict__.get("_pf_arrays")
    if cached is None:
        levels = self.energy_levels
        g = np.empty(len(levels), dtype=np.float64)
        E = np.empty(len(levels), dtype=np.float64)
        for k, (J_i, E_i) in enumerate(levels):
            g[k] = 2 * J_i + 1
            E[k] = E_i
        cached = (g, E)
        self.__dict__["_pf_arrays"] = cached
    return cached


def internal_partition_function_vec(self, T: float, dE: float) -> float:
    g, E = _level_arrays(self)
    cutoff = self.ionisation_energy - dE

    # Index of the first level that is NOT below the cutoff; the original
    # loop breaks there.  argmax returns 0 when nothing matches, so check.
    over = E >= cutoff
    if over[0]:
        return 0.0
    k = int(np.argmax(over))
    if k == 0:  # no level reaches the cutoff -> use them all
        k = E.shape[0]

    beta = 1 / (u.k_b * T)
    return float(g[:k] @ np.exp(-beta * E[:k]))


def patch():
    """Install the vectorised version; returns an undo callable."""
    original = Monatomic.internal_partition_function
    Monatomic.internal_partition_function = internal_partition_function_vec

    def undo():
        Monatomic.internal_partition_function = original

    return undo
