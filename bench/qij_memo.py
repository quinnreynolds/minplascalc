"""Memoise Qij_mix on (mixture, T, P, x0, l, s).

All four property calls at a single temperature need the same sixteen
collision-integral matrices, but currently recompute them 71 times:
``viscosity`` 7, ``electrical_conductivity`` 16, ``thermal_conductivity``
48 (it reaches ``q()`` three times -- directly, via ``DTi`` and via
``Dij``).

The collision integrals are a deterministic function of the mixture state,
so this caches them on that state.  It is a measurement prototype: the
structural fix is for the property functions to compute q/qhat once and
pass them down, rather than each re-deriving them.
"""

import minplascalc.functions_transport as ft

_cache: dict = {}
_stats = {"hit": 0, "miss": 0}


def patch(maxentries=64):
    real = ft.Qij_mix
    _cache.clear()
    _stats.update(hit=0, miss=0)

    def cached_Qij_mix(mixture, l, s):
        key = (
            id(mixture),
            mixture.T,
            mixture.P,
            tuple(mixture.x0),
            l,
            s,
        )
        hit = _cache.get(key)
        if hit is not None:
            _stats["hit"] += 1
            return hit
        _stats["miss"] += 1
        if len(_cache) > maxentries * 16:
            _cache.clear()
        val = real(mixture, l, s)
        _cache[key] = val
        return val

    ft.Qij_mix = cached_Qij_mix

    def undo():
        ft.Qij_mix = real
        _cache.clear()

    return undo


def stats():
    return dict(_stats)
