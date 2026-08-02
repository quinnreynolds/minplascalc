"""Count how often q(), qhat() and calculate_composition() are recomputed.

Each property call re-derives the q-matrix from scratch, and
``thermal_conductivity`` reaches it three ways (directly, via ``DTi`` and
via ``Dij``) at the same temperature and composition.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from workloads import make_sico  # noqa: E402

import minplascalc.functions_transport as ft  # noqa: E402
import minplascalc.mixture as mxmod  # noqa: E402

COUNTS: dict[str, int] = {}


def wrap_module_fn(mod, name):
    real = getattr(mod, name)
    COUNTS[name] = 0

    def wrapper(*a, **kw):
        COUNTS[name] += 1
        return real(*a, **kw)

    setattr(mod, name, wrapper)
    return lambda: setattr(mod, name, real)


def wrap_method(cls, name):
    real = getattr(cls, name)
    COUNTS[name] = 0

    def wrapper(self, *a, **kw):
        COUNTS[name] += 1
        return real(self, *a, **kw)

    setattr(cls, name, wrapper)
    return lambda: setattr(cls, name, real)


def main():
    undos = [
        wrap_module_fn(ft, "q"),
        wrap_module_fn(ft, "qhat"),
        wrap_module_fn(ft, "Qij_mix"),
        wrap_module_fn(ft, "_pair_integrals"),
        wrap_module_fn(ft, "collision_integrals"),
        wrap_module_fn(ft, "Dij"),
        wrap_module_fn(ft, "DTi"),
        wrap_method(mxmod.LTE, "calculate_composition"),
    ]
    try:
        m = make_sico(0.5)
        m.T = 12000
        m.calculate_composition()

        props = [
            ("viscosity", lambda: m.calculate_viscosity()),
            (
                "electrical_conductivity",
                lambda: m.calculate_electrical_conductivity(),
            ),
            (
                "total_emission_coefficient",
                lambda: m.calculate_total_emission_coefficient(),
            ),
            (
                "thermal_conductivity",
                lambda: m.calculate_thermal_conductivity(),
            ),
        ]
        keys = [
            "q",
            "qhat",
            "collision_integrals",
            "_pair_integrals",
            "Dij",
            "DTi",
        ]
        print(f"  {'property':<26s}" + "".join(f"{k:>21s}" for k in keys))
        for label, fn in props:
            m.T = 12000
            m.calculate_composition()
            for k in COUNTS:
                COUNTS[k] = 0
            fn()
            print(
                f"  {label:<26s}" + "".join(f"{COUNTS[k]:>21d}" for k in keys)
            )
    finally:
        for u_ in undos:
            u_()


if __name__ == "__main__":
    main()
