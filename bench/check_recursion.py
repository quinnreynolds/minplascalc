import sys

sys.path.insert(0, "bench")
from workloads import make_sico

import minplascalc.functions_transport as ft

m = make_sico(0.5)
m.T = 12000
m.calculate_composition()
sp = m.species
n = len(sp)

# classify pairs the way Qij dispatches
kinds = {}
for i, si in enumerate(sp):
    for j, sj in enumerate(sp):
        if si.charge_number != 0 and sj.charge_number != 0:
            k = "Qc"
        elif sj.name == "e" or si.name == "e":
            k = "Qe"
        elif si.charge_number == 0 and sj.charge_number == 0:
            k = "Qnn"
        elif (
            si.stoichiometry == sj.stoichiometry
            and abs(si.charge_number - sj.charge_number) == 1
        ):
            k = "Qtr/Qin"
        else:
            k = "Qin"
        kinds[k] = kinds.get(k, 0) + 1
print("  species-pair dispatch for one Qij_mix:", kinds, f"(total {n * n})")

counts = {}
for name in ("pot_parameters_ion_neut", "pot_parameters_neut_neut"):
    real = getattr(ft, name)
    counts[name] = [0, real]

    def mk(nm, fn):
        def w(*a, **kw):
            counts[nm][0] += 1
            return fn(*a, **kw)

        return w

    setattr(ft, name, mk(name, real))
try:
    ft.q(m)
finally:
    for nm, (_, r) in counts.items():
        setattr(ft, nm, r)

n_in = counts["pot_parameters_ion_neut"][0]
n_nn = counts["pot_parameters_neut_neut"][0]
print(f"\n  base Qin evaluations per q(): {n_in}")
print(f"  base Qnn evaluations per q(): {n_nn}")
print("\n  If there were no recursion and no (l,s) re-derivation,")
print(
    f"  16 (l,s) x pairs would need: Qin {16 * kinds.get('Qin', 0)}, "
    f"Qnn {16 * kinds.get('Qnn', 0)}"
)
