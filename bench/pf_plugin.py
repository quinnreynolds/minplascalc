"""pytest plugin: run the suite with the prototype optimisations applied.

Usage::

    PYTHONPATH=bench MPC_PATCH=pf,cache uv run pytest tests -q -p pf_plugin
"""

import os


def pytest_configure(config):
    which = os.environ.get("MPC_PATCH", "pf")
    parts = {p.strip() for p in which.split(",") if p.strip()}
    config._mpc_undos = []
    if "pf" in parts:
        import pf_vectorised

        config._mpc_undos.append(pf_vectorised.patch())
        print("\n[pf_plugin] vectorised partition function ENABLED")
    if "cache" in parts:
        import collision_cache

        config._mpc_undos.append(collision_cache.patch())
        print("[pf_plugin] collision-parameter cache ENABLED")
    if "memo" in parts:
        import qij_memo

        config._mpc_undos.append(qij_memo.patch())
        print("[pf_plugin] Qij_mix memoisation ENABLED")
    if "analytic" in parts:
        import q_analytic_dropin

        config._mpc_undos.append(q_analytic_dropin.patch())
        print("[pf_plugin] analytic (l,s) derivative ENABLED")


def pytest_unconfigure(config):
    for undo in reversed(getattr(config, "_mpc_undos", [])):
        undo()
