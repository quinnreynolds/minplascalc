"""Benchmark the experimental log-equilibrium solver against production."""

from __future__ import annotations

import statistics
import time

import numpy as np

from bench.log_equilibrium import LogEquilibriumSystem
from bench.workloads import make_sico
from minplascalc.mixture import LTE

TEMPERATURES = np.linspace(1000.0, 25000.0, 20)
MIXTURES = (0.1, 0.5, 0.9)


def production_sweep():
    """Run the current solver and count nonlinear iterations."""
    original = LTE._LTE__get_reference_energies
    iterations = 0

    def counted(self):
        nonlocal iterations
        iterations += 1
        return original(self)

    LTE._LTE__get_reference_energies = counted
    states = []
    try:
        for fraction in MIXTURES:
            mixture = make_sico(fraction)
            for temperature in TEMPERATURES:
                mixture.T = float(temperature)
                number_densities = mixture.calculate_composition()
                states.append(number_densities / number_densities.sum())
    finally:
        LTE._LTE__get_reference_energies = original
    return np.asarray(states), iterations


def log_sweep():
    """Run independent continuation through the requested temperature path."""
    states = []
    iterations = 0
    evaluations = 0
    solves = 0
    for fraction in MIXTURES:
        system = LogEquilibriumSystem(make_sico(fraction))
        path = system.solve_temperature_path(TEMPERATURES)
        iterations += path.total_iterations
        evaluations += path.total_residual_evaluations
        solves += path.continuation_solves
        states.extend(
            state.number_densities / state.number_densities.sum()
            for state in path.states
        )
    return np.asarray(states), iterations, evaluations, solves


def main():
    """Report alternating timings, convergence, and branch differences."""
    production_sweep()
    log_sweep()
    samples = {"production": [], "log": []}
    results = {}
    for label, function in [
        ("production", production_sweep),
        ("log", log_sweep),
    ] * 7:
        start = time.perf_counter()
        results[label] = function()
        samples[label].append(time.perf_counter() - start)

    production_states, production_iterations = results["production"]
    log_states, log_iterations, log_evaluations, log_solves = results["log"]
    medians = {
        label: statistics.median(values) for label, values in samples.items()
    }
    differences = np.abs(log_states - production_states)
    state_index, species_index = np.unravel_index(
        np.argmax(differences), differences.shape
    )
    mixture_index, temperature_index = divmod(state_index, len(TEMPERATURES))

    print("SiCO equilibrium: 20 temperatures x 3 mixtures")
    for label in ("production", "log"):
        print(f"{label:<12} median={medians[label]:.6f} s {samples[label]}")
    print(f"speedup      {medians['production'] / medians['log']:.3f}x")
    print(f"production iterations       {production_iterations}")
    print(
        f"log iterations/evaluations  {log_iterations}/{log_evaluations} "
        f"across {log_solves} continuation solves"
    )
    print(
        f"iterations per state        "
        f"{production_iterations / len(production_states):.2f} production, "
        f"{log_iterations / log_solves:.2f} log"
    )
    print(f"max absolute mole difference {differences.max():.6e}")
    print(
        "largest difference at "
        f"SiO fraction={MIXTURES[mixture_index]}, "
        f"T={TEMPERATURES[temperature_index]:.1f} K, "
        f"species index={species_index}"
    )
    weights = np.arange(1, production_states.shape[1] + 1)
    print(
        "weighted checksums            "
        f"production={np.sum(production_states @ weights):.12e}, "
        f"log={np.sum(log_states @ weights):.12e}"
    )


if __name__ == "__main__":
    main()
