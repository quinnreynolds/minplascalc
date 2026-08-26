"""Benchmark analytical heat capacity against the former finite difference."""

from __future__ import annotations

import statistics
import time

import numpy as np

from bench.workloads import make_sico

TEMPERATURES = np.linspace(1000.0, 25000.0, 20)
MIXTURES = (0.1, 0.5, 0.9)


def finite_difference_heat_capacity(mixture, relative_step=0.001):
    """Reproduce the heat-capacity implementation before this experiment."""
    temperature = mixture.T
    with mixture._at_temperature(temperature * (1 - relative_step)):
        enthalpy_low = mixture.calculate_enthalpy()
    with mixture._at_temperature(temperature * (1 + relative_step)):
        enthalpy_high = mixture.calculate_enthalpy()
    return (enthalpy_high - enthalpy_low) / (2 * relative_step * temperature)


def sweep(method):
    """Run a tutorial-style SiCO heat-capacity sweep."""
    values = []
    for fraction in MIXTURES:
        mixture = make_sico(fraction)
        for temperature in TEMPERATURES:
            mixture.T = float(temperature)
            values.append(method(mixture))
    return np.asarray(values)


def main():
    """Report alternating timings and the finite-difference discrepancy."""
    methods = {
        "finite difference": finite_difference_heat_capacity,
        "analytical": lambda mixture: mixture.calculate_heat_capacity(),
    }
    for method in methods.values():
        sweep(method)

    samples = {label: [] for label in methods}
    results = {}
    for label, method in list(methods.items()) * 7:
        start = time.perf_counter()
        results[label] = sweep(method)
        samples[label].append(time.perf_counter() - start)

    medians = {
        label: statistics.median(values) for label, values in samples.items()
    }
    for label in methods:
        print(f"{label:<18} median={medians[label]:.6f} s {samples[label]}")
    print(
        "speedup            "
        f"{medians['finite difference'] / medians['analytical']:.3f}x"
    )
    difference = np.abs(results["finite difference"] - results["analytical"])
    print(f"maximum difference {difference.max():.6e} J/(kg K)")


if __name__ == "__main__":
    main()
