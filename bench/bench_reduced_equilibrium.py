"""Benchmark production, full-log, and reduced equilibrium solvers.

This is a research benchmark, not a package entry point.  It follows the
existing benchmark sweep convention (1,000--25,000 K and SiO fractions
0.1/0.5/0.9), and can emit either a compact table or JSON for later analysis.

Example::

    PYTHONPATH=src:. .venv/bin/python bench/bench_reduced_equilibrium.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

from bench.log_equilibrium import LogEquilibriumSystem
from bench.reduced_equilibrium import ReducedEquilibriumSystem
from bench.workloads import make_sico, make_simple
from minplascalc.mixture import LTE

TEMPERATURES = np.linspace(1000.0, 25000.0, 20)
MIXTURES = (0.1, 0.5, 0.9)
DEFAULT_WARMUP = 1
DEFAULT_REPEATS = 3


@dataclass(frozen=True)
class SolverRun:
    """One solver execution over one workload sweep."""

    states: np.ndarray
    seconds: float
    setup_seconds: float
    equilibrium_seconds: float
    species_count: int
    dimension: int
    iterations: int
    residual_evaluations: int
    reconstruction_evaluations: int
    reconstruction_cache_hits: int
    backtracks: int
    solves: int
    failures: int
    max_residual: float
    tangent_seconds: float
    linear_solve_seconds: float
    component_microseconds: dict[str, float]


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregate benchmark result for one named workload."""

    workload: str
    temperature_count: int
    mixture_count: int
    temperatures: tuple[float, ...]
    solver_runs: dict[str, SolverRun]
    median_seconds: dict[str, float]
    median_setup_seconds: dict[str, float]
    median_equilibrium_seconds: dict[str, float]
    median_tangent_seconds: dict[str, float]
    median_linear_solve_seconds: dict[str, float]
    max_mole_fraction_error_vs_full_log: dict[str, float]
    max_log_density_error_vs_full_log: dict[str, float]


def _mole_fractions(states: list[np.ndarray]) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return values / values.sum(axis=1, keepdims=True)


def _finite_max(values: np.ndarray) -> float:
    """Return a NaN sentinel without warning when a solver fully fails."""
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float("nan")


def _median_call_microseconds(function, repeats: int = 31) -> float:
    """Return a small, warmed component timing without affecting sweep time."""
    function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return 1e6 * statistics.median(samples)


def _component_probe(system, state, reduced: bool) -> dict[str, float]:
    """Time representative thermodynamic, assembly, and dense-solve calls."""
    if reduced:
        potentials = state.potentials
        _, jacobian = system.evaluate(potentials)
        assert jacobian is not None
        right_hand_side = np.ones(system.potential_count)
        return {
            "lowering": _median_call_microseconds(
                lambda: system._coupled_lowering(*potentials[-2:])
            ),
            "reconstruction": _median_call_microseconds(
                lambda: system._reconstruct(potentials)
            ),
            "cached_residual_jacobian": _median_call_microseconds(
                lambda: system._evaluate_state(potentials, jacobian=True)
            ),
            "dense_solve": _median_call_microseconds(
                lambda: np.linalg.solve(jacobian, right_hand_side)
            ),
        }
    logs = state.log_particles
    multipliers = state.scaled_multipliers
    _, jacobian = system.evaluate(logs, multipliers)
    assert jacobian is not None
    right_hand_side = np.ones(jacobian.shape[0])
    return {
        "packed_thermodynamics": _median_call_microseconds(
            lambda: system._packed_thermodynamics(logs, derivatives=True)
        ),
        "residual_jacobian": _median_call_microseconds(
            lambda: system.evaluate(logs, multipliers)
        ),
        "dense_solve": _median_call_microseconds(
            lambda: np.linalg.solve(jacobian, right_hand_side)
        ),
    }


def _production_sweep(
    factory: Callable,
    fractions: tuple[float, ...],
    temperatures: np.ndarray,
    pressure: float,
) -> SolverRun:
    """Run production composition and count reference-energy evaluations."""
    states: list[np.ndarray] = []
    iterations = 0
    failures = 0
    species_count = 0
    setup_seconds = 0.0
    equilibrium_seconds = 0.0
    original = LTE._LTE__get_reference_energies

    def counted(self):
        nonlocal iterations
        iterations += 1
        return original(self)

    LTE._LTE__get_reference_energies = counted

    try:
        for fraction in fractions:
            setup_started = time.perf_counter()
            mixture = (
                factory(P=pressure)
                if factory is make_simple
                else factory(fraction, P=pressure)
            )
            setup_seconds += time.perf_counter() - setup_started
            species_count = len(mixture.species)
            equilibrium_started = time.perf_counter()
            for temperature in temperatures:
                mixture.T = float(temperature)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        values = np.asarray(
                            mixture.calculate_composition(), dtype=np.float64
                        )
                    if values.shape != (species_count,) or not np.all(
                        np.isfinite(values)
                    ):
                        raise RuntimeError("non-finite production composition")
                    states.append(values)
                except (RuntimeError, FloatingPointError, ValueError):
                    failures += 1
                    states.append(np.full(species_count, np.nan))
            equilibrium_seconds += time.perf_counter() - equilibrium_started
    finally:
        LTE._LTE__get_reference_energies = original
    return SolverRun(
        states=np.asarray(states),
        seconds=setup_seconds + equilibrium_seconds,
        setup_seconds=setup_seconds,
        equilibrium_seconds=equilibrium_seconds,
        species_count=species_count,
        dimension=species_count,
        iterations=iterations,
        residual_evaluations=0,
        reconstruction_evaluations=0,
        reconstruction_cache_hits=0,
        backtracks=0,
        solves=len(states),
        failures=failures,
        max_residual=float("nan"),
        tangent_seconds=float("nan"),
        linear_solve_seconds=float("nan"),
        component_microseconds={},
    )


def _prototype_sweep(
    factory: Callable,
    fractions: tuple[float, ...],
    temperatures: np.ndarray,
    pressure: float,
    reduced: bool,
) -> SolverRun:
    """Run one prototype using its independent continuation path."""
    states: list[np.ndarray] = []
    iterations = 0
    evaluations = 0
    reconstruction_evaluations = 0
    reconstruction_cache_hits = 0
    backtracks = 0
    solves = 0
    failures = 0
    max_residual = 0.0
    species_count = 0
    dimension = 0
    tangent_seconds = 0.0
    linear_solve_seconds = 0.0
    setup_seconds = 0.0
    equilibrium_seconds = 0.0
    component_totals: dict[str, float] = {}
    component_probes = 0
    for fraction in fractions:
        setup_started = time.perf_counter()
        mixture = (
            factory(P=pressure)
            if factory is make_simple
            else factory(fraction, P=pressure)
        )
        system = (
            ReducedEquilibriumSystem(mixture, coupled_ionization_lowering=True)
            if reduced
            else LogEquilibriumSystem(mixture)
        )
        setup_seconds += time.perf_counter() - setup_started
        species_count = system.species_count
        dimension = (
            system.potential_count
            if reduced
            else system.species_count + system.constraint_count
        )
        original_solve = np.linalg.solve

        def timed_solve(matrix, vector):
            nonlocal linear_solve_seconds
            solve_started = time.perf_counter()
            try:
                return original_solve(matrix, vector)
            finally:
                linear_solve_seconds += time.perf_counter() - solve_started

        np.linalg.solve = timed_solve
        path = None
        equilibrium_started = time.perf_counter()
        try:
            path = (
                system.solve_temperature_path(
                    temperatures,
                    bootstrap_temperature=12000.0,
                    max_temperature_step=1000.0,
                    method="least_squares",
                    tolerance=1e-9,
                )
                if reduced
                else system.solve_temperature_path(
                    temperatures,
                    bootstrap_temperature=12000.0,
                    maximum_temperature_step=1000.0,
                    tolerance=1e-9,
                )
            )
            states.extend(state.number_densities for state in path.states)
            iterations += path.total_iterations
            evaluations += path.total_residual_evaluations
            if reduced:
                reconstruction_evaluations += (
                    path.total_reconstruction_evaluations
                )
                reconstruction_cache_hits += (
                    path.total_reconstruction_cache_hits
                )
            backtracks += 0 if not reduced else path.total_backtracks
            solves += path.continuation_solves
            max_residual = max(
                max_residual,
                max(state.residual_norm for state in path.states),
            )
        except (RuntimeError, ValueError, FloatingPointError):
            failures += 1
            states.extend(np.full(species_count, np.nan) for _ in temperatures)
        finally:
            equilibrium_seconds += time.perf_counter() - equilibrium_started
            np.linalg.solve = original_solve
        if path is not None:
            tangent_started = time.perf_counter()
            for state, temperature in zip(path.states, temperatures):
                system.mixture.T = float(temperature)
                system.temperature_tangent(state)
            tangent_seconds += time.perf_counter() - tangent_started
            middle = len(path.states) // 2
            system.mixture.T = float(temperatures[middle])
            for name, value in _component_probe(
                system, path.states[middle], reduced
            ).items():
                component_totals[name] = (
                    component_totals.get(name, 0.0) + value
                )
            component_probes += 1
    return SolverRun(
        states=np.asarray(states),
        seconds=setup_seconds + equilibrium_seconds,
        setup_seconds=setup_seconds,
        equilibrium_seconds=equilibrium_seconds,
        species_count=species_count,
        dimension=dimension,
        iterations=iterations,
        residual_evaluations=evaluations,
        reconstruction_evaluations=reconstruction_evaluations,
        reconstruction_cache_hits=reconstruction_cache_hits,
        backtracks=backtracks,
        solves=solves,
        failures=failures,
        max_residual=max_residual if solves else float("nan"),
        tangent_seconds=tangent_seconds,
        linear_solve_seconds=linear_solve_seconds,
        component_microseconds={
            name: value / component_probes
            for name, value in component_totals.items()
        },
    )


def _run_once(
    factory: Callable,
    name: str,
    fractions: tuple[float, ...],
    temperatures: np.ndarray,
    pressure: float,
) -> dict[str, SolverRun]:
    """Run all solvers once, returning results and wall-clock timings."""
    functions = {
        "production": lambda: _production_sweep(
            factory, fractions, temperatures, pressure
        ),
        "full_log": lambda: _prototype_sweep(
            factory, fractions, temperatures, pressure, reduced=False
        ),
        "reduced": lambda: _prototype_sweep(
            factory, fractions, temperatures, pressure, reduced=True
        ),
    }
    runs: dict[str, SolverRun] = {}
    for solver, function in functions.items():
        runs[solver] = function()
    return runs


def benchmark(
    *,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
    temperatures: np.ndarray = TEMPERATURES,
    pressure: float = 101325.0,
) -> tuple[BenchmarkSummary, ...]:
    """Run warmups and repeated sweeps for oxygen and SiCO workloads."""
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats positive")
    temperatures = np.asarray(temperatures, dtype=np.float64)
    workloads = (
        ("oxygen", make_simple, (0.0,)),
        ("sico", make_sico, MIXTURES),
    )
    summaries = []
    for name, factory, fractions in workloads:
        for _ in range(warmup):
            _run_once(factory, name, fractions, temperatures, pressure)
        repetitions = []
        for _ in range(repeats):
            repetitions.append(
                _run_once(factory, name, fractions, temperatures, pressure)
            )
        selected = repetitions[-1]
        median_seconds = {
            solver: statistics.median(
                run[solver].seconds for run in repetitions
            )
            for solver in selected
        }
        median_setup_seconds = {
            solver: statistics.median(
                run[solver].setup_seconds for run in repetitions
            )
            for solver in selected
        }
        median_equilibrium_seconds = {
            solver: statistics.median(
                run[solver].equilibrium_seconds for run in repetitions
            )
            for solver in selected
        }
        median_tangent_seconds = {
            solver: statistics.median(
                run[solver].tangent_seconds for run in repetitions
            )
            for solver in selected
        }
        median_linear_solve_seconds = {
            solver: statistics.median(
                run[solver].linear_solve_seconds for run in repetitions
            )
            for solver in selected
        }
        full = _mole_fractions(selected["full_log"].states)
        errors: dict[str, float] = {}
        log_errors: dict[str, float] = {}
        for solver in ("production", "reduced"):
            actual = _mole_fractions(selected[solver].states)
            with np.errstate(invalid="ignore", divide="ignore"):
                difference = np.abs(actual - full)
                log_difference = np.abs(
                    np.log(selected[solver].states)
                    - np.log(selected["full_log"].states)
                )
            errors[solver] = _finite_max(difference)
            log_errors[solver] = _finite_max(log_difference)
        summaries.append(
            BenchmarkSummary(
                workload=name,
                temperature_count=len(temperatures),
                mixture_count=len(fractions),
                temperatures=tuple(float(value) for value in temperatures),
                solver_runs=selected,
                median_seconds=median_seconds,
                median_setup_seconds=median_setup_seconds,
                median_equilibrium_seconds=median_equilibrium_seconds,
                median_tangent_seconds=median_tangent_seconds,
                median_linear_solve_seconds=median_linear_solve_seconds,
                max_mole_fraction_error_vs_full_log=errors,
                max_log_density_error_vs_full_log=log_errors,
            )
        )
    return tuple(summaries)


def _json_ready(summary: BenchmarkSummary) -> dict:
    """Convert arrays and dataclasses to JSON-safe values."""
    value = asdict(summary)
    for run in value["solver_runs"].values():
        run["states"] = run["states"].tolist()
    return value


def _print_table(summaries: tuple[BenchmarkSummary, ...]) -> None:
    """Print a compact human-readable benchmark report."""
    for summary in summaries:
        print(
            f"{summary.workload}: {summary.temperature_count} temperatures "
            f"x {summary.mixture_count} mixture(s)"
        )
        print(
            "solver        setup s   equilibrium s  end-to-end s  dimension  "
            "iterations  evaluations  reconstructions  cache hits  solves  "
            "failures  tangent s  dense s"
        )
        for solver, run in summary.solver_runs.items():
            print(
                f"{solver:<12} {summary.median_setup_seconds[solver]:9.6f} "
                f"{summary.median_equilibrium_seconds[solver]:13.6f} "
                f"{summary.median_seconds[solver]:12.6f} {run.dimension:10d} "
                f"{run.iterations:11d} {run.residual_evaluations:11d} "
                f"{run.reconstruction_evaluations:15d} "
                f"{run.reconstruction_cache_hits:10d} "
                f"{run.solves:7d} {run.failures:9d} "
                f"{summary.median_tangent_seconds[solver]:10.6f} "
                f"{summary.median_linear_solve_seconds[solver]:8.6f}"
            )
        production_seconds = summary.median_seconds["production"]
        full_seconds = summary.median_seconds["full_log"]
        reduced_seconds = summary.median_seconds["reduced"]
        print(
            "end-to-end speed relative to production: "
            f"full_log={production_seconds / full_seconds:.3f}x, "
            f"reduced={production_seconds / reduced_seconds:.3f}x; "
            f"reduced/full_log={reduced_seconds / full_seconds:.3f}x slower"
        )
        print(
            "accuracy vs full_log: "
            f"production max |dx|="
            f"{summary.max_mole_fraction_error_vs_full_log['production']:.3e}"
            ", "
            f"reduced max |dx|="
            f"{summary.max_mole_fraction_error_vs_full_log['reduced']:.3e}; "
            f"reduced max |dlog n|="
            f"{summary.max_log_density_error_vs_full_log['reduced']:.3e}"
        )
        print(
            "notes: tangent time is excluded from end-to-end; dense time "
            "counts Python-visible numpy.linalg.solve calls and does not "
            "include SciPy trust-region internals"
        )
        for solver in ("full_log", "reduced"):
            components = summary.solver_runs[solver].component_microseconds
            formatted = ", ".join(
                f"{name}={value:.2f} us" for name, value in components.items()
            )
            print(f"{solver} representative components: {formatted}")
        print()


def main() -> None:
    """Parse options and run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    summaries = benchmark(warmup=args.warmup, repeats=args.repeats)
    if args.as_json:
        print(json.dumps([_json_ready(summary) for summary in summaries]))
    else:
        _print_table(summaries)


if __name__ == "__main__":
    main()
