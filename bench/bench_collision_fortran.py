"""Compare scalar, Numba, and Fortran collision-pair kernels.

The Fortran implementation consumes the same numeric pair descriptors as the
production Numba kernel. Build products live in a temporary directory, so this
benchmark leaves the source tree clean.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import subprocess
import sys
import tempfile
import time
import weakref
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from workloads import (  # noqa: E402
    make_sico,
    make_simple,
    sico_transport_sweep,
)

from minplascalc import functions_transport as ft  # noqa: E402

SOURCE = Path(__file__).with_name("fortran_collision_kernel.f90")


def compile_fortran(build_dir: Path, fast_math: bool) -> Path:
    """Compile the Fortran benchmark kernel into a temporary shared library."""
    suffix = "fast" if fast_math else "strict"
    library = build_dir / f"libcollision_{suffix}.dylib"
    command = [
        "gfortran",
        "-O3",
        "-march=native",
        "-fPIC",
        "-shared",
        "-J",
        str(build_dir),
        str(SOURCE),
        "-o",
        str(library),
    ]
    if sys.platform == "darwin":
        # OpenModelica may put a non-Apple ld first on PATH. GCC's -B option
        # keeps the benchmark tied to the system Mach-O linker.
        command[1:1] = ["-B/usr/bin/"]
    if fast_math:
        command[1:1] = ["-ffast-math", "-funroll-loops", "-flto"]
    subprocess.run(command, check=True)
    return library


class FortranCollisionKernel:
    """ctypes binding with descriptors converted once, outside timed calls."""

    def __init__(self, library: ctypes.CDLL, model, moments):
        self.species_count = model.kinds.shape[0]
        self.moments = np.ascontiguousarray(moments, dtype=np.int32)
        self.kinds = np.ascontiguousarray(model.kinds, dtype=np.int32)
        self.charges = np.ascontiguousarray(model.charges, dtype=np.int32)
        self.fit_parameters = np.ascontiguousarray(model.fit_parameters)
        self.electron_parameters = np.ascontiguousarray(
            model.electron_parameters
        )
        self.electron_gamma_ratios = np.ascontiguousarray(
            model.electron_gamma_ratios
        )
        self.resonant = np.ascontiguousarray(model.resonant, dtype=np.int32)
        self.resonant_parameters = np.ascontiguousarray(
            model.resonant_parameters
        )
        self.neutral_table = np.ascontiguousarray(ft.c_nn)
        self.ion_table = np.ascontiguousarray(ft.c_in)
        self.psi_values = np.ascontiguousarray(ft._PSICONST_ARRAY)
        self.sum1_values = np.ascontiguousarray(ft._SUM1_ARRAY)
        self.sum2_values = np.ascontiguousarray(ft._SUM2_ARRAY)
        self.electron_index = model.electron_index + 1

        self.function = library.collision_integrals_fortran
        self.function.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            *([ctypes.c_void_p] * 15),
        ]
        self.function.restype = None

        self._descriptor_addresses = [
            array.ctypes.data
            for array in (
                self.moments,
                self.kinds,
                self.charges,
                self.fit_parameters,
                self.electron_parameters,
                self.electron_gamma_ratios,
                self.resonant,
                self.resonant_parameters,
                self.neutral_table,
                self.ion_table,
                self.psi_values,
                self.sum1_values,
                self.sum2_values,
            )
        ]

    def run_into(
        self,
        number_densities: np.ndarray,
        temperature: float,
        values: np.ndarray,
    ) -> None:
        """Run the kernel into an existing C-contiguous output array."""
        densities = np.ascontiguousarray(number_densities)
        self.function(
            self.species_count,
            len(self.moments),
            temperature,
            self.electron_index,
            densities.ctypes.data,
            *self._descriptor_addresses,
            values.ctypes.data,
        )

    def evaluate_raw(
        self, number_densities: np.ndarray, temperature: float
    ) -> np.ndarray:
        """Allocate and return the same array layout as the Numba kernel."""
        values = np.empty(
            (len(self.moments), self.species_count, self.species_count)
        )
        self.run_into(number_densities, temperature, values)
        return values

    def evaluate_dict(self, number_densities, temperature):
        """Return the production API's moment-to-matrix mapping."""
        values = self.evaluate_raw(number_densities, temperature)
        return {
            tuple(moment): values[k] for k, moment in enumerate(self.moments)
        }


def numba_raw(model, number_densities, temperature, moments):
    """Call the production Numba kernel without dictionary construction."""
    return ft._collision_integrals_kernel(
        number_densities,
        temperature,
        moments,
        model.kinds,
        model.charges,
        model.electron_index,
        model.fit_parameters,
        model.electron_parameters,
        model.electron_gamma_ratios,
        model.resonant,
        model.resonant_parameters,
    )


def scalar_raw(mixture, state, moments):
    """Evaluate the retained Python pair-loop reference into dense arrays."""
    count = len(mixture.species)
    values = np.empty((len(moments), count, count))
    for i, (density_i, species_i) in enumerate(
        zip(state.number_densities, mixture.species)
    ):
        for j, (density_j, species_j) in enumerate(
            zip(state.number_densities, mixture.species)
        ):
            pair = ft._pair_integrals(
                species_i,
                density_i,
                species_j,
                density_j,
                state.T,
                tuple(map(tuple, moments)),
            )
            for k, moment in enumerate(moments):
                values[k, i, j] = pair[tuple(moment)]
    return values


def median_call(
    function: Callable[[], object], calls: int, batches=7
) -> float:
    """Return median wall time per call from several timed batches."""
    samples = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(calls):
            function()
        samples.append((time.perf_counter() - start) / calls)
    return statistics.median(samples)


def max_relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    """Return maximum relative error, treating exact zero pairs separately."""
    nonzero = expected != 0
    relative = np.abs(
        (actual[nonzero] - expected[nonzero]) / expected[nonzero]
    )
    zero_error = np.max(np.abs(actual[~nonzero]), initial=0.0)
    return max(float(np.max(relative, initial=0.0)), float(zero_error))


def validate_kernel(library: ctypes.CDLL, moments: np.ndarray) -> float:
    """Check both mixtures over a representative temperature range."""
    maximum = 0.0
    for factory in (make_simple, make_sico):
        for temperature in (2000.0, 12000.0, 25000.0):
            mixture = factory(T=temperature)
            state = mixture._equilibrium_state()
            model = mixture._collision_model()
            fortran = FortranCollisionKernel(library, model, moments)
            expected = numba_raw(
                model, state.number_densities, state.T, moments
            )
            actual = fortran.evaluate_raw(state.number_densities, state.T)
            np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=0.0)
            maximum = max(maximum, max_relative_error(actual, expected))
    return maximum


def benchmark_kernel(library_path: Path, label: str) -> tuple[float, float]:
    """Check and time one Fortran compiler configuration."""
    library = ctypes.CDLL(str(library_path))
    moments = np.ascontiguousarray(ft.LS_PAIRS, dtype=np.int32)
    mixture = make_sico(0.5, T=12000.0)
    state = mixture._equilibrium_state()
    model = mixture._collision_model()
    fortran = FortranCollisionKernel(library, model, moments)
    maximum_error = validate_kernel(library, moments)

    reusable = np.empty(
        (len(moments), len(mixture.species), len(mixture.species))
    )
    allocated_time = median_call(
        lambda: fortran.evaluate_raw(state.number_densities, state.T), 1000
    )
    reused_time = median_call(
        lambda: fortran.run_into(state.number_densities, state.T, reusable),
        1000,
    )
    print(
        f"{label:<24} {allocated_time * 1e6:9.3f} us "
        f"(reused output {reused_time * 1e6:9.3f} us, "
        f"max rel {maximum_error:.3e})"
    )
    return allocated_time, reused_time


def benchmark_end_to_end(library_path: Path) -> tuple[float, float]:
    """Alternate Numba and Fortran in the tutorial-10 transport workload."""
    library = ctypes.CDLL(str(library_path))
    original_evaluate = ft._CollisionModel.evaluate
    wrappers = weakref.WeakKeyDictionary()

    def fortran_evaluate(model, number_densities, temperature, moments):
        key = tuple(moments)
        by_moment = wrappers.setdefault(model, {})
        wrapper = by_moment.get(key)
        if wrapper is None:
            wrapper = FortranCollisionKernel(library, model, moments)
            by_moment[key] = wrapper
        return wrapper.evaluate_dict(number_densities, temperature)

    samples = {"numba": [], "fortran": []}
    checksums = {}
    # Warm both routes before alternating them to reduce drift bias.
    sico_transport_sweep(n_T=2, n_mixtures=1)
    ft._CollisionModel.evaluate = fortran_evaluate
    try:
        sico_transport_sweep(n_T=2, n_mixtures=1)
    finally:
        ft._CollisionModel.evaluate = original_evaluate

    for label in ["numba", "fortran"] * 5:
        ft._CollisionModel.evaluate = (
            original_evaluate if label == "numba" else fortran_evaluate
        )
        try:
            start = time.perf_counter()
            result = sico_transport_sweep()
            samples[label].append(time.perf_counter() - start)
            checksums[label] = sum(sum(row) for row in result)
        finally:
            ft._CollisionModel.evaluate = original_evaluate

    medians = {
        label: statistics.median(values) for label, values in samples.items()
    }
    print("\n60-state tutorial-10 transport sweep")
    for label in ("numba", "fortran"):
        print(f"{label:<24} {medians[label]:9.6f} s  {samples[label]}")
    speedup = medians["numba"] / medians["fortran"]
    print(f"speedup                  {speedup:.3f}x")
    print(f"checksums                {checksums}")
    return medians["numba"], medians["fortran"]


def main() -> None:
    """Build, verify, and benchmark both conservative and floor variants."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-end-to-end",
        action="store_true",
        help="Only run the isolated collision-kernel comparison.",
    )
    args = parser.parse_args()

    moments = np.ascontiguousarray(ft.LS_PAIRS, dtype=np.int32)
    mixture = make_sico(0.5, T=12000.0)
    state = mixture._equilibrium_state()
    model = mixture._collision_model()
    # Trigger compilation before measuring Numba.
    numba_raw(model, state.number_densities, state.T, moments)

    scalar_time = median_call(
        lambda: scalar_raw(mixture, state, moments), 20, batches=5
    )
    numba_time = median_call(
        lambda: numba_raw(model, state.number_densities, state.T, moments),
        1000,
    )
    print("16-species, 16-moment collision evaluation")
    print(f"{'scalar pair loop':<24} {scalar_time * 1e6:9.3f} us")
    print(f"{'Numba kernel':<24} {numba_time * 1e6:9.3f} us")

    with tempfile.TemporaryDirectory(prefix="minplas-fortran-") as temporary:
        build_dir = Path(temporary)
        strict_library = compile_fortran(build_dir, fast_math=False)
        fast_library = compile_fortran(build_dir, fast_math=True)
        strict_time, _ = benchmark_kernel(strict_library, "Fortran -O3")
        fast_time, _ = benchmark_kernel(fast_library, "Fortran fast-math")
        print(f"Numba / Fortran -O3       {numba_time / strict_time:9.3f}x")
        print(f"Numba / Fortran floor     {numba_time / fast_time:9.3f}x")
        if not args.skip_end_to_end:
            benchmark_end_to_end(fast_library)


if __name__ == "__main__":
    main()
