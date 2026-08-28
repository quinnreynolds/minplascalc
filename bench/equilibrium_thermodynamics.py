"""Packed thermodynamic data shared by experimental equilibrium solvers."""

from __future__ import annotations

import numpy as np

from minplascalc import species as species_module
from minplascalc import units as u


class PackedEquilibriumThermodynamics:
    """Mixin providing packed partition and reference-energy evaluation."""

    def _prepare_packed_thermodynamics(self) -> None:
        """Pack immutable species data for vectorized evaluation."""
        count = self.species_count
        species = self.mixture.species
        self.charges = np.asarray(
            self.mixture.charge_numbers, dtype=np.float64
        )
        self.positive_indices = np.flatnonzero(self.charges > 0)
        self.positive_charges = self.charges[self.positive_indices]
        self.electron_index = next(
            (index for index, item in enumerate(species) if item.name == "e"),
            -1,
        )
        self.log_translational_prefactors = 1.5 * np.log(
            2
            * np.pi
            * np.array([item.molar_mass for item in species])
            * u.k_b
            / (u.N_a * u.h**2)
        )

        monatomic = [
            (index, item)
            for index, item in enumerate(species)
            if isinstance(item, species_module.Monatomic)
        ]
        self.monatomic_indices = np.array(
            [index for index, _ in monatomic], dtype=np.int64
        )
        self.monatomic_ionization = np.zeros(count)
        level_owners = []
        level_energies = []
        level_degeneracies = []
        for index, item in monatomic:
            self.monatomic_ionization[index] = item.ionisation_energy
            level_owners.extend([index] * len(item._level_energies))
            level_energies.extend(item._level_energies)
            level_degeneracies.extend(item._degeneracies)
        self.level_owners = np.asarray(level_owners, dtype=np.int64)
        self.level_energies = np.asarray(level_energies, dtype=np.float64)
        self.level_degeneracies = np.asarray(
            level_degeneracies, dtype=np.float64
        )

        diatomic = [
            (index, item)
            for index, item in enumerate(species)
            if isinstance(item, species_module.Diatomic)
        ]
        self.diatomic_indices = np.array(
            [index for index, _ in diatomic], dtype=np.int64
        )
        self.diatomic_g0 = np.array([item.g0 for _, item in diatomic])
        self.diatomic_w = np.array([item.w_e for _, item in diatomic])
        self.diatomic_rotation = np.array(
            [item.sigma_s * item.b_e for _, item in diatomic]
        )
        known = set(self.monatomic_indices) | set(self.diatomic_indices)
        if self.electron_index >= 0:
            known.add(self.electron_index)
        self.fallback_indices = np.array(
            [index for index in range(count) if index not in known],
            dtype=np.int64,
        )

        reference_offsets = np.zeros(count)
        for index, item in enumerate(species):
            if sum(item.stoichiometry.values()) >= 2:
                reference_offsets[index] = -item.dissociation_energy
        lowering_coefficients = np.zeros((count, count))
        neutral_species = [item for item in species if item.charge_number == 0]
        for neutral in neutral_species:
            negative = sorted(
                (
                    (index, item)
                    for index, item in enumerate(species)
                    if item.stoichiometry == neutral.stoichiometry
                    and item.charge_number <= 0
                ),
                key=lambda pair: pair[1].charge_number,
                reverse=True,
            )
            positive = sorted(
                (
                    (index, item)
                    for index, item in enumerate(species)
                    if item.stoichiometry == neutral.stoichiometry
                    and item.charge_number >= 0
                ),
                key=lambda pair: pair[1].charge_number,
            )
            for (source, source_species), (target, _) in zip(
                positive[:-1], positive[1:]
            ):
                reference_offsets[target] = (
                    reference_offsets[source]
                    + source_species.ionisation_energy
                )
                lowering_coefficients[target] = lowering_coefficients[source]
                lowering_coefficients[target, source] -= 1.0
            for (source, _), (target, target_species) in zip(
                negative[:-1], negative[1:]
            ):
                reference_offsets[target] = (
                    reference_offsets[source]
                    - target_species.ionisation_energy
                )
                lowering_coefficients[target] = lowering_coefficients[source]
                lowering_coefficients[target, target] += 1.0
        self._packed_reference_offsets = reference_offsets
        self._packed_reference_lowering_coefficients = lowering_coefficients
        self._packed_cached_temperature = np.nan
        self._packed_partition_cache = {}

    def _prepare_temperature_thermodynamics(self) -> None:
        """Cache factors invariant during a fixed-temperature solve."""
        temperature = float(self.mixture.T)
        if temperature == self._packed_cached_temperature:
            return
        kbt = u.k_b * temperature
        self.level_boltzmann_terms = self.level_degeneracies * np.exp(
            -self.level_energies / kbt
        )
        self.temperature_log_internal = np.full(self.species_count, np.nan)
        if self.diatomic_indices.size:
            vibration_ratio = self.diatomic_w / kbt
            self.temperature_log_internal[self.diatomic_indices] = (
                np.log(self.diatomic_g0)
                - vibration_ratio / 2
                - np.log1p(-np.exp(-vibration_ratio))
                + np.log(kbt / self.diatomic_rotation)
            )
        if self.electron_index >= 0:
            self.temperature_log_internal[self.electron_index] = np.log(2.0)
        self.temperature_log_partition = (
            self.log_translational_prefactors + 1.5 * np.log(temperature)
        )
        self._packed_cached_temperature = temperature
        self._packed_partition_cache = {}

    def _packed_log_partition_per_volume(
        self, lowering: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return packed log partition density and active-level counts."""
        self._prepare_temperature_thermodynamics()
        owners = self.level_owners
        active = self.level_energies < (
            self.monatomic_ionization[owners] - lowering[owners]
        )
        key = (
            active.tobytes(),
            np.asarray(lowering[self.fallback_indices]).tobytes(),
        )
        cached = self._packed_partition_cache.get(key)
        if cached is not None:
            return cached

        log_internal = self.temperature_log_internal.copy()
        active_counts = np.zeros(self.species_count, dtype=np.int64)
        if self.monatomic_indices.size:
            sums = np.bincount(
                owners,
                weights=self.level_boltzmann_terms * active,
                minlength=self.species_count,
            )
            active_counts = np.bincount(
                owners, weights=active, minlength=self.species_count
            ).astype(np.int64)
            log_internal[self.monatomic_indices] = np.log(
                sums[self.monatomic_indices]
            )
        for index in self.fallback_indices:
            log_internal[index] = np.log(
                self.mixture.species[index].internal_partition_function(
                    self.mixture.T, lowering[index]
                )
            )
        result = self.temperature_log_partition + log_internal
        result.setflags(write=False)
        active_counts.setflags(write=False)
        packed = (result, active_counts)
        self._packed_partition_cache[key] = packed
        return packed

    def _packed_reference_from_lowering(
        self,
        lowering: np.ndarray,
        lowering_derivatives: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Return reference energies and optional derivatives."""
        coefficients = self._packed_reference_lowering_coefficients
        reference = self._packed_reference_offsets + coefficients @ lowering
        derivative = (
            None
            if lowering_derivatives is None
            else coefficients @ lowering_derivatives
        )
        return reference, derivative
