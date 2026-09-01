r"""Derive Devoto's collision-bracket matrices with exact symbolic algebra.

This module is a development-time oracle for the explicit expressions in the
appendix of Devoto (1966).  It starts before equations A3--A22: from the
Sonine-polynomial trial functions, elastic two-body collision kinematics, and
Maxwellian averages.  SymPy then performs the centre-of-mass Gaussian moments
and collects the remaining relative-speed and scattering-angle polynomial in
Devoto's averaged transport cross sections.

The production implementation deliberately does not import this module.  Its
purpose is to independently check the fast, handwritten Numba kernels in
``minplascalc.functions_transport``.

For species masses ``m_i`` and ``m_l``, introduce dimensionless centre-of-mass
and relative velocities ``Y`` and ``gamma``.  With

``A**2 = m_i / (m_i + m_l)`` and ``B**2 = m_l / (m_i + m_l)``, the
pre-collision peculiar velocities are

``W_i = A Y + B gamma n`` and ``W_l = B Y - A gamma n``.

An elastic collision rotates ``n`` to ``n_prime``.  The vector trial functions
for diffusion and heat conduction are

``W L_m**(3/2)(W**2)``,

and the traceless-tensor trial functions for viscosity are

``(W W - W**2 I / 3) L_m**(5/2)(W**2)``.

No coefficients from Devoto's appendix appear below.  The only convention
specific to that paper is the normalization relating a raw radial/angular
moment to its barred ``Q^(l,s)`` in equation 11.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from math import sqrt
from types import SimpleNamespace
from typing import Literal

import numpy as np
import sympy as sp

Rank = Literal[1, 2]
Channel = Literal["self", "cross"]

_YX, _YY, _YZ = sp.symbols("Y_x Y_y Y_z", real=True)
_GAMMA, _COS_CHI, _SIN_CHI = sp.symbols("gamma cos_chi sin_chi", real=True)
_A, _B = sp.symbols("A B", positive=True, real=True)

_Y = sp.Matrix([_YX, _YY, _YZ])
_N = sp.Matrix([0, 0, 1])
_N_PRIME = sp.Matrix([_SIN_CHI, 0, _COS_CHI])
_IDENTITY = sp.eye(3)


def _gaussian_moment(power: int) -> sp.Rational:
    r"""Return ``E[X**power]`` for density ``exp(-X**2) / sqrt(pi)``."""
    if power % 2:
        return sp.S.Zero
    if power == 0:
        return sp.S.One
    return sp.factorial2(power - 1) / sp.Integer(2) ** (power // 2)


def _maxwellian_com_average(expression: sp.Expr) -> sp.Expr:
    """Integrate a polynomial over the dimensionless COM Maxwellian."""
    polynomial = sp.Poly(sp.expand(expression), _YX, _YY, _YZ)
    result = sp.S.Zero
    for powers, coefficient in polynomial.terms():
        result += coefficient * sp.prod(
            _gaussian_moment(power) for power in powers
        )

    # Rotational invariance makes the result even in sin(chi).  Replacing its
    # even powers by (1 - cos(chi)**2) completes the azimuth-independent
    # scattering-angle reduction.
    sin_polynomial = sp.Poly(sp.expand(result), _SIN_CHI)
    reduced = sp.S.Zero
    for (power,), coefficient in sin_polynomial.terms():
        if power % 2:
            raise ValueError("Unexpected odd scattering-azimuth contribution")
        reduced += coefficient * (1 - _COS_CHI**2) ** (power // 2)
    return sp.factor(sp.expand(reduced))


def _trial_function(velocity: sp.Matrix, rank: Rank, order: int):
    """Return a vector or traceless-tensor Sonine trial function."""
    speed_squared = sp.expand(velocity.dot(velocity))
    if rank == 1:
        basis = velocity
        alpha = sp.Rational(3, 2)
    elif rank == 2:
        basis = velocity * velocity.T - speed_squared * _IDENTITY / sp.Integer(
            3
        )
        alpha = sp.Rational(5, 2)
    else:  # pragma: no cover - protected by the public type and tests
        raise ValueError(f"Unsupported tensor rank: {rank}")
    return basis * sp.assoc_laguerre(order, alpha, speed_squared)


def _collision_change(
    com_coefficient: sp.Expr,
    relative_coefficient: sp.Expr,
    rank: Rank,
    order: int,
):
    """Return post-collision minus pre-collision trial functions."""
    before = com_coefficient * _Y + relative_coefficient * _GAMMA * _N
    after = com_coefficient * _Y + relative_coefficient * _GAMMA * _N_PRIME
    return _trial_function(after, rank, order) - _trial_function(
        before, rank, order
    )


def _inner_product(left, right, rank: Rank) -> sp.Expr:
    """Contract vector or tensor collision changes."""
    if rank == 1:
        return left.dot(right)
    return sum(
        left[row, column] * right[row, column]
        for row in range(3)
        for column in range(3)
    )


def _q_normalization(l_value: int, s_value: int) -> sp.Rational:
    r"""Prefactor mapping a raw moment to Devoto's barred Q, equation 11."""
    return sp.Rational(
        4 * (l_value + 1),
        sp.factorial(s_value + 1) * (2 * l_value + 1 - (-1) ** l_value),
    )


def _collect_transport_moments(
    averaged_bracket: sp.Expr,
) -> dict[tuple[int, int], sp.Expr]:
    r"""Collect a bracket as coefficients of barred ``Q^(l,s)`` moments."""
    radial_polynomial = sp.Poly(sp.expand(averaged_bracket), _GAMMA)
    by_radial_power: dict[int, sp.Expr] = {}
    for (power,), coefficient in radial_polynomial.terms():
        if power % 2:
            raise ValueError("Unexpected odd relative-speed contribution")
        by_radial_power[power // 2] = coefficient

    result: dict[tuple[int, int], sp.Expr] = {}
    for s_value, angular_expression in by_radial_power.items():
        angular_polynomial = sp.Poly(sp.expand(angular_expression), _COS_CHI)
        raw_coefficients: dict[int, sp.Expr] = {}
        for l_value in range(1, angular_polynomial.degree() + 1):
            coefficient = angular_polynomial.coeff_monomial(_COS_CHI**l_value)
            if coefficient != 0:
                # P(cos chi) = sum_l a_l (1 - cos(chi)**l).
                raw_coefficients[l_value] = -coefficient

        if (
            sp.simplify(
                angular_polynomial.coeff_monomial(1)
                - sum(raw_coefficients.values(), sp.S.Zero)
            )
            != 0
        ):
            raise ValueError("Collision bracket does not vanish at chi = 0")

        for l_value, raw_coefficient in raw_coefficients.items():
            result[l_value, s_value] = sp.factor(
                raw_coefficient / _q_normalization(l_value, s_value)
            )
    return result


@lru_cache(maxsize=None)
def derive_pair_moments(
    rank: Rank,
    left_order: int,
    right_order: int,
    channel: Channel,
) -> dict[tuple[int, int], sp.Expr]:
    """Derive exact collision-moment coefficients for one ordered pair.

    ``self`` contracts the change of the two basis functions belonging to
    species ``i``.  ``cross`` contracts the species-``i`` change with the
    species-``l`` change.  Together these are the diagonal and off-diagonal
    binary-collision contributions to Devoto's bracket matrix.
    """
    left_change = _collision_change(_A, _B, rank, left_order)
    if channel == "self":
        right_change = _collision_change(_A, _B, rank, right_order)
    elif channel == "cross":
        right_change = _collision_change(_B, -_A, rank, right_order)
    else:  # pragma: no cover - protected by the public type and tests
        raise ValueError(f"Unsupported collision channel: {channel}")

    contracted = _inner_product(left_change, right_change, rank)
    averaged = _maxwellian_com_average(contracted)
    return _collect_transport_moments(averaged)


@lru_cache(maxsize=None)
def _moment_evaluator(
    rank: Rank,
    left_order: int,
    right_order: int,
    channel: Channel,
):
    moments = derive_pair_moments(rank, left_order, right_order, channel)
    keys = tuple(sorted(moments))
    function = sp.lambdify((_A, _B), [moments[key] for key in keys], "numpy")
    return keys, function


def _evaluate_pair_bracket(
    rank: Rank,
    left_order: int,
    right_order: int,
    channel: Channel,
    mass_fraction_i: float,
    mass_fraction_l: float,
    collision_integrals: dict[tuple[int, int], float],
) -> float:
    keys, function = _moment_evaluator(rank, left_order, right_order, channel)
    coefficients = np.atleast_1d(
        function(sqrt(mass_fraction_i), sqrt(mass_fraction_l))
    ).astype(float)
    return float(
        sum(
            coefficient * collision_integrals[key]
            for key, coefficient in zip(keys, coefficients)
        )
    )


def assemble_upper_bracket_matrix(
    masses: np.ndarray,
    number_densities: np.ndarray,
    collision_integrals: dict[tuple[int, int], np.ndarray],
    *,
    rank: Rank,
    maximum_order: int,
) -> np.ndarray:
    """Assemble upper Sonine blocks from the symbolic collision brackets.

    For the vector ``(0, 0)`` block, :func:`assemble_bracket_block` also
    applies the diffusion subsidiary constraint that turns the raw bracket
    into Devoto's equation A3.  The remaining blocks are raw bracket results.
    """
    masses = np.asarray(masses, dtype=float)
    number_densities = np.asarray(number_densities, dtype=float)
    species_count = len(masses)
    result = np.full(
        (
            (maximum_order + 1) * species_count,
            (maximum_order + 1) * species_count,
        ),
        np.nan,
    )

    for left_order in range(maximum_order + 1):
        for right_order in range(left_order, maximum_order + 1):
            block = assemble_bracket_block(
                masses,
                number_densities,
                collision_integrals,
                rank=rank,
                left_order=left_order,
                right_order=right_order,
            )

            row = slice(
                left_order * species_count,
                (left_order + 1) * species_count,
            )
            column = slice(
                right_order * species_count,
                (right_order + 1) * species_count,
            )
            result[row, column] = block
    return result


def assemble_bracket_block(
    masses: np.ndarray,
    number_densities: np.ndarray,
    collision_integrals: dict[tuple[int, int], np.ndarray],
    *,
    rank: Rank,
    left_order: int,
    right_order: int,
) -> np.ndarray:
    """Assemble one exact-symbolic Devoto bracket block numerically.

    The vector ``(0, 0)`` coefficients obey the mass-flux subsidiary
    condition ``sum_i n_i sqrt(m_i) c_i0 = 0``.  If ``R`` is the raw bracket
    and ``w_i = n_i sqrt(m_i)``, adding any outer product ``a w.T`` therefore
    leaves its action on an admissible coefficient vector unchanged.  Devoto
    chooses ``a_i = -R_ii / w_i``, giving equation A3:

    ``q00 = R - outer(diag(R) / w, w)``.
    """
    masses = np.asarray(masses, dtype=float)
    number_densities = np.asarray(number_densities, dtype=float)
    species_count = len(masses)
    block = np.zeros((species_count, species_count))
    for i in range(species_count):
        for l_index in range(species_count):
            total_mass = masses[i] + masses[l_index]
            x_value = masses[i] / total_mass
            y_value = masses[l_index] / total_mass
            pair_q = {
                key: values[i, l_index]
                for key, values in collision_integrals.items()
            }
            prefactor = (
                4
                * number_densities[i]
                * number_densities[l_index]
                / sqrt(y_value)
            )
            self_bracket = _evaluate_pair_bracket(
                rank,
                left_order,
                right_order,
                "self",
                x_value,
                y_value,
                pair_q,
            )
            if i == l_index:
                cross_bracket = _evaluate_pair_bracket(
                    rank,
                    left_order,
                    right_order,
                    "cross",
                    x_value,
                    y_value,
                    pair_q,
                )
                block[i, i] += prefactor * (self_bracket + cross_bracket)
            else:
                block[i, i] += prefactor * self_bracket
                block[i, l_index] += prefactor * _evaluate_pair_bracket(
                    rank,
                    left_order,
                    right_order,
                    "cross",
                    x_value,
                    y_value,
                    pair_q,
                )
    if rank == 1 and left_order == right_order == 0:
        constraint_weights = number_densities * np.sqrt(masses)
        block -= np.outer(
            np.diag(block) / constraint_weights,
            constraint_weights,
        )
        # The subtraction above is exact algebraically.  Preserve Devoto's
        # exact zero-diagonal convention rather than retaining roundoff from
        # subtracting two equal floating-point values.
        np.fill_diagonal(block, 0.0)
    return block


def format_pair_moments(
    rank: Rank, left_order: int, right_order: int, channel: Channel
) -> str:
    """Return a stable, human-readable exact derivation result."""
    moments = derive_pair_moments(rank, left_order, right_order, channel)
    return " + ".join(
        f"({sp.sstr(coefficient)}) Q^({l_value},{s_value})"
        for (l_value, s_value), coefficient in sorted(moments.items())
    )


def check_production_kernels() -> dict[str, float]:
    """Derive every supported block and compare with production kernels."""
    from minplascalc import functions_transport as ft

    rng = np.random.default_rng(1966)
    species_count = 3
    masses = np.array([1.7, 2.9, 5.3])
    number_densities = np.array([2.3, 3.1, 4.7])
    collision_integrals = {}
    for moment in ft.LS_PAIRS:
        values = rng.uniform(0.5, 2.0, (species_count, species_count))
        collision_integrals[moment] = (values + values.T) / 2

    state = SimpleNamespace(
        masses=masses,
        number_densities=number_densities,
    )

    class ArtificialMixture:
        species = tuple(range(species_count))

        def __init__(self):
            self.masses = masses

        @staticmethod
        def calculate_composition():
            return number_densities

        @staticmethod
        def _equilibrium_state():
            return state

    mixture = ArtificialMixture()
    comparisons = (
        (
            "q",
            ft.q(mixture, collision_integrals),
            assemble_upper_bracket_matrix(
                masses,
                number_densities,
                collision_integrals,
                rank=1,
                maximum_order=3,
            ),
            3,
        ),
        (
            "qhat",
            ft.qhat(mixture, collision_integrals),
            assemble_upper_bracket_matrix(
                masses,
                number_densities,
                collision_integrals,
                rank=2,
                maximum_order=1,
            ),
            1,
        ),
    )

    errors = {}
    for name, actual, derived, maximum_order in comparisons:
        for left_order in range(maximum_order + 1):
            for right_order in range(left_order, maximum_order + 1):
                row = slice(
                    left_order * species_count,
                    (left_order + 1) * species_count,
                )
                column = slice(
                    right_order * species_count,
                    (right_order + 1) * species_count,
                )
                actual_block = actual[row, column]
                derived_block = derived[row, column]
                block_scale = max(
                    float(np.max(np.abs(actual_block))),
                    float(np.max(np.abs(derived_block))),
                    float(np.finfo(float).tiny),
                )
                scaled_error = float(
                    np.max(np.abs(actual_block - derived_block)) / block_scale
                )
                label = f"{name}[{left_order},{right_order}]"
                errors[label] = scaled_error
                if not np.allclose(
                    actual_block,
                    derived_block,
                    rtol=2e-12,
                    atol=1e-12,
                ):
                    raise AssertionError(
                        f"{label} differs by block-scaled error {scaled_error}"
                    )
    return errors


def _print_derivations() -> None:
    for tensor_rank, highest_order in ((1, 3), (2, 1)):
        print(f"rank {tensor_rank}")
        for m_order in range(highest_order + 1):
            for p_order in range(m_order, highest_order + 1):
                print(
                    f"  ({m_order}, {p_order}) self : "
                    f"{format_pair_moments(tensor_rank, m_order, p_order, 'self')}"  # noqa: E501
                )
                print(
                    f"  ({m_order}, {p_order}) cross: "
                    f"{format_pair_moments(tensor_rank, m_order, p_order, 'cross')}"  # noqa: E501
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare every derived block with the production kernels",
    )
    arguments = parser.parse_args()
    if arguments.check:
        for block, error in check_production_kernels().items():
            print(f"{block}: block-scaled error {error:.3e}")
    else:
        _print_derivations()
