"""Tensor algebra and numerical model from Pope (1975), section 4."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

Array = NDArray[np.float64]


def strain_rotation(p: float, q: float, r: float) -> tuple[Array, Array]:
    """Return general planar symmetric-traceless strain and antisymmetric rotation.

    The tensors are embedded in three dimensions, with the third direction the
    homogeneous direction. All quantities are nondimensional as in Pope (1975).
    """
    strain = np.array([[p, q, 0.0], [q, -p, 0.0], [0.0, 0.0, 0.0]])
    rotation = np.array([[0.0, r, 0.0], [-r, 0.0, 0.0], [0.0, 0.0, 0.0]])
    return strain, rotation


def basis_tensors(p: float, q: float, r: float) -> tuple[Array, Array, Array]:
    """Return Pope's three independent symmetric-traceless planar tensors."""
    strain, rotation = strain_rotation(p, q, r)
    identity_3d = np.eye(3)
    identity_2d = np.diag([1.0, 1.0, 0.0])
    t0 = identity_3d / 3.0 - identity_2d / 2.0
    t1 = strain
    t2 = strain @ rotation - rotation @ strain
    return t0, t1, t2


def model_coefficients(c2: float = 0.4) -> tuple[float, float, float]:
    """Return b1, b2, and b3 from Pope's equation (4.2)."""
    return 8.0 / 15.0, (5.0 - 9.0 * c2) / 11.0, (7.0 * c2 + 1.0) / 11.0


def equation_4_2_residual(
    anisotropy: Array,
    strain: Array,
    rotation: Array,
    g: float,
    c2: float = 0.4,
) -> Array:
    """Return left minus right sides of Pope's implicit equation (4.2)."""
    b1, b2, b3 = model_coefficients(c2)
    identity = np.eye(3)
    strain_coupling = (
        anisotropy @ strain
        + strain @ anisotropy
        - (2.0 / 3.0) * identity * np.trace(anisotropy @ strain)
    )
    rotation_coupling = anisotropy @ rotation - rotation @ anisotropy
    return anisotropy + g * (
        b1 * strain + b2 * strain_coupling - b3 * rotation_coupling
    )


def solve_implicit_anisotropy(
    strain: Array,
    rotation: Array,
    g: float,
    c2: float = 0.4,
) -> Array:
    """Solve equation (4.2) directly as a three-unknown linear system.

    This reference solver does not use the explicit equations (4.3)--(4.4).
    It assumes planar flow and represents a as a symmetric, traceless tensor.
    """
    component_basis = (
        np.diag([1.0, 0.0, -1.0]),
        np.diag([0.0, 1.0, -1.0]),
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
    ) # Component basis for the symmetric, traceless tensor space
    zero = np.zeros((3, 3))
    constant = equation_4_2_residual(zero, strain, rotation, g, c2)
    selected = ((0, 0), (1, 1), (0, 1))
    matrix = np.array(
        [
            [
                (equation_4_2_residual(basis, strain, rotation, g, c2) - constant)[i, j]
                for basis in component_basis
            ]
            for i, j in selected
        ]
    )
    right_hand_side = -np.array([constant[i, j] for i, j in selected])
    coefficients = np.linalg.solve(matrix, right_hand_side)
    return sum(
        (coefficient * basis for coefficient, basis in zip(coefficients, component_basis)),
        start=np.zeros((3, 3)),
    )


def production_over_dissipation(anisotropy: Array, strain: Array) -> float:
    """Return P/epsilon = -tr(a s) for incompressible planar flow."""
    return -float(np.trace(anisotropy @ strain))


def explicit_c_mu(
    strain: Array,
    rotation: Array,
    g: float,
    c2: float = 0.4,
) -> float:
    """Evaluate equation (4.4) for specified strain, rotation, and g."""
    b1, b2, b3 = model_coefficients(c2)
    strain_invariant = float(np.trace(strain @ strain))
    rotation_invariant = float(np.trace(rotation @ rotation))
    denominator = (
        1.0
        - 2.0 * rotation_invariant * b3**2 * g**2
        - (2.0 / 3.0) * b2**2 * g**2 * strain_invariant
    )
    scale = 1.0 + abs(2.0 * rotation_invariant * b3**2 * g**2) + abs(
        (2.0 / 3.0) * b2**2 * g**2 * strain_invariant
    )
    if abs(denominator) <= 32.0 * np.finfo(float).eps * scale:
        raise ValueError("equation (4.4) has a singular denominator")
    return 0.5 * b1 * g / denominator


def figure_1_solution(
    sigma: float,
    omega: float,
    c1: float = 1.5,
    c2: float = 0.4,
) -> tuple[float, float]:
    """Return ``(C_mu, g)`` at a point in Pope's Figure 1.

    The physical root is positive and continuous from zero strain. Here
    ``sigma = sqrt(tr(s^2)/2)`` and ``omega = sqrt(-tr(w^2)/2)``.
    """
    if sigma < 0.0 or omega < 0.0:
        raise ValueError("sigma and omega must be nonnegative")
    offset = c1 - 1.0
    if offset <= 0.0:
        raise ValueError("c1 must be greater than one")
    b1, b2, b3 = model_coefficients(c2)

    def c_mu_for_g(g_value: float) -> float:
        denominator = (
            1.0
            + 4.0 * b3**2 * g_value**2 * omega**2
            - (4.0 / 3.0) * b2**2 * g_value**2 * sigma**2
        )
        return 0.5 * b1 * g_value / denominator

    if sigma == 0.0:
        g_value = 1.0 / offset
        return c_mu_for_g(g_value), g_value

    coefficient = 4.0 * b3**2 * omega**2 - (4.0 / 3.0) * b2**2 * sigma**2

    def eliminated_equation(g_value: float) -> float:
        denominator = 1.0 + coefficient * g_value**2
        return (1.0 - offset * g_value) * denominator - 2.0 * b1 * sigma**2 * g_value**2

    g_value = brentq(
        eliminated_equation,
        0.0,
        1.0 / offset,
        xtol=5e-15,
        rtol=1e-14,
    )
    return c_mu_for_g(g_value), g_value


def explicit_anisotropy(
    strain: Array,
    rotation: Array,
    g: float,
    c2: float = 0.4,
) -> Array:
    """Evaluate the explicit effective-viscosity hypothesis, equation (4.3)."""
    _, b2, b3 = model_coefficients(c2)
    c_mu = explicit_c_mu(strain, rotation, g, c2)
    identity_difference = (2.0 / 3.0) * np.eye(3) - np.diag([1.0, 1.0, 0.0])
    strain_invariant = float(np.trace(strain @ strain))
    return -2.0 * c_mu * (
        strain
        + g * b3 * (strain @ rotation - rotation @ strain)
        + g * b2 * strain_invariant * identity_difference
    )
