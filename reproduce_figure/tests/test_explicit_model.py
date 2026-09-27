import numpy as np
import pytest

from pope1975 import (
    explicit_anisotropy,
    explicit_c_mu,
    solve_implicit_anisotropy,
    strain_rotation,
)


@pytest.mark.parametrize(
    ("p", "q", "r", "g"),
    [
        (0.22, -0.13, 0.31, 0.8),
        (-0.41, 0.09, -0.25, 1.1),
        (0.0, 0.35, 0.0, 0.6),
        (0.16, 0.0, 0.47, 1.3),
    ],
)
def test_explicit_equation_matches_independent_direct_solve(p, q, r, g):
    strain, rotation = strain_rotation(p, q, r)
    expected = solve_implicit_anisotropy(strain, rotation, g)
    actual = explicit_anisotropy(strain, rotation, g)

    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=3e-15)


def test_c_mu_at_zero_invariants_has_analytical_limit():
    # Equation (4.4) reduces exactly to b1*g/2 when both invariants vanish.
    g = 1.7
    assert explicit_c_mu(np.zeros((3, 3)), np.zeros((3, 3)), g) == (8 / 15) * g / 2


def test_explicit_c_mu_rejects_singular_denominator():
    strain, rotation = strain_rotation(1.0, 0.0, 0.0)
    b2 = (5.0 - 9.0 * 0.4) / 11.0
    singular_g = np.sqrt(3.0 / (4.0 * b2**2))

    with pytest.raises(ValueError, match="singular"):
        explicit_c_mu(strain, rotation, singular_g)
