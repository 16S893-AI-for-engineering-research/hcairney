import numpy as np
import pytest

from pope1975 import figure_1_solution, model_coefficients


def test_origin_has_exact_limiting_solution():
    c_mu, g = figure_1_solution(0.0, 0.0)

    assert g == 2.0
    assert c_mu == 8.0 / 15.0


@pytest.mark.parametrize(
    ("sigma", "omega"),
    [(0.25, 0.4), (0.8, 1.2), (1.7, 0.3), (2.5, 2.5)],
)
def test_figure_solution_satisfies_both_coupled_equations(sigma, omega):
    c_mu, g = figure_1_solution(sigma, omega)
    b1, b2, b3 = model_coefficients()
    denominator = (
        1.0
        + 4.0 * b3**2 * g**2 * omega**2
        - (4.0 / 3.0) * b2**2 * g**2 * sigma**2
    )

    np.testing.assert_allclose(c_mu, 0.5 * b1 * g / denominator, rtol=2e-13)
    np.testing.assert_allclose(1.0 / g, 0.5 + 4.0 * c_mu * sigma**2, rtol=2e-13)
    assert c_mu > 0.0
    assert g > 0.0


def test_rotation_reduces_c_mu_at_fixed_strain():
    values = [figure_1_solution(0.8, omega)[0] for omega in (0.0, 0.5, 1.0, 2.0)]
    assert all(left > right for left, right in zip(values, values[1:]))


@pytest.mark.parametrize(("sigma", "omega"), [(-0.1, 0.2), (0.1, -0.2)])
def test_figure_solution_rejects_negative_invariants(sigma, omega):
    with pytest.raises(ValueError, match="nonnegative"):
        figure_1_solution(sigma, omega)
