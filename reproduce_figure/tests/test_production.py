import numpy as np

from pope1975 import (
    explicit_anisotropy,
    explicit_c_mu,
    production_over_dissipation,
    strain_rotation,
)


def test_contraction_gives_production_identity():
    strain, rotation = strain_rotation(0.27, -0.19, 0.33)
    g = 0.9
    anisotropy = explicit_anisotropy(strain, rotation, g)
    c_mu = explicit_c_mu(strain, rotation, g)
    sigma_squared = 0.5 * np.trace(strain @ strain)

    actual = production_over_dissipation(anisotropy, strain)
    expected = 4.0 * c_mu * sigma_squared
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=1e-16)


def test_rotation_and_out_of_plane_terms_do_not_contribute_to_production():
    strain, rotation = strain_rotation(0.27, -0.19, 0.33)
    commutator = strain @ rotation - rotation @ strain
    out_of_plane = (2.0 / 3.0) * np.eye(3) - np.diag([1.0, 1.0, 0.0])

    assert abs(np.trace(commutator @ strain)) < 1e-15
    assert abs(np.trace(out_of_plane @ strain)) < 1e-15
