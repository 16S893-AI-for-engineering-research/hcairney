import numpy as np

from pope1975 import equation_4_2_residual, solve_implicit_anisotropy, strain_rotation


def test_zero_strain_has_zero_anisotropy():
    strain, rotation = strain_rotation(0.0, 0.0, 0.6)
    anisotropy = solve_implicit_anisotropy(strain, rotation, g=0.8)
    np.testing.assert_array_equal(anisotropy, np.zeros((3, 3)))


def test_direct_solution_satisfies_equation_4_2():
    strain, rotation = strain_rotation(0.22, -0.13, 0.31)
    anisotropy = solve_implicit_anisotropy(strain, rotation, g=0.8)
    residual = equation_4_2_residual(anisotropy, strain, rotation, g=0.8)

    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=2e-15) # check the residual is zero
    np.testing.assert_allclose(anisotropy, anisotropy.T, rtol=0.0, atol=1e-15) # check solution is symmetric
    assert abs(np.trace(anisotropy)) < 1e-15 # check solution is traceless


def test_direct_solution_is_invariant_under_rotation_of_coordinates():
    strain, rotation = strain_rotation(0.22, -0.13, 0.31)
    anisotropy = solve_implicit_anisotropy(strain, rotation, g=0.8)
    angle = 0.37
    transform = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated_solution = solve_implicit_anisotropy(
        transform @ strain @ transform.T,
        transform @ rotation @ transform.T,
        g=0.8,
    )

    np.testing.assert_allclose(
        rotated_solution,
        transform @ anisotropy @ transform.T,
        rtol=2e-14,
        atol=2e-15,
    )
