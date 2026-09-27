import numpy as np

from pope1975 import basis_tensors, strain_rotation


def test_strain_and_rotation_have_required_symmetries():
    strain, rotation = strain_rotation(p=0.7, q=-0.2, r=0.4)

    np.testing.assert_array_equal(strain.T, strain) # Check symmetry of strain rate tensor
    np.testing.assert_array_equal(rotation.T, -rotation) # Check antisymmetry of rotation rate tensor
    assert np.trace(strain) == 0.0 # Check tracelessness of strain rate tensor


def test_basis_tensors_are_symmetric_traceless_and_orthogonal():
    basis = basis_tensors(p=0.7, q=-0.2, r=0.4)

    for tensor in basis:
        np.testing.assert_allclose(tensor.T, tensor, rtol=0.0, atol=1e-15) # Check symmetry of all basis tensors
        assert abs(np.trace(tensor)) < 1e-15 # Check tracelessness of all basis tensors

    for i, left in enumerate(basis):
        for right in basis[i + 1 :]:
            assert abs(np.trace(left @ right)) < 1e-15 # Check orthogonality of all basis tensors


def test_tensor_invariants_match_analytical_forms():
    p, q, r = 0.7, -0.2, 0.4
    strain, rotation = strain_rotation(p, q, r)

    assert np.trace(strain @ strain) == 2.0 * (p * p + q * q)
    assert np.trace(rotation @ rotation) == -2.0 * r * r
