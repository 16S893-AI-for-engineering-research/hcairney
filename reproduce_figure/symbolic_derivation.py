"""Exact SymPy check of the derivation of Pope (1975), equations (4.3)--(4.4)."""

from __future__ import annotations

import sympy as sp


def derive() -> dict[str, object]:
    """Construct, project, solve, and exactly verify equation (4.2)."""
    p, q, r = sp.symbols("p q r", real=True)
    g, b1, b2, b3 = sp.symbols("g b1 b2 b3")
    G0, G1, G2 = sp.symbols("G0 G1 G2")
    S2, W2 = sp.symbols("S2 W2")

    identity_3d = sp.eye(3)
    identity_2d = sp.diag(1, 1, 0)
    strain = sp.Matrix([[p, q, 0], [q, -p, 0], [0, 0, 0]])
    rotation = sp.Matrix([[0, r, 0], [-r, 0, 0], [0, 0, 0]])
    basis = (
        identity_3d / 3 - identity_2d / 2,
        strain,
        strain * rotation - rotation * strain,
    )
    anisotropy = G0 * basis[0] + G1 * basis[1] + G2 * basis[2] # effective viscosity hypothesis
    strain_coupling = (
        anisotropy * strain
        + strain * anisotropy
        - sp.Rational(2, 3) * identity_3d * sp.trace(anisotropy * strain)
    )
    rotation_coupling = anisotropy * rotation - rotation * anisotropy
    residual = sp.simplify(
        anisotropy
        + g * (b1 * strain + b2 * strain_coupling - b3 * rotation_coupling)
    ) # plug in the effective viscosity hypothesis to Eq. (4.2)

    projected = tuple(
        sp.factor(sp.trace(residual * tensor) / sp.trace(tensor * tensor))
        for tensor in basis
    )
    invariant_equations = (
        G0 - 2 * b2 * g * S2 * G1,
        G1 + g * (b1 - sp.Rational(1, 3) * b2 * G0 - 2 * b3 * W2 * G2),
        G2 - b3 * g * G1,
    )
    solution_invariants = sp.solve(
        invariant_equations, (G0, G1, G2), dict=True, simplify=True
    )[0] # solve for the basis coefficients
    concrete_invariants = {S2: sp.trace(strain * strain), W2: sp.trace(rotation * rotation)}
    solution = {
        coefficient: sp.factor(value.subs(concrete_invariants))
        for coefficient, value in solution_invariants.items()
    }

    # Check that explicit matrix projection agrees with the invariant equations.
    expected_projected = tuple(
        equation.subs(concrete_invariants) for equation in invariant_equations
    )
    assert all(
        sp.simplify(actual - expected) == 0
        for actual, expected in zip(projected, expected_projected)
    )

    return {
        "G": (G0, G1, G2),
        "parameters": (g, b1, b2, b3),
        "invariants": (S2, W2),
        "basis": basis,
        "projected": projected,
        "invariant_equations": invariant_equations,
        "solution_invariants": solution_invariants,
        "solution": solution,
        "residual": residual,
    }


if __name__ == "__main__":
    result = derive()
    print("Projected coefficient equations:")
    for equation in result["invariant_equations"]:
        print("  0 =", equation)
    print("\nSolution:")
    for coefficient, value in result["solution_invariants"].items():
        print(f"  {coefficient} = {sp.factor(value)}")
