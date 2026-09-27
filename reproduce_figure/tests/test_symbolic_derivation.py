import sympy as sp

from symbolic_derivation import derive


def test_symbolic_projection_produces_three_coefficient_equations():
    result = derive()
    G0, G1, G2 = result["G"]
    g, b1, b2, b3 = result["parameters"]
    S2, W2 = result["invariants"]
    expected = (
        G0 - 2 * b2 * g * S2 * G1,
        G1 + g * (b1 - sp.Rational(1, 3) * b2 * G0 - 2 * b3 * W2 * G2),
        G2 - b3 * g * G1,
    )

    for actual, target in zip(result["invariant_equations"], expected):
        assert sp.simplify(actual - target) == 0


def test_symbolic_solution_exactly_satisfies_equation_4_2():
    result = derive()
    checked = result["residual"].subs(result["solution"]).applyfunc(
        lambda entry: sp.factor(sp.cancel(entry))
    )

    assert checked == sp.zeros(3)
