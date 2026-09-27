# Pope (1975) reproduction

This directory independently re-derives equations (4.3)–(4.4) of Pope (1975), verifies the algebra symbolically and numerically, and reproduces Figure 1.

## Reproduce

`uv` manages the Python environment and lockfile:

```bash
python -m uv sync
python -m uv run pytest
python -m uv run python symbolic_derivation.py
python -m uv run python plot_figure1.py
latexmk -pdf -interaction=nonstopmode -halt-on-error derivation.tex
```

Outputs:

- `figure1.png`: reproduced contour plot
- `derivation.pdf`: compiled verified derivation
- `derivation.tex`: LaTeX source

## Verification design

Development used focused red–green–refactor cycles for:

1. tensor definitions and analytical invariants;
2. a direct linear solve of implicit equation (4.2);
3. the explicit equations (4.3)–(4.4) and exact SymPy substitution;
4. the production identity;
5. the coupled scalar solver used for Figure 1;
6. final plot generation.

The direct equation-(4.2) solver is independent of the explicit implementation and is used as its numerical reference. Passing tests verifies the algebra and implementation; it does not validate the turbulence closure as a physical model.
