from pathlib import Path

import numpy as np

from plot_figure1 import CONTOUR_LEVELS, generate_figure
from plot_figure1_overlay import generate_overlay


def test_plot_generation_creates_nonempty_png_and_expected_grid(tmp_path: Path):
    output = tmp_path / "figure1.png"
    sigma, omega, values = generate_figure(output, points=31)

    assert output.is_file()
    assert output.stat().st_size > 10_000
    assert sigma.shape == (31,)
    assert omega.shape == (31,)
    assert values.shape == (31, 31)
    assert np.isfinite(values).all()
    assert tuple(CONTOUR_LEVELS) == (0.03, 0.06, 0.09, 0.12, 0.18, 0.24, 0.30)


def test_overlay_generation_creates_nonempty_png(tmp_path: Path):
    output = tmp_path / "figure1_overlay.png"
    reference = Path(__file__).parents[1] / "pope_1975.pdf"

    generate_overlay(reference, output, points=31, render_dpi=150)

    assert output.is_file()
    assert output.stat().st_size > 10_000
