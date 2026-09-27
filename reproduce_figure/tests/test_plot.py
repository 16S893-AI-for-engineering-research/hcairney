from pathlib import Path

import numpy as np

from plot_figure1 import CONTOUR_LEVELS, generate_figure


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
