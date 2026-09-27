"""Reproduce Figure 1 from Pope (1975)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from pope1975 import figure_1_solution

CONTOUR_LEVELS = np.array([0.03, 0.06, 0.09, 0.12, 0.18, 0.24, 0.30])


def generate_figure(
    output: str | Path = "figure1.png",
    points: int = 401,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Evaluate the model and save a monochrome reproduction of Figure 1."""
    if points < 3:
        raise ValueError("points must be at least three")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sigma = np.linspace(0.0, 2.5, points)
    omega = np.linspace(0.0, 2.5, points)
    c_mu = np.empty((points, points))
    for row, omega_value in enumerate(omega):
        for column, sigma_value in enumerate(sigma):
            c_mu[row, column] = figure_1_solution(sigma_value, omega_value)[0]

    with plt.rc_context(
        {
            "font.family": "serif",
            "font.size": 12,
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 1.1,
        }
    ):
        figure, axis = plt.subplots(figsize=(5.2, 5.2), constrained_layout=True)
        contours = axis.contour(
            sigma,
            omega,
            c_mu,
            levels=CONTOUR_LEVELS,
            colors="black",
            linewidths=1.25,
        )
        axis.clabel(contours, inline=True, fontsize=10, fmt=lambda value: f"{value:g}")
        axis.set(
            xlim=(0.0, 2.5),
            ylim=(0.0, 2.5),
            xlabel=r"$\sigma$",
            ylabel=r"$\Omega$",
            xticks=[0, 1, 2],
            yticks=[0, 1, 2],
        )
        axis.set_aspect("equal")
        axis.tick_params(direction="in", top=True, right=True, length=5)
        figure.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(figure)
    return sigma, omega, c_mu


if __name__ == "__main__":
    generate_figure(Path(__file__).with_name("figure1.png"))
