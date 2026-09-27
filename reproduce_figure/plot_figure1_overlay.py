"""Overlay the reproduced contours on Pope's published Figure 1."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from plot_figure1 import CONTOUR_LEVELS
from pope1975 import figure_1_solution

# Fractional pixel bounds (left, top, right, bottom) of the plotting box on
# page 337 (PDF page 7). Fractions make the crop independent of render DPI.
REFERENCE_AXES_BOUNDS = (0.3200, 0.0940, 0.6831, 0.3438)
DIAGNOSTIC_LEVEL = 0.50


def _render_reference_page(pdf: Path, output: Path, dpi: int) -> None:
    """Rasterize the page containing Figure 1 with Ghostscript."""
    ghostscript = shutil.which("gs")
    if ghostscript is None:
        raise RuntimeError(
            "Ghostscript ('gs') is required to rasterize pope_1975.pdf"
        )
    subprocess.run(
        [
            ghostscript,
            "-q",
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-dFirstPage=7",
            "-dLastPage=7",
            "-sDEVICE=pnggray",
            f"-r{dpi}",
            f"-sOutputFile={output}",
            str(pdf),
        ],
        check=True,
    )


def _crop_reference_axes(page: NDArray[np.generic]) -> NDArray[np.generic]:
    """Crop the published page raster to the Figure 1 plotting box."""
    height, width = page.shape[:2]
    left, top, right, bottom = REFERENCE_AXES_BOUNDS
    return page[
        round(top * height) : round(bottom * height),
        round(left * width) : round(right * width),
    ]


def _evaluate_model(points: int) -> tuple[NDArray[np.float64], ...]:
    sigma = np.linspace(0.0, 2.5, points)
    omega = np.linspace(0.0, 2.5, points)
    c_mu = np.empty((points, points))
    for row, omega_value in enumerate(omega):
        for column, sigma_value in enumerate(sigma):
            c_mu[row, column] = figure_1_solution(sigma_value, omega_value)[0]
    return sigma, omega, c_mu


def generate_overlay(
    reference_pdf: str | Path = "pope_1975.pdf",
    output: str | Path = "figure1_overlay.png",
    points: int = 401,
    render_dpi: int = 600,
) -> Path:
    """Save Pope's rasterized figure overlaid with computed contours.

    Solid red curves are the levels used by :mod:`plot_figure1`. The dashed
    blue curve is the computed 0.50 level; it exposes that the small contour
    printed as ``0.3`` in the paper has the geometry of the 0.50 contour.
    """
    if points < 3:
        raise ValueError("points must be at least three")
    if render_dpi < 72:
        raise ValueError("render_dpi must be at least 72")

    reference_pdf = Path(reference_pdf).resolve()
    output = Path(output).resolve()
    if not reference_pdf.is_file():
        raise FileNotFoundError(reference_pdf)
    output.parent.mkdir(parents=True, exist_ok=True)

    sigma, omega, c_mu = _evaluate_model(points)
    with tempfile.TemporaryDirectory() as temporary_directory:
        page_path = Path(temporary_directory) / "pope-page-7.png"
        _render_reference_page(reference_pdf, page_path, render_dpi)
        reference_axes = _crop_reference_axes(mpimg.imread(page_path))

        with plt.rc_context(
            {
                "font.family": "serif",
                "font.size": 11,
                "mathtext.fontset": "dejavuserif",
                "axes.linewidth": 1.0,
            }
        ):
            figure, axis = plt.subplots(figsize=(6.2, 6.2), constrained_layout=True)
            axis.imshow(
                reference_axes,
                cmap="gray",
                extent=(0.0, 2.5, 0.0, 2.5),
                origin="upper",
                interpolation="bilinear",
                alpha=0.72,
                zorder=0,
            )
            reproduced = axis.contour(
                sigma,
                omega,
                c_mu,
                levels=CONTOUR_LEVELS,
                colors="#d62728",
                linewidths=2.25,
                zorder=2,
            )
            axis.clabel(
                reproduced,
                inline=True,
                fontsize=10,
                fmt=lambda value: f"{value:g}",
                colors="#d62728",
            )
            axis.contour(
                sigma,
                omega,
                c_mu,
                levels=[DIAGNOSTIC_LEVEL],
                colors="#1f77b4",
                linestyles="--",
                linewidths=2.25,
                zorder=3,
            )
            axis.plot(
                [],
                [],
                color="#1f77b4",
                linestyle="--",
                linewidth=2.25,
                label=r"computed $C_\mu=0.50$",
            )
            axis.legend(loc="upper right", frameon=True, framealpha=0.9)
            axis.set(
                xlim=(0.0, 2.5),
                ylim=(0.0, 2.5),
                xlabel=r"$\sigma$",
                ylabel=r"$\Omega$",
                xticks=[0, 1, 2],
                yticks=[0, 1, 2],
                title="Pope (black) with reproduced contours (red)",
            )
            axis.set_aspect("equal")
            axis.tick_params(direction="in", top=True, right=True, length=5)
            figure.savefig(output, dpi=300, bbox_inches="tight")
            plt.close(figure)

    return output


def _parse_args() -> argparse.Namespace:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        type=Path,
        default=directory / "pope_1975.pdf",
        help="path to Pope's PDF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=directory / "figure1_overlay.png",
        help="output PNG path",
    )
    parser.add_argument("--points", type=int, default=401)
    parser.add_argument("--render-dpi", type=int, default=600)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    generate_overlay(
        arguments.reference,
        arguments.output,
        points=arguments.points,
        render_dpi=arguments.render_dpi,
    )
