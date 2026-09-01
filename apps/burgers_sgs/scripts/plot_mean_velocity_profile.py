#!/usr/bin/env python3
"""Plot a mean Burgers velocity profile with standard-error shading."""

import argparse
from pathlib import Path


def read_mean_velocity_profile(path):
    """Read a run or ensemble mean profile from an analysis NPZ file."""
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "NumPy is required to read an analysis NPZ file"
        ) from error

    required_arrays = ("x", "mean_profile", "mean_profile_standard_error")
    with np.load(path) as arrays:
        missing = [name for name in required_arrays if name not in arrays]
        if missing:
            raise ValueError(
                f"missing required array(s): {', '.join(missing)}"
            )
        coordinates, mean_profile, standard_error = (
            np.asarray(arrays[name], dtype=float).copy()
            for name in required_arrays
        )
        if "mean_profile_standard_error_kind" in arrays:
            kind_array = np.asarray(
                arrays["mean_profile_standard_error_kind"]
            )
            if kind_array.ndim != 0:
                raise ValueError(
                    "mean_profile_standard_error_kind must be a scalar"
                )
            standard_error_kind = str(kind_array.item())
        else:
            standard_error_kind = "temporal"

        seed_count = None
        if "seed_count" in arrays:
            seed_count_array = np.asarray(arrays["seed_count"])
            if seed_count_array.ndim != 0:
                raise ValueError("seed_count must be a scalar")
            seed_count = int(seed_count_array.item())

    values = (coordinates, mean_profile, standard_error)
    if any(value.ndim != 1 for value in values):
        raise ValueError("mean-profile arrays must be one-dimensional")
    if coordinates.size == 0:
        raise ValueError("mean profile contains no data")
    if not (coordinates.size == mean_profile.size == standard_error.size):
        raise ValueError("mean-profile arrays must have the same length")
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("mean profile contains non-finite values")
    if np.any(standard_error < 0.0):
        raise ValueError("mean-profile standard errors must be non-negative")
    if standard_error_kind not in ("temporal", "between_seed"):
        raise ValueError(
            "mean_profile_standard_error_kind must be temporal or between_seed"
        )
    if standard_error_kind == "between_seed" and (
            seed_count is None or seed_count < 2):
        raise ValueError(
            "between-seed uncertainty requires a seed_count of at least two"
        )
    return (
        coordinates, mean_profile, standard_error,
        standard_error_kind, seed_count,
    )


def profile_labels(standard_error_kind, seed_count):
    """Return a source-aware title and uncertainty label."""
    if standard_error_kind == "between_seed":
        return (
            f"Ensemble mean velocity profile ({seed_count} seeds)",
            r"Between-seed standard error ($\pm 1$ SE)",
        )
    return (
        "Temporal mean velocity profile",
        r"Temporal block standard error ($\pm 1$ SE)",
    )


def make_figure(path, xlim=None, ylim=None):
    """Create a mean-profile figure from a run or ensemble analysis NPZ."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Matplotlib is required to plot the mean velocity profile"
        ) from error

    (coordinates, mean_profile, standard_error,
     standard_error_kind, seed_count) = read_mean_velocity_profile(path)
    title, uncertainty_label = profile_labels(
        standard_error_kind, seed_count
    )

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    axis.plot(coordinates, mean_profile, linewidth=1.5, label="Mean velocity")
    axis.fill_between(
        coordinates,
        mean_profile - standard_error,
        mean_profile + standard_error,
        alpha=0.25,
        label=uncertainty_label,
    )
    axis.set_title(title)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"Mean velocity $\overline{u}$")
    if xlim is not None:
        axis.set_xlim(*xlim)
    if ylim is not None:
        axis.set_ylim(*ylim)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def output_path_for_profile(profile_path, output_path):
    """Place the output file alongside the input analysis NPZ file."""
    return profile_path.parent / output_path.name


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis_file", type=Path,
        help="run or ensemble analysis-arrays NPZ file",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("mean_velocity_profile.png"),
        help=(
            "save the figure with this filename beside the analysis file "
            "(default: mean_velocity_profile.png)"
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="resolution used with --output (default: 150)",
    )
    parser.add_argument(
        "--xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"),
        help="set the x-axis limits (for example: --xlim 0.0 1.0)",
    )
    parser.add_argument(
        "--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"),
        help="set the y-axis limits (for example: --ylim -0.5 0.5)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="display the figure even when --output is supplied",
    )
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    try:
        figure = make_figure(
            arguments.analysis_file,
            xlim=arguments.xlim,
            ylim=arguments.ylim,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    output_path = output_path_for_profile(
        arguments.analysis_file, arguments.output
    )
    figure.savefig(output_path, dpi=arguments.dpi)
    if arguments.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
