#!/usr/bin/env python3
"""Plot the distribution of the Burgers SGS observation input feature."""

import argparse
import csv
import json
import math
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit(
        "NumPy is required; install requirements-analysis.txt"
    ) from error


PROFILE_FILENAME = "burgers_profiles.csv"
METADATA_FILENAME = "burgers_run_metadata.json"
DEFAULT_OUTPUT_FILENAME = "input_distribution.png"


def read_run_configuration(run_directory):
    """Return the grid size, cell width, and viscosity for a run."""
    metadata_path = run_directory / METADATA_FILENAME
    with metadata_path.open() as input_file:
        metadata = json.load(input_file)

    try:
        configuration = metadata["configuration"]
        grid = configuration["grid"]
        cell_count = int(grid["cell_count"])
        x_begin = float(grid["x_begin"])
        x_end = float(grid["x_end"])
        viscosity = float(configuration["viscosity"]["molecular"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{metadata_path} does not contain a valid grid and viscosity"
        ) from error

    if cell_count < 3:
        raise ValueError("the periodic centered gradient requires at least 3 cells")
    if not all(math.isfinite(value) for value in
               (x_begin, x_end, viscosity)):
        raise ValueError(f"{metadata_path} contains non-finite run parameters")
    if x_end <= x_begin:
        raise ValueError("the metadata grid must have positive domain length")
    if viscosity <= 0.0:
        raise ValueError("molecular viscosity must be positive")
    return cell_count, (x_end - x_begin) / cell_count, viscosity


def read_profiles(path, cell_count):
    """Read complete profile frames and return their times and velocities."""
    with path.open(newline="") as input_file:
        header = next(csv.reader(input_file), None)
    if header != ["step", "time", "x", "cell_average"]:
        raise ValueError(
            f"{path} must contain step,time,x,cell_average columns in order"
        )
    try:
        data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    except ValueError as error:
        raise ValueError(f"could not parse numeric profile data in {path}") from error

    if data.shape[1] != 4:
        raise ValueError(
            f"{path} must contain step,time,x,cell_average columns"
        )
    if data.shape[0] == 0:
        raise ValueError(f"{path} contains no profile data")
    if not np.all(np.isfinite(data)):
        raise ValueError(f"{path} contains non-finite values")
    if data.shape[0] % cell_count != 0:
        raise ValueError(f"{path} does not contain complete profile frames")

    frame_count = data.shape[0] // cell_count
    frames = data.reshape(frame_count, cell_count, 4)
    steps = frames[:, 0, 0]
    times = frames[:, 0, 1]
    coordinates = frames[0, :, 2]

    if not np.all(frames[:, :, 0] == steps[:, None]):
        raise ValueError(f"{path} has inconsistent steps within a profile frame")
    if not np.all(frames[:, :, 1] == times[:, None]):
        raise ValueError(f"{path} has inconsistent times within a profile frame")
    coordinate_tolerance = 32.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(coordinates)))
    )
    if not np.allclose(
        frames[:, :, 2], coordinates[None, :], rtol=0.0,
        atol=coordinate_tolerance,
    ):
        raise ValueError(f"{path} spatial grid changes between profile frames")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path} frame times must be strictly increasing")
    return times, frames[:, :, 3]


def input_features(profiles, cell_width, viscosity):
    """Compute the exact scalar feature used by ObservationBuilder."""
    gradients = (
        np.roll(profiles, -1, axis=1) - np.roll(profiles, 1, axis=1)
    ) / (2.0 * cell_width)
    return np.log1p(np.abs(gradients) * cell_width ** 2 / viscosity)


def default_bin_width(values):
    """Choose a histogram width using the Freedman-Diaconis rule."""
    flattened = np.asarray(values, dtype=float).ravel()
    if flattened.size == 0:
        raise ValueError("cannot choose a bin width for empty data")
    data_range = float(np.ptp(flattened))
    if data_range == 0.0:
        return max(1.0, abs(float(flattened[0]))) * 0.1

    first_quartile, third_quartile = np.percentile(flattened, [25.0, 75.0])
    width = 2.0 * float(third_quartile - first_quartile) / np.cbrt(
        flattened.size
    )
    if not math.isfinite(width) or width <= 0.0:
        width = 3.5 * float(np.std(flattened)) / np.cbrt(flattened.size)
    if not math.isfinite(width) or width <= 0.0:
        width = data_range / math.sqrt(flattened.size)
    return min(width, data_range)


def bin_edges(values, width):
    """Return fixed-width edges aligned to integer multiples of *width*."""
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("bin width must be finite and positive")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    scaled_minimum = minimum / width
    if not math.isfinite(scaled_minimum):
        raise ValueError("bin width is too small for the feature range")
    start = math.floor(scaled_minimum) * width
    bins_needed = (maximum - start) / width
    if not math.isfinite(bins_needed) or bins_needed > 1_000_000:
        raise ValueError("bin width would create more than 1,000,000 bins")
    bin_count = max(1, int(math.ceil(bins_needed)))
    return start + width * np.arange(bin_count + 1, dtype=float)


def load_run(run_directory):
    """Load one normal or RL evaluation run and calculate all features."""
    if not run_directory.is_dir():
        raise ValueError(f"run directory does not exist: {run_directory}")
    cell_count, cell_width, viscosity = read_run_configuration(run_directory)
    times, profiles = read_profiles(
        run_directory / PROFILE_FILENAME, cell_count
    )
    return times, input_features(profiles, cell_width, viscosity), cell_width, viscosity


def make_figure(features, width, frame_count, cell_count):
    """Create a fixed-bin-width histogram of input feature values."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Matplotlib is required; install requirements-analysis.txt"
        ) from error

    flattened = np.asarray(features, dtype=float).ravel()
    edges = bin_edges(flattened, width)
    figure, axis = plt.subplots(figsize=(8.0, 4.8))
    weights = np.full(flattened.size, 1.0 / flattened.size)
    axis.hist(
        flattened, bins=edges, weights=weights,
        color="tab:blue", edgecolor="white",
    )
    axis.set_title(
        f"Burgers input-feature distribution ({frame_count} frames, "
        f"{cell_count} cells)"
    )
    axis.set_xlabel(r"$\ln(1 + |\partial u/\partial x|\,\Delta^2/\nu)$")
    axis.set_ylabel("Frequency")
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_directory", type=Path,
        help=(
            "run_solver output directory or SMARTIES evaluation episode "
            "directory"
        ),
    )
    parser.add_argument(
        "--bin-width", type=float,
        help="histogram bin width (default: Freedman-Diaconis data estimate)",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help=(
            "output image path (default: input_distribution.png in the run "
            "directory)"
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="output resolution (default: 150)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="display the figure after saving it",
    )
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    run_directory = arguments.run_directory.expanduser()
    if arguments.dpi <= 0:
        parser.error("--dpi must be positive")
    if arguments.bin_width is not None and (
        not math.isfinite(arguments.bin_width) or arguments.bin_width <= 0.0
    ):
        parser.error("--bin-width must be finite and positive")

    try:
        times, features, cell_width, viscosity = load_run(run_directory)
        width = (
            default_bin_width(features)
            if arguments.bin_width is None else arguments.bin_width
        )
        figure = make_figure(
            features, width, frame_count=features.shape[0],
            cell_count=features.shape[1],
        )
        output_path = (
            run_directory / DEFAULT_OUTPUT_FILENAME
            if arguments.output is None else arguments.output.expanduser()
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=arguments.dpi)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    print(
        f"Saved {features.size} samples from {times.size} frames to "
        f"{output_path} (Delta={cell_width:g}, nu={viscosity:g}, "
        f"bin width={width:g})."
    )
    if arguments.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
