#!/usr/bin/env python3
"""Plot a final Burgers velocity profile written by run_solver."""

import argparse
import csv
import math
from pathlib import Path


def read_profile(path):
    """Return the time, cell centers, and cell averages in a profile CSV."""
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {"time", "x", "cell_average"}
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"missing required column(s): {missing}")

        times = []
        cell_centers = []
        cell_averages = []
        for line_number, row in enumerate(reader, start=2):
            try:
                time = float(row["time"])
                cell_center = float(row["x"])
                cell_average = float(row["cell_average"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid numeric value on line {line_number}"
                ) from error
            if not all(math.isfinite(value) for value in
                       (time, cell_center, cell_average)):
                raise ValueError(f"non-finite value on line {line_number}")
            times.append(time)
            cell_centers.append(cell_center)
            cell_averages.append(cell_average)

    if not cell_centers:
        raise ValueError("profile contains no data rows")
    return times, cell_centers, cell_averages


def make_figure(path):
    """Create a matplotlib figure for the profile in *path*."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Matplotlib is required to plot the profile"
        ) from error

    times, cell_centers, cell_averages = read_profile(path)

    figure, axes = plt.subplots(figsize=(7.0, 4.5))
    axes.plot(cell_centers, cell_averages, linewidth=1.5)
    axes.set_xlabel(r"$x$")
    axes.set_ylabel(r"Cell-average velocity $u$")
    if all(math.isclose(time, times[0]) for time in times[1:]):
        axes.set_title(f"Final profile at t = {times[0]:g}")
    else:
        axes.set_title("Burgers profile")
    axes.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot a burgers_final_profile.csv file produced by run_solver."
    )
    parser.add_argument("csv_file", type=Path, help="final-profile CSV file")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="save the figure to this path instead of only displaying it",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="resolution used with --output (default: 150)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="display the figure even when --output is supplied",
    )
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    try:
        figure = make_figure(arguments.csv_file)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    if arguments.output is not None:
        figure.savefig(arguments.output, dpi=arguments.dpi)
    if arguments.output is None or arguments.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
