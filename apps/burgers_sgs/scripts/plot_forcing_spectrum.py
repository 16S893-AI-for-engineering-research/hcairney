#!/usr/bin/env python3
"""Plot time-averaged raw and applied action spectra after the warmup cutoff."""

import argparse
import csv
import json
import math
from pathlib import Path

from plot_mean_spectrum import output_path_for_spectrum, positive_log_data


def read_spectrum(path):
    """Average each wavenumber equally over rows with end_time >= the cutoff."""
    path = Path(path)
    metadata_path = path.parent / "burgers_run_metadata.json"
    with metadata_path.open() as input_file:
        metadata = json.load(input_file)
    try:
        start = float(metadata["configuration"]["output"]["statistics_start_time"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{metadata_path}: missing or invalid configuration.output.statistics_start_time"
        ) from error
    if not math.isfinite(start):
        raise ValueError("statistics_start_time must be finite")

    totals = {}
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        columns = ("end_time", "wavenumber", "raw_action_energy", "applied_action_energy")
        missing = set(columns).difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"missing required column(s): {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                time, wavenumber, raw, applied = (float(row[name]) for name in columns)
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid numeric value on line {line_number}") from error
            if not all(math.isfinite(value) for value in (time, wavenumber, raw, applied)):
                raise ValueError(f"non-finite value on line {line_number}")
            if time < start:
                continue
            sums = totals.setdefault(wavenumber, [0.0, 0.0, 0])
            sums[0] += raw
            sums[1] += applied
            sums[2] += 1

    if not totals:
        raise ValueError(f"spectrum contains no data at or after statistics_start_time={start:g}")
    wavenumbers = sorted(totals)
    return (
        wavenumbers,
        [totals[k][0] / totals[k][2] for k in wavenumbers],
        [totals[k][1] / totals[k][2] for k in wavenumbers],
    )


def make_figure(path):
    """Create a single, uncompensated plot of the two mean action spectra."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError("Matplotlib is required to plot the spectrum") from error

    wavenumbers, raw, applied = read_spectrum(path)
    figure, axis = plt.subplots(figsize=(6.0, 4.5))
    for values, label in ((raw, "Raw action"), (applied, "Applied action")):
        k, energy = positive_log_data(wavenumbers, values)
        axis.loglog(k, energy, linewidth=1.5, label=label)
    axis.set_title("Mean forcing spectrum")
    axis.set_xlabel(r"Wavenumber $k$")
    axis.set_ylabel(r"$E(k)$")
    axis.grid(which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(description=(
        "Plot arithmetic means of raw and applied action spectra from burgers_rl_spectra.csv, "
        "including end_time >= configuration.output.statistics_start_time from the "
        "burgers_run_metadata.json in the same directory."
    ))
    parser.add_argument("spectrum_file", type=Path, help="burgers_rl_spectra.csv file")
    parser.add_argument("-o", "--output", type=Path, default=Path("forcing_spectrum.png"),
                        help="output filename beside the CSV (default: forcing_spectrum.png)")
    parser.add_argument("--dpi", type=int, default=150, help="output resolution (default: 150)")
    parser.add_argument("--show", action="store_true", help="also display the figure")
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    try:
        figure = make_figure(arguments.spectrum_file)
        figure.savefig(output_path_for_spectrum(arguments.spectrum_file, arguments.output),
                       dpi=arguments.dpi)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    if arguments.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
