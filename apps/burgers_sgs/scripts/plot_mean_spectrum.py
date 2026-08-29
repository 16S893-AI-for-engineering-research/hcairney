#!/usr/bin/env python3
"""Plot raw and compensated Burgers spectra from CSV or analysis NPZ files."""

import argparse
import csv
import math
from pathlib import Path


def read_csv_spectrum(path):
    """Return the wavenumbers and both energy columns in a spectrum CSV."""
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        required_columns = {
            "wavenumber",
            "mean_energy",
            "k_five_thirds_mean_energy",
        }
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"missing required column(s): {missing}")

        wavenumbers = []
        mean_energies = []
        compensated_energies = []
        for line_number, row in enumerate(reader, start=2):
            try:
                wavenumber = float(row["wavenumber"])
                mean_energy = float(row["mean_energy"])
                compensated_energy = float(
                    row["k_five_thirds_mean_energy"]
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid numeric value on line {line_number}"
                ) from error
            if not all(math.isfinite(value) for value in
                       (wavenumber, mean_energy, compensated_energy)):
                raise ValueError(f"non-finite value on line {line_number}")
            wavenumbers.append(wavenumber)
            mean_energies.append(mean_energy)
            compensated_energies.append(compensated_energy)

    if not wavenumbers:
        raise ValueError("spectrum contains no data rows")
    return wavenumbers, mean_energies, compensated_energies


def read_npz_spectrum(path):
    """Return the wavenumbers and energy arrays in an analysis NPZ file."""
    try:
        import numpy as np
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "NumPy is required to read an analysis NPZ file"
        ) from error

    required_arrays = (
        "wavenumber",
        "mean_energy_spectrum",
        "compensated_energy_spectrum",
    )
    with np.load(path) as arrays:
        missing = [name for name in required_arrays if name not in arrays]
        if missing:
            raise ValueError(
                f"missing required array(s): {', '.join(missing)}"
            )
        wavenumbers, mean_energies, compensated_energies = (
            np.asarray(arrays[name], dtype=float) for name in required_arrays
        )

    values = (wavenumbers, mean_energies, compensated_energies)
    if any(value.ndim != 1 for value in values):
        raise ValueError("spectrum arrays must be one-dimensional")
    if not wavenumbers.size:
        raise ValueError("spectrum contains no data")
    if not (wavenumbers.size == mean_energies.size
            == compensated_energies.size):
        raise ValueError("spectrum arrays must have the same length")
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("spectrum contains non-finite values")
    return wavenumbers, mean_energies, compensated_energies


def read_spectrum(path):
    """Read a mean spectrum CSV or an analysis arrays NPZ file."""
    if path.suffix.lower() == ".npz":
        return read_npz_spectrum(path)
    return read_csv_spectrum(path)


def positive_log_data(wavenumbers, values):
    """Remove values that cannot be represented on logarithmic axes."""
    filtered = [
        (wavenumber, value)
        for wavenumber, value in zip(wavenumbers, values)
        if wavenumber > 0.0 and value > 0.0
    ]
    if not filtered:
        raise ValueError("spectrum has no positive data for logarithmic axes")
    return tuple(zip(*filtered))


def make_figure(path):
    """Create side-by-side raw and compensated spectrum plots."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Matplotlib is required to plot the spectrum"
        ) from error

    wavenumbers, mean_energies, compensated_energies = read_spectrum(path)
    raw_wavenumbers, raw_energies = positive_log_data(
        wavenumbers, mean_energies
    )
    compensated_wavenumbers, positive_compensated_energies = positive_log_data(
        wavenumbers, compensated_energies
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    axes[0].loglog(raw_wavenumbers, raw_energies, linewidth=1.5)
    axes[0].set_title("Mean energy spectrum")
    axes[0].set_xlabel(r"Wavenumber $k$")
    axes[0].set_ylabel(r"$E(k)$")

    axes[1].loglog(
        compensated_wavenumbers,
        positive_compensated_energies,
        linewidth=1.5,
    )
    axes[1].set_title(r"$k^{5/3}$-compensated spectrum")
    axes[1].set_xlabel(r"Wavenumber $k$")
    axes[1].set_ylabel(r"$k^{5/3} E(k)$")

    for axis in axes:
        axis.grid(which="both", alpha=0.3)
    figure.tight_layout()
    return figure


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a burgers_mean_spectrum.csv file produced by run_solver or "
            "an analysis_arrays.npz file produced by analyze_run.py."
        )
    )
    parser.add_argument(
        "spectrum_file", type=Path,
        help="mean-spectrum CSV or analysis-arrays NPZ file",
    )
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
        figure = make_figure(arguments.spectrum_file)
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    if arguments.output is not None:
        figure.savefig(arguments.output, dpi=arguments.dpi)
    if arguments.output is None or arguments.show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
