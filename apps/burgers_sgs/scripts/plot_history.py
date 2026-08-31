#!/usr/bin/env python3
"""Plot kinetic energy and the energy budget from burgers_history.csv."""

import argparse
import csv
import math
from pathlib import Path


BASE_COLUMNS = (
    "time",
    "kinetic_energy",
    "molecular_dissipation",
    "deterministic_power",
    "stochastic_power",
    "manufactured_power",
    "interval_numerical_dissipation_rate",
)


def read_history(path):
    """Read and validate the scalar history needed by the plots."""
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or ())
        missing = set(BASE_COLUMNS).difference(fieldnames)
        if missing:
            raise ValueError(
                f"missing required column(s): {', '.join(sorted(missing))}"
            )

        columns = {name: [] for name in fieldnames}
        for line_number, row in enumerate(reader, start=2):
            for name in fieldnames:
                try:
                    value = float(row[name])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"invalid numeric value for {name!r} on line "
                        f"{line_number}"
                    ) from error
                if not math.isfinite(value):
                    raise ValueError(
                        f"non-finite value for {name!r} on line {line_number}"
                    )
                columns[name].append(value)

    if not columns["time"]:
        raise ValueError("history contains no data rows")
    if any(right <= left for left, right in
           zip(columns["time"], columns["time"][1:])):
        raise ValueError("history times must be strictly increasing")

    if "prescribed_power" not in columns:
        columns["prescribed_power"] = [0.0] * len(columns["time"])

    total_power = columns.get("total_power")
    if total_power is None:
        total_power = [
            deterministic + stochastic + prescribed + manufactured
            for deterministic, stochastic, prescribed, manufactured in zip(
                columns["deterministic_power"],
                columns["stochastic_power"],
                columns["prescribed_power"],
                columns["manufactured_power"],
            )
        ]
        columns["total_power"] = total_power

    if "interval_energy_change_rate" not in columns:
        energy_rate = [0.0]
        energy_rate.extend(
            (right_energy - left_energy) / (right_time - left_time)
            for left_time, right_time, left_energy, right_energy in zip(
                columns["time"],
                columns["time"][1:],
                columns["kinetic_energy"],
                columns["kinetic_energy"][1:],
            )
        )
        columns["interval_energy_change_rate"] = energy_rate

    if "interval_budget_residual_rate" not in columns:
        columns["interval_budget_residual_rate"] = [
            power - molecular - numerical - energy_rate
            for power, molecular, numerical, energy_rate in zip(
                columns["total_power"],
                columns["molecular_dissipation"],
                columns["interval_numerical_dissipation_rate"],
                columns["interval_energy_change_rate"],
            )
        ]
    return columns


def make_figures(history):
    """Create the kinetic-energy and energy-balance figures."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Matplotlib is required; install requirements-analysis.txt"
        ) from error

    times = history["time"]

    kinetic_figure, kinetic_axis = plt.subplots(figsize=(8.0, 4.8))
    kinetic_axis.plot(
        times, history["kinetic_energy"], linewidth=1.5, color="tab:blue"
    )
    kinetic_axis.set_title("Kinetic energy history")
    kinetic_axis.set_xlabel("Time")
    kinetic_axis.set_ylabel("Kinetic energy")
    kinetic_axis.grid(alpha=0.3)
    kinetic_figure.tight_layout()

    # The first row is an instantaneous initial-state diagnostic, whereas all
    # later rows contain averages over the preceding history interval.
    start = 1 if len(times) > 1 else 0
    budget_times = times[start:]
    molecular = [
        -value for value in history["molecular_dissipation"][start:]
    ]
    numerical = [
        -value
        for value in history["interval_numerical_dissipation_rate"][start:]
    ]
    power = history["total_power"][start:]

    balance_figure, balance_axis = plt.subplots(figsize=(8.0, 4.8))
    balance_axis.plot(
        budget_times, power, linewidth=1.8, label="Total power input"
    )
    balance_axis.plot(
        budget_times,
        molecular,
        linewidth=1.0,
        alpha=0.8,
        label="Molecular dissipation (negative)",
    )
    balance_axis.plot(
        budget_times,
        numerical,
        linewidth=1.0,
        alpha=0.8,
        label="Numerical dissipation (negative)",
    )
    balance_axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    balance_axis.set_title("Energy injection/dissipation balance")
    balance_axis.set_xlabel("Time")
    balance_axis.set_ylabel("Energy rate")
    balance_axis.grid(alpha=0.3)
    balance_axis.legend()
    balance_figure.tight_layout()
    return kinetic_figure, balance_figure


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot kinetic energy and the energy budget from a "
            "burgers_history.csv file."
        )
    )
    parser.add_argument("history_file", type=Path, help="history CSV to plot")
    parser.add_argument(
        "-o",
        "--output-directory",
        type=Path,
        help="output directory (default: directory containing the input CSV)",
    )
    parser.add_argument(
        "--dpi", type=int, default=150, help="PNG resolution (default: 150)"
    )
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    if arguments.dpi <= 0:
        parser.error("--dpi must be positive")

    history_path = arguments.history_file.expanduser()
    output_directory = (
        arguments.output_directory.expanduser()
        if arguments.output_directory is not None
        else history_path.parent
    )
    try:
        history = read_history(history_path)
        kinetic_figure, balance_figure = make_figures(history)
        output_directory.mkdir(parents=True, exist_ok=True)
        kinetic_figure.savefig(output_directory / "KE.png", dpi=arguments.dpi)
        balance_figure.savefig(
            output_directory / "balance.png", dpi=arguments.dpi
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
