#!/usr/bin/env python3
"""Compute block-aware forced-run statistics and spectra."""

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


CORE_SNAPSHOT_COLUMNS = ("mean", "kinetic_energy", "spatial_variance")
CLOSURE_SNAPSHOT_COLUMNS = (
    "minimum_coefficient",
    "mean_coefficient",
    "maximum_coefficient",
    "mean_eddy_viscosity",
    "maximum_eddy_viscosity",
)
SNAPSHOT_COLUMNS = (*CORE_SNAPSHOT_COLUMNS, *CLOSURE_SNAPSHOT_COLUMNS)
RATE_COLUMNS = (
    "molecular_dissipation",
    "sgs_dissipation",
    "deterministic_power",
    "stochastic_power",
    "manufactured_power",
    "interval_numerical_dissipation_rate",
    "total_power",
    "interval_energy_change_rate",
    "interval_budget_residual_rate",
)


def finite_float(value, context):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def read_metadata(path):
    with path.open() as input_file:
        document = json.load(input_file)
    if document.get("result", {}).get("status") != "completed":
        raise ValueError(f"run is not completed: {path}")
    return document


def read_history(path):
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = set(reader.fieldnames or ())
        # Closure columns were added in Phase 7. Retain support for completed
        # Phase 5/6 histories by requiring only their original columns.
        required = {
            "time", *CORE_SNAPSHOT_COLUMNS, "molecular_dissipation",
            "deterministic_power", "stochastic_power", "manufactured_power",
            "interval_numerical_dissipation_rate",
        }
        missing = required.difference(fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError(f"{path} must contain at least two history rows")

    columns = {name: [] for name in fieldnames}
    for line, row in enumerate(rows, start=2):
        for name in fieldnames:
            columns[name].append(finite_float(row[name], f"{path}:{line}:{name}"))
    arrays = {name: np.asarray(values, dtype=float)
              for name, values in columns.items()}
    times = arrays["time"]
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path} times must be strictly increasing")

    # Read old Phase 5/6 histories as zero-SGS runs.
    for name in CLOSURE_SNAPSHOT_COLUMNS:
        if name not in arrays:
            arrays[name] = np.zeros_like(times)
    if "sgs_dissipation" not in arrays:
        arrays["sgs_dissipation"] = np.zeros_like(times)
    if "interval_start_time" not in arrays:
        arrays["interval_start_time"] = np.concatenate(([times[0]], times[:-1]))
    if "interval_duration" not in arrays:
        arrays["interval_duration"] = times - arrays["interval_start_time"]
    if "total_power" not in arrays:
        arrays["total_power"] = (
            arrays["deterministic_power"]
            + arrays["stochastic_power"]
            + arrays["manufactured_power"]
        )
    if "interval_energy_change_rate" not in arrays:
        energy_rate = np.zeros_like(times)
        energy_rate[1:] = np.diff(arrays["kinetic_energy"]) / np.diff(times)
        arrays["interval_energy_change_rate"] = energy_rate
    if "interval_budget_residual_rate" not in arrays:
        arrays["interval_budget_residual_rate"] = (
            arrays["total_power"]
            - arrays["molecular_dissipation"]
            - arrays["sgs_dissipation"]
            - arrays["interval_numerical_dissipation_rate"]
            - arrays["interval_energy_change_rate"]
        )
    return arrays


def integrated_autocorrelation_time(times, values):
    if values.size < 4:
        return None
    differences = np.diff(times)
    time_step = float(np.median(differences))
    tolerance = 1e-6 * max(1.0, abs(time_step))
    if np.max(np.abs(differences - time_step)) > tolerance:
        return None
    centered = values - np.mean(values)
    variance = float(np.dot(centered, centered) / centered.size)
    if variance <= np.finfo(float).eps * max(1.0, float(np.mean(values ** 2))):
        return 0.0
    sample_count = int(centered.size)
    transform_size = 1 << (2 * sample_count - 1).bit_length()
    transform = np.fft.rfft(centered, n=transform_size)
    covariance = np.fft.irfft(transform * np.conjugate(transform),
                              n=transform_size)[:centered.size]
    covariance /= np.arange(centered.size, 0, -1, dtype=float)
    correlation = covariance / covariance[0]
    nonpositive = np.nonzero(correlation[1:] <= 0.0)[0]
    stop = int(nonpositive[0] + 1) if nonpositive.size else centered.size
    return max(0.5 * time_step,
               time_step * (0.5 + float(np.sum(correlation[1:stop]))))


def snapshot_block_means(times, values, start, end, block_duration):
    block_count = int(math.floor((end - start) / block_duration + 1e-12))
    means = []
    for block in range(block_count):
        left = start + block * block_duration
        right = left + block_duration
        mask = (times >= left) & (times < right)
        if np.any(mask):
            means.append(float(np.mean(values[mask])))
    return np.asarray(means, dtype=float)


def rate_block_means(starts, ends, values, start, end, block_duration):
    block_count = int(math.floor((end - start) / block_duration + 1e-12))
    means = []
    for block in range(block_count):
        left = start + block * block_duration
        right = left + block_duration
        overlap = np.maximum(0.0, np.minimum(ends, right) - np.maximum(starts, left))
        weight = float(np.sum(overlap))
        if weight > 0.0:
            means.append(float(np.dot(overlap, values) / weight))
    return np.asarray(means, dtype=float)


def summarize_blocks(block_means):
    if block_means.size < 2:
        return {"block_count": int(block_means.size), "standard_error": None}
    return {
        "block_count": int(block_means.size),
        "standard_error": float(np.std(block_means, ddof=1)
                                / math.sqrt(block_means.size)),
    }


def scalar_statistics(history, start, end, block_duration):
    times = history["time"]
    interval_starts = history["interval_start_time"]
    interval_ends = times
    statistics = {}
    block_values = {}

    # Use (start, end] so adjacent stationarity windows do not share a sample.
    snapshot_mask = (times > start) & (times <= end)
    for name in SNAPSHOT_COLUMNS:
        selected_times = times[snapshot_mask]
        selected = history[name][snapshot_mask]
        if selected.size == 0:
            raise ValueError(f"window contains no {name} samples")
        blocks = snapshot_block_means(
            selected_times, selected, start, end, block_duration
        )
        block_values[name] = blocks
        entry = {
            "mean": float(np.mean(selected)),
            "sample_count": int(selected.size),
            "block_means": blocks.tolist(),
            "integrated_autocorrelation_time":
                integrated_autocorrelation_time(selected_times, selected),
        }
        entry.update(summarize_blocks(blocks))
        statistics[name] = entry

    overlap = np.maximum(
        0.0, np.minimum(interval_ends, end) - np.maximum(interval_starts, start)
    )
    rate_mask = overlap > 0.0
    total_weight = float(np.sum(overlap))
    if total_weight <= 0.0:
        raise ValueError("window contains no scalar-history intervals")
    for name in RATE_COLUMNS:
        selected = history[name]
        blocks = rate_block_means(
            interval_starts, interval_ends, selected,
            start, end, block_duration
        )
        block_values[name] = blocks
        entry = {
            "mean": float(np.dot(overlap, selected) / total_weight),
            "sample_count": int(np.count_nonzero(rate_mask)),
            "block_means": blocks.tolist(),
            "integrated_autocorrelation_time":
                integrated_autocorrelation_time(
                    times[rate_mask], selected[rate_mask]
                ),
        }
        entry.update(summarize_blocks(blocks))
        statistics[name] = entry
    return statistics, block_values


def read_profiles(path, cell_count):
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != 4 or data.shape[0] % cell_count != 0:
        raise ValueError(f"{path} does not contain complete profile frames")
    frame_count = data.shape[0] // cell_count
    reshaped = data.reshape(frame_count, cell_count, 4)
    times = reshaped[:, 0, 1]
    coordinates = reshaped[0, :, 2]
    profiles = reshaped[:, :, 3]
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(f"{path} frame times must be strictly increasing")
    if not np.allclose(reshaped[:, :, 2], coordinates[None, :], rtol=0.0,
                       atol=32.0 * np.finfo(float).eps):
        raise ValueError(f"{path} spatial grid changes between frames")
    return times, coordinates, profiles


def one_sided_energy_spectra(profiles, domain_length):
    cell_count = profiles.shape[1]
    coefficients = np.fft.rfft(profiles, axis=1) / cell_count
    spectra = domain_length * np.abs(coefficients) ** 2
    spectra[:, 0] *= 0.5
    if cell_count % 2 == 0:
        spectra[:, -1] *= 0.5
    return spectra


def array_standard_error(block_arrays):
    if block_arrays.shape[0] < 2:
        return np.full(block_arrays.shape[1:], np.nan)
    return np.std(block_arrays, axis=0, ddof=1) / math.sqrt(block_arrays.shape[0])


def profile_statistics(times, coordinates, profiles, start, end,
                       block_duration, domain_length):
    mask = (times > start) & (times <= end)
    times = times[mask]
    profiles = profiles[mask]
    if profiles.shape[0] < 2:
        raise ValueError("analysis window must contain at least two profiles")
    cell_count = profiles.shape[1]
    cell_width = domain_length / cell_count
    spectra = one_sided_energy_spectra(profiles, domain_length)
    gradients = (np.roll(profiles, -1, axis=1) - profiles) / cell_width

    block_count = int(math.floor((end - start) / block_duration + 1e-12))
    mean_profile_blocks = []
    variance_profile_blocks = []
    spectrum_blocks = []
    retained_block_starts = []
    retained_block_ends = []
    for block in range(block_count):
        left = start + block * block_duration
        right = left + block_duration
        block_mask = (times >= left) & (times < right)
        if np.count_nonzero(block_mask) < 2:
            continue
        block_profiles = profiles[block_mask]
        mean_profile_blocks.append(np.mean(block_profiles, axis=0))
        variance_profile_blocks.append(np.var(block_profiles, axis=0, ddof=1))
        spectrum_blocks.append(np.mean(spectra[block_mask], axis=0))
        retained_block_starts.append(left)
        retained_block_ends.append(right)

    spectrum_size = spectra.shape[1]
    mean_profile_blocks = (np.asarray(mean_profile_blocks)
                           if mean_profile_blocks
                           else np.empty((0, cell_count)))
    variance_profile_blocks = (np.asarray(variance_profile_blocks)
                               if variance_profile_blocks
                               else np.empty((0, cell_count)))
    spectrum_blocks = (np.asarray(spectrum_blocks)
                       if spectrum_blocks
                       else np.empty((0, spectrum_size)))
    mean_profile = np.mean(profiles, axis=0)
    temporal_variance = np.var(profiles, axis=0, ddof=1)
    mean_spectrum = np.mean(spectra, axis=0)
    wavenumbers = np.arange(mean_spectrum.size)
    compensated = np.zeros_like(mean_spectrum)
    compensated[1:] = wavenumbers[1:] ** (5.0 / 3.0) * mean_spectrum[1:]

    arrays = {
        "x": coordinates,
        "mean_profile": mean_profile,
        "mean_profile_standard_error": array_standard_error(mean_profile_blocks),
        "temporal_variance_profile": temporal_variance,
        "temporal_variance_profile_standard_error":
            array_standard_error(variance_profile_blocks),
        "wavenumber": wavenumbers,
        "mean_energy_spectrum": mean_spectrum,
        "mean_energy_spectrum_standard_error": array_standard_error(spectrum_blocks),
        "compensated_energy_spectrum": compensated,
        # Retain complete-block estimates so Phase 6 comparisons can preserve
        # temporal and spatial correlations instead of reconstructing them
        # from pointwise standard errors.
        "block_start_time": np.asarray(retained_block_starts, dtype=float),
        "block_end_time": np.asarray(retained_block_ends, dtype=float),
        "mean_profile_block_means": mean_profile_blocks,
        "temporal_variance_profile_block_means": variance_profile_blocks,
        "mean_energy_spectrum_block_means": spectrum_blocks,
    }
    summary = {
        "profile_sample_count": int(profiles.shape[0]),
        "profile_block_count": int(mean_profile_blocks.shape[0]),
        "domain_averaged_temporal_variance": float(np.mean(temporal_variance)),
        "mean_gradient_square": float(np.mean(gradients ** 2)),
        "maximum_absolute_gradient": float(np.max(np.abs(gradients))),
        "maximum_absolute_cell_jump": float(
            np.max(np.abs(np.roll(profiles, -1, axis=1) - profiles))
        ),
        "high_wavenumber_energy_fraction": float(
            np.sum(mean_spectrum[max(1, mean_spectrum.size * 3 // 4):])
            / np.sum(mean_spectrum[1:])
        ) if np.sum(mean_spectrum[1:]) > 0.0 else 0.0,
    }
    return summary, arrays


def stationarity_comparison(history, start, split, end, block_duration):
    first, first_blocks = scalar_statistics(history, start, split, block_duration)
    second, second_blocks = scalar_statistics(history, split, end, block_duration)
    comparison = {}
    for name in (*SNAPSHOT_COLUMNS, *RATE_COLUMNS):
        left = first_blocks[name]
        right = second_blocks[name]
        difference = second[name]["mean"] - first[name]["mean"]
        if left.size >= 2 and right.size >= 2:
            left_variance = float(np.var(left, ddof=1) / left.size)
            right_variance = float(np.var(right, ddof=1) / right.size)
            standard_error = math.sqrt(left_variance + right_variance)
            statistic = (abs(difference) / standard_error
                         if standard_error > 0.0 else None)
            numerator = (left_variance + right_variance) ** 2
            denominator = (
                left_variance ** 2 / (left.size - 1)
                + right_variance ** 2 / (right.size - 1)
            )
            degrees_of_freedom = numerator / denominator if denominator else None
        else:
            standard_error = None
            statistic = None
            degrees_of_freedom = None
        comparison[name] = {
            "first_mean": first[name]["mean"],
            "second_mean": second[name]["mean"],
            "difference": difference,
            "difference_standard_error": standard_error,
            "welch_t_statistic": statistic,
            "welch_degrees_of_freedom": degrees_of_freedom,
        }
    return comparison


def parse_window(value):
    try:
        start_text, end_text = value.split(":", 1)
        start = float(start_text)
        end = float(end_text)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("window must be START:END") from error
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise argparse.ArgumentTypeError("window must have finite END > START")
    return start, end


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--metadata-filename",
                        default="burgers_run_metadata.json")
    parser.add_argument("--window", type=parse_window,
                        help="analysis window START:END; defaults to metadata")
    parser.add_argument("--block-duration", type=float,
                        help="physical duration of independent sampling blocks")
    parser.add_argument("--stationarity-split", type=float,
                        help="compare scalar statistics before and after TIME")
    parser.add_argument("--output-prefix", default="analysis",
                        help="output basename within the run directory")
    parser.add_argument("--no-profiles", action="store_true",
                        help="analyze only scalar history")
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    try:
        run_directory = arguments.run_directory
        metadata_path = run_directory / arguments.metadata_filename
        metadata = read_metadata(metadata_path)
        configuration = metadata["configuration"]
        history_path = run_directory / configuration["output"][
            "scalar_history_filename"
        ]
        history = read_history(history_path)
        if arguments.window is None:
            start = float(configuration["output"]["statistics_start_time"])
            end = float(metadata["result"]["final_time"])
        else:
            start, end = arguments.window
        if end <= start:
            raise ValueError("analysis window must have END > START")

        candidate_columns = (
            "kinetic_energy", "spatial_variance", "molecular_dissipation",
            "sgs_dissipation", "interval_numerical_dissipation_rate",
            "total_power",
        )
        window_mask = (history["time"] > start) & (history["time"] <= end)
        correlation_times = []
        for name in candidate_columns:
            estimate = integrated_autocorrelation_time(
                history["time"][window_mask], history[name][window_mask]
            )
            if estimate is not None:
                correlation_times.append(estimate)
        if arguments.block_duration is None:
            positive = [value for value in correlation_times if value > 0.0]
            if positive:
                block_duration = 5.0 * max(positive)
            else:
                block_duration = (end - start) / 10.0
        else:
            block_duration = arguments.block_duration
        if not math.isfinite(block_duration) or block_duration <= 0.0:
            raise ValueError("block duration must be finite and positive")

        scalars, _ = scalar_statistics(history, start, end, block_duration)
        summary = {
            "schema_version": 1,
            "run_directory": str(run_directory.resolve()),
            "source_metadata": str(metadata_path.resolve()),
            "seed": configuration["random"]["seed"],
            "window": {"start": start, "end": end, "duration": end - start},
            "block_duration": block_duration,
            "scalars": scalars,
            "energy_budget": {
                "mean_total_power": scalars["total_power"]["mean"],
                "mean_molecular_dissipation":
                    scalars["molecular_dissipation"]["mean"],
                "mean_sgs_dissipation":
                    scalars["sgs_dissipation"]["mean"],
                "mean_numerical_dissipation":
                    scalars["interval_numerical_dissipation_rate"]["mean"],
                "mean_energy_change_rate":
                    scalars["interval_energy_change_rate"]["mean"],
                "mean_residual_rate":
                    scalars["interval_budget_residual_rate"]["mean"],
                "numerical_to_molecular_dissipation_ratio": (
                    scalars["interval_numerical_dissipation_rate"]["mean"]
                    / scalars["molecular_dissipation"]["mean"]
                    if scalars["molecular_dissipation"]["mean"] != 0.0
                    else None
                ),
                "numerical_to_total_physical_dissipation_ratio": (
                    scalars["interval_numerical_dissipation_rate"]["mean"]
                    / (
                        scalars["molecular_dissipation"]["mean"]
                        + scalars["sgs_dissipation"]["mean"]
                    )
                    if (
                        scalars["molecular_dissipation"]["mean"]
                        + scalars["sgs_dissipation"]["mean"]
                    ) != 0.0 else None
                ),
            },
        }
        arrays = {}
        if not arguments.no_profiles:
            profile_filename = configuration["output"]["profile_history_filename"]
            profile_path = run_directory / profile_filename
            cell_count = int(configuration["grid"]["cell_count"])
            profile_times, coordinates, profiles = read_profiles(
                profile_path, cell_count
            )
            domain_length = (
                float(configuration["grid"]["x_end"])
                - float(configuration["grid"]["x_begin"])
            )
            profile_summary, arrays = profile_statistics(
                profile_times, coordinates, profiles,
                start, end, block_duration, domain_length
            )
            summary["profiles"] = profile_summary

        if arguments.stationarity_split is not None:
            split = arguments.stationarity_split
            if not start < split < end:
                raise ValueError("stationarity split must lie inside the window")
            summary["stationarity"] = {
                "split_time": split,
                "comparison": stationarity_comparison(
                    history, start, split, end, block_duration
                ),
            }

        summary_path = run_directory / f"{arguments.output_prefix}_summary.json"
        arrays_path = run_directory / f"{arguments.output_prefix}_arrays.npz"
        with summary_path.open("w") as output:
            json.dump(summary, output, indent=2, allow_nan=False)
            output.write("\n")
        if arrays:
            np.savez_compressed(arrays_path, **arrays)
        print(f"Wrote {summary_path}")
        if arrays:
            print(f"Wrote {arrays_path}")
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
