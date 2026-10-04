#!/usr/bin/env python3
"""Compare two analyzed runs over their common resolved scales."""

import argparse
import json
import math
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit(
        "NumPy is required; install requirements-analysis.txt"
    ) from error


def analysis_paths(value):
    if value.is_dir():
        return value / "analysis_summary.json", value / "analysis_arrays.npz"
    if value.name.endswith("_summary.json"):
        return value, value.with_name(value.name.replace("_summary.json",
                                                         "_arrays.npz"))
    raise ValueError(f"cannot identify analysis files from {value}")


def load_analysis(value):
    summary_path, arrays_path = analysis_paths(value)
    with summary_path.open() as input_file:
        summary = json.load(input_file)
    arrays = np.load(arrays_path)
    return summary_path, arrays_path, summary, arrays


def restrict_periodic_cell_averages(values, target_size):
    source_size = values.size
    if source_size == target_size:
        return values
    if source_size % target_size != 0:
        raise ValueError(
            f"cannot conservatively restrict {source_size} cells to {target_size}"
        )
    ratio = source_size // target_size
    return np.mean(values.reshape(target_size, ratio), axis=1)


def relative_l2(left, right):
    denominator = float(np.linalg.norm(right))
    difference = float(np.linalg.norm(left - right))
    return difference / denominator if denominator > 0.0 else difference


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("comparison.json"))
    parser.add_argument("--maximum-mode", type=int,
                        help="largest common Fourier mode to compare")
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    try:
        candidate = load_analysis(arguments.candidate)
        reference = load_analysis(arguments.reference)
        candidate_summary = candidate[2]
        reference_summary = reference[2]
        if candidate_summary["window"] != reference_summary["window"]:
            raise ValueError("analysis windows differ")

        candidate_arrays = candidate[3]
        reference_arrays = reference[3]
        candidate_cells = candidate_arrays["mean_profile"].size
        reference_cells = reference_arrays["mean_profile"].size
        common_cells = min(candidate_cells, reference_cells)
        candidate_mean = restrict_periodic_cell_averages(
            candidate_arrays["mean_profile"], common_cells
        )
        reference_mean = restrict_periodic_cell_averages(
            reference_arrays["mean_profile"], common_cells
        )
        candidate_variance = restrict_periodic_cell_averages(
            candidate_arrays["temporal_variance_profile"], common_cells
        )
        reference_variance = restrict_periodic_cell_averages(
            reference_arrays["temporal_variance_profile"], common_cells
        )

        default_maximum_mode = common_cells // 4
        maximum_mode = (arguments.maximum_mode
                        if arguments.maximum_mode is not None
                        else default_maximum_mode)
        available_mode = min(
            candidate_arrays["mean_energy_spectrum"].size,
            reference_arrays["mean_energy_spectrum"].size,
        ) - 1
        if maximum_mode < 1 or maximum_mode > available_mode:
            raise ValueError(
                f"maximum mode must lie between 1 and {available_mode}"
            )
        mode_slice = slice(1, maximum_mode + 1)
        candidate_spectrum = candidate_arrays["mean_energy_spectrum"][mode_slice]
        reference_spectrum = reference_arrays["mean_energy_spectrum"][mode_slice]
        spectrum_denominator = float(np.sum(np.abs(reference_spectrum)))
        spectrum_l1 = float(np.sum(np.abs(candidate_spectrum-reference_spectrum)))

        scalar_comparison = {}
        common_scalars = set(candidate_summary["scalars"]).intersection(
            reference_summary["scalars"]
        )
        for name in sorted(common_scalars):
            candidate_entry = candidate_summary["scalars"][name]
            reference_entry = reference_summary["scalars"][name]
            difference = candidate_entry["mean"] - reference_entry["mean"]
            scale = abs(reference_entry["mean"])
            candidate_se = candidate_entry.get("standard_error")
            reference_se = reference_entry.get("standard_error")
            combined_se = None
            standardized = None
            if candidate_se is not None and reference_se is not None:
                combined_se = math.sqrt(candidate_se ** 2 + reference_se ** 2)
                standardized = (abs(difference) / combined_se
                                if combined_se > 0.0 else None)
            scalar_comparison[name] = {
                "candidate_mean": candidate_entry["mean"],
                "reference_mean": reference_entry["mean"],
                "difference": difference,
                "relative_difference": abs(difference) / scale if scale else None,
                "combined_block_standard_error": combined_se,
                "standardized_absolute_difference": standardized,
            }

        output = {
            "schema_version": 1,
            "candidate_summary": str(candidate[0].resolve()),
            "reference_summary": str(reference[0].resolve()),
            "window": candidate_summary["window"],
            "common_cell_count": common_cells,
            "maximum_compared_mode": maximum_mode,
            "mean_profile_relative_l2_difference":
                relative_l2(candidate_mean, reference_mean),
            "temporal_variance_profile_relative_l2_difference":
                relative_l2(candidate_variance, reference_variance),
            "spectrum_relative_l1_difference":
                spectrum_l1 / spectrum_denominator
                if spectrum_denominator > 0.0 else spectrum_l1,
            "scalars": scalar_comparison,
        }
        with arguments.output.open("w") as output_file:
            json.dump(output, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
        print(f"Wrote {arguments.output}")
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
