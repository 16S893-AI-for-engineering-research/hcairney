#!/usr/bin/env python3
"""Compare one analyzed LES baseline with an accepted Phase 6 DNS bundle."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit(
        "NumPy is required; install requirements-analysis.txt"
    ) from error


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analysis_paths(value):
    if value.is_dir():
        return value / "analysis_summary.json", value / "analysis_arrays.npz"
    if value.name.endswith("_summary.json"):
        return value, value.with_name(
            value.name.replace("_summary.json", "_arrays.npz")
        )
    raise ValueError(f"cannot identify analysis files from {value}")


def read_csv_columns(path, required):
    with path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        fields = set(reader.fieldnames or ())
        missing = set(required).difference(fields)
        if missing:
            raise ValueError(
                f"{path} is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no data")
    result = {}
    for name in required:
        values = np.asarray([float(row[name]) for row in rows], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{path}:{name} contains non-finite values")
        result[name] = values
    return result


def restrict_periodic_cell_averages(values, target_size):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("periodic restriction requires a one-dimensional field")
    if values.size == target_size:
        return values.copy()
    if target_size <= 0 or values.size % target_size != 0:
        raise ValueError(
            f"cannot conservatively restrict {values.size} cells to "
            f"{target_size}"
        )
    ratio = values.size // target_size
    return np.mean(values.reshape(target_size, ratio), axis=1)


def relative_l2(candidate, reference):
    difference = float(np.linalg.norm(candidate - reference))
    scale = float(np.linalg.norm(reference))
    return difference / scale if scale > 0.0 else difference


def scalar_comparisons(candidate_scalars, target_scalars):
    comparisons = {}
    for name in sorted(set(candidate_scalars).intersection(target_scalars)):
        candidate = float(candidate_scalars[name]["mean"])
        target = float(target_scalars[name]["mean"])
        difference = candidate - target
        target_se = target_scalars[name].get("combined_standard_error")
        candidate_se = candidate_scalars[name].get("standard_error")
        combined_se = None
        if target_se is not None and candidate_se is not None:
            combined_se = math.sqrt(float(target_se) ** 2
                                    + float(candidate_se) ** 2)
        comparisons[name] = {
            "les_mean": candidate,
            "dns_mean": target,
            "difference": difference,
            "relative_difference": (
                abs(difference) / abs(target) if target != 0.0 else None
            ),
            "combined_standard_error": combined_se,
            "standardized_absolute_difference": (
                abs(difference) / combined_se
                if combined_se is not None and combined_se > 0.0 else None
            ),
        }
    return comparisons


def compare(analysis, target_metadata_path, maximum_mode=None):
    summary_path, arrays_path = analysis_paths(analysis)
    with summary_path.open() as input_file:
        summary = json.load(input_file)
    with np.load(arrays_path) as archive:
        required_arrays = (
            "mean_profile",
            "temporal_variance_profile",
            "mean_energy_spectrum",
        )
        missing = [name for name in required_arrays if name not in archive]
        if missing:
            raise ValueError(
                f"{arrays_path} is missing arrays: {', '.join(missing)}"
            )
        arrays = {
            name: np.asarray(archive[name], dtype=float).copy()
            for name in required_arrays
        }
    for name, values in arrays.items():
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{arrays_path}:{name} must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{arrays_path}:{name} contains non-finite values")
    if (arrays["temporal_variance_profile"].size
            != arrays["mean_profile"].size):
        raise ValueError("LES mean and variance profile sizes differ")

    with target_metadata_path.open() as input_file:
        metadata = json.load(input_file)
    if metadata.get("status") != "accepted_dns_target":
        raise ValueError("DNS metadata does not describe an accepted target")
    target_directory = target_metadata_path.parent
    profile_path = target_directory / metadata["target_csv_filename"]
    spectrum_path = target_directory / metadata["target_spectrum_filename"]
    if file_sha256(profile_path) != metadata["target_csv_sha256"]:
        raise ValueError("DNS target profile hash does not match metadata")
    if file_sha256(spectrum_path) != metadata["target_spectrum_sha256"]:
        raise ValueError("DNS target spectrum hash does not match metadata")

    profile = read_csv_columns(profile_path, (
        "mean_velocity",
        "combined_standard_error",
        "temporal_variance",
        "temporal_variance_standard_error",
    ))
    spectrum = read_csv_columns(spectrum_path, (
        "wavenumber",
        "mean_energy",
        "combined_standard_error",
    ))
    if not np.array_equal(
            spectrum["wavenumber"],
            np.arange(spectrum["wavenumber"].size, dtype=float)):
        raise ValueError("DNS spectrum wavenumbers must be contiguous from zero")
    if int(metadata["cell_count"]) != profile["mean_velocity"].size:
        raise ValueError("DNS profile size does not match target metadata")
    les_cells = arrays["mean_profile"].size
    dns_mean = restrict_periodic_cell_averages(
        profile["mean_velocity"], les_cells
    )
    dns_mean_se = restrict_periodic_cell_averages(
        profile["combined_standard_error"], les_cells
    )
    dns_variance = restrict_periodic_cell_averages(
        profile["temporal_variance"], les_cells
    )
    dns_variance_se = restrict_periodic_cell_averages(
        profile["temporal_variance_standard_error"], les_cells
    )

    available_mode = min(
        arrays["mean_energy_spectrum"].size,
        spectrum["mean_energy"].size,
    ) - 1
    selected_maximum = (
        min(les_cells // 4, available_mode)
        if maximum_mode is None else maximum_mode
    )
    if selected_maximum < 1 or selected_maximum > available_mode:
        raise ValueError(
            f"maximum mode must lie between 1 and {available_mode}"
        )
    mode_slice = slice(1, selected_maximum + 1)
    les_spectrum = arrays["mean_energy_spectrum"][mode_slice]
    dns_spectrum = spectrum["mean_energy"][mode_slice]
    spectrum_difference = float(np.sum(np.abs(les_spectrum - dns_spectrum)))
    spectrum_scale = float(np.sum(np.abs(dns_spectrum)))

    profile_difference = arrays["mean_profile"] - dns_mean
    variance_difference = arrays["temporal_variance_profile"] - dns_variance
    return {
        "schema_version": 1,
        "les_analysis_summary": str(summary_path.resolve()),
        "les_analysis_arrays": str(arrays_path.resolve()),
        "dns_target_metadata": str(target_metadata_path.resolve()),
        "les_cell_count": int(les_cells),
        "dns_cell_count": int(profile["mean_velocity"].size),
        "field_restriction": (
            "contiguous periodic fine-cell averages are arithmetically "
            "restricted to each LES cell"
        ),
        "maximum_compared_mode": int(selected_maximum),
        "mean_profile": {
            "relative_l2_error": relative_l2(
                arrays["mean_profile"], dns_mean
            ),
            "absolute_l2_error": float(np.linalg.norm(profile_difference)),
            "dns_combined_standard_error_l2":
                float(np.linalg.norm(dns_mean_se)),
        },
        "temporal_variance_profile": {
            "relative_l2_error": relative_l2(
                arrays["temporal_variance_profile"], dns_variance
            ),
            "absolute_l2_error": float(np.linalg.norm(variance_difference)),
            "dns_combined_standard_error_l2":
                float(np.linalg.norm(dns_variance_se)),
        },
        "spectrum_relative_l1_error": (
            spectrum_difference / spectrum_scale
            if spectrum_scale > 0.0 else spectrum_difference
        ),
        "scalars": scalar_comparisons(
            summary["scalars"], metadata["scalar_statistics"]
        ),
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "analysis", type=Path,
        help="LES run directory or analysis_summary.json",
    )
    parser.add_argument("dns_target_metadata", type=Path)
    parser.add_argument(
        "--maximum-mode", type=int,
        help="largest common Fourier mode to compare",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("dns_comparison.json")
    )
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    try:
        result = compare(
            arguments.analysis,
            arguments.dns_target_metadata,
            arguments.maximum_mode,
        )
        with arguments.output.open("w") as output:
            json.dump(result, output, indent=2, allow_nan=False)
            output.write("\n")
        print(f"Wrote {arguments.output}")
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
