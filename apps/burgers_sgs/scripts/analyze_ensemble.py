#!/usr/bin/env python3
"""Combine independent-seed run summaries with small-sample uncertainty."""

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


T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def load_summary(path):
    if path.is_dir():
        path = path / "analysis_summary.json"
    with path.open() as input_file:
        return path, json.load(input_file)


def arrays_path_for_summary(path):
    if path.name.endswith("_summary.json"):
        return path.with_name(path.name.replace("_summary.json", "_arrays.npz"))
    return path.with_name("analysis_arrays.npz")


def load_profile_arrays(summary_path):
    arrays_path = arrays_path_for_summary(summary_path)
    with np.load(arrays_path) as arrays:
        required = ("x", "mean_profile")
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(
                f"missing required array(s) in {arrays_path}: "
                f"{', '.join(missing)}"
            )
        coordinates = np.asarray(arrays["x"], dtype=float).copy()
        mean_profile = np.asarray(arrays["mean_profile"], dtype=float).copy()
    if coordinates.ndim != 1 or mean_profile.ndim != 1:
        raise ValueError(f"profile arrays in {arrays_path} must be one-dimensional")
    if coordinates.size == 0:
        raise ValueError(f"profile arrays in {arrays_path} contain no data")
    if coordinates.size != mean_profile.size:
        raise ValueError(f"profile arrays in {arrays_path} have different lengths")
    if not (np.all(np.isfinite(coordinates))
            and np.all(np.isfinite(mean_profile))):
        raise ValueError(f"profile arrays in {arrays_path} contain non-finite values")
    return arrays_path, coordinates, mean_profile


def pointwise_profile_statistics(seed_profiles):
    seed_profiles = np.asarray(seed_profiles, dtype=float)
    if seed_profiles.ndim != 2 or seed_profiles.shape[0] < 2:
        raise ValueError("at least two one-dimensional seed profiles are required")
    sample_standard_deviation = np.std(seed_profiles, axis=0, ddof=1)
    return {
        "mean_profile": np.mean(seed_profiles, axis=0),
        "mean_profile_standard_error": (
            sample_standard_deviation / math.sqrt(seed_profiles.shape[0])
        ),
        "mean_profile_seed_sample_standard_deviation":
            sample_standard_deviation,
    }


def arrays_output_path(summary_output_path):
    if summary_output_path.name.endswith("_summary.json"):
        filename = summary_output_path.name.replace(
            "_summary.json", "_arrays.npz"
        )
    else:
        filename = f"{summary_output_path.stem}_arrays.npz"
    return summary_output_path.with_name(filename)


def hierarchical_bootstrap(entries, sample_count, random_seed):
    generator = np.random.default_rng(random_seed)
    seed_count = len(entries)
    estimates = np.empty(sample_count)
    for sample in range(sample_count):
        selected_seeds = generator.integers(0, seed_count, size=seed_count)
        run_estimates = []
        for seed_index in selected_seeds:
            blocks = np.asarray(entries[seed_index].get("block_means", []),
                                dtype=float)
            if blocks.size >= 2:
                selected_blocks = generator.choice(
                    blocks, size=blocks.size, replace=True
                )
                run_estimates.append(float(np.mean(selected_blocks)))
            else:
                run_estimates.append(float(entries[seed_index]["mean"]))
        estimates[sample] = np.mean(run_estimates)
    return np.quantile(estimates, [0.025, 0.975])


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path,
                        help="analysis_summary.json files or run directories")
    parser.add_argument("-o", "--output", type=Path,
                        default=Path("ensemble_summary.json"))
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    if len(arguments.summaries) < 2:
        parser.error("at least two independent-seed summaries are required")
    if arguments.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    try:
        loaded = [load_summary(path) for path in arguments.summaries]
        documents = [document for _, document in loaded]
        loaded_profiles = [
            load_profile_arrays(path) for path, _ in loaded
        ]
        reference_window = documents[0]["window"]
        reference_block = documents[0]["block_duration"]
        scalar_names = set(documents[0]["scalars"])
        reference_coordinates = loaded_profiles[0][1]
        seeds = []
        for (path, document), (_, coordinates, _) in zip(
                loaded, loaded_profiles):
            if document["window"] != reference_window:
                raise ValueError(f"analysis window differs in {path}")
            if not math.isclose(document["block_duration"], reference_block,
                                rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError(f"block duration differs in {path}")
            if set(document["scalars"]) != scalar_names:
                raise ValueError(f"scalar fields differ in {path}")
            if (coordinates.shape != reference_coordinates.shape
                    or not np.allclose(
                        coordinates, reference_coordinates,
                        rtol=1e-12, atol=1e-14
                    )):
                raise ValueError(f"profile coordinates differ in {path}")
            seeds.append(document["seed"])
        if len(set(seeds)) != len(seeds):
            raise ValueError("ensemble summaries must have distinct seeds")

        scalar_results = {}
        seed_count = len(documents)
        degrees_of_freedom = seed_count - 1
        multiplier = T_975.get(degrees_of_freedom, 1.96)
        for scalar_index, name in enumerate(sorted(scalar_names)):
            entries = [document["scalars"][name] for document in documents]
            values = np.asarray([entry["mean"] for entry in entries], dtype=float)
            sample_standard_deviation = float(np.std(values, ddof=1))
            standard_error = sample_standard_deviation / math.sqrt(seed_count)
            bootstrap_interval = hierarchical_bootstrap(
                entries, arguments.bootstrap_samples,
                arguments.bootstrap_seed + scalar_index
            )
            scalar_results[name] = {
                "mean": float(np.mean(values)),
                "seed_means": values.tolist(),
                "seed_sample_standard_deviation": sample_standard_deviation,
                "seed_standard_error": standard_error,
                "student_t_degrees_of_freedom": degrees_of_freedom,
                "student_t_95_confidence_interval": [
                    float(np.mean(values) - multiplier * standard_error),
                    float(np.mean(values) + multiplier * standard_error),
                ],
                "hierarchical_block_bootstrap_95_confidence_interval":
                    bootstrap_interval.tolist(),
            }

        seed_profiles = np.stack([
            mean_profile for _, _, mean_profile in loaded_profiles
        ])
        profile_results = pointwise_profile_statistics(seed_profiles)
        profile_arrays_path = arrays_output_path(arguments.output)

        output = {
            "schema_version": 1,
            "confidence_level": 0.95,
            "seed_count": seed_count,
            "seeds": seeds,
            "window": reference_window,
            "block_duration": reference_block,
            "source_summaries": [str(path.resolve()) for path, _ in loaded],
            "source_arrays": [
                str(path.resolve()) for path, _, _ in loaded_profiles
            ],
            "scalars": scalar_results,
            "profiles": {
                "arrays": str(profile_arrays_path.resolve()),
                "standard_error_basis": "between_seed_temporal_means",
            },
        }
        with arguments.output.open("w") as output_file:
            json.dump(output, output_file, indent=2, allow_nan=False)
            output_file.write("\n")
        np.savez_compressed(
            profile_arrays_path,
            x=reference_coordinates,
            seed_mean_profiles=seed_profiles,
            seed_count=np.asarray(seed_count),
            mean_profile_standard_error_kind=np.asarray("between_seed"),
            **profile_results,
        )
        print(f"Wrote {arguments.output}")
        print(f"Wrote {profile_arrays_path}")
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
