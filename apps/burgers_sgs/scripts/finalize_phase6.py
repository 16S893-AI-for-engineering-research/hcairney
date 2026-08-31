#!/usr/bin/env python3
"""Validate Phase 6 evidence and export a portable offline DNS target."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError as error:
    raise SystemExit(
        "NumPy is required; install requirements-analysis.txt"
    ) from error


def finite_float(value, context):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def resolve_path(base, value):
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def arrays_path_for_summary(path):
    if path.name.endswith("_summary.json"):
        return path.with_name(path.name.replace("_summary.json", "_arrays.npz"))
    raise ValueError(f"analysis summary filename must end in _summary.json: {path}")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, document):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as output:
        json.dump(document, output, indent=2, allow_nan=False)
        output.write("\n")
    os.replace(temporary, path)


def load_json(path):
    with path.open() as input_file:
        return json.load(input_file)


def source_metadata_path(summary_path, summary):
    declared = summary.get("source_metadata")
    if declared:
        declared_path = Path(declared)
        if declared_path.exists():
            return declared_path
    fallback = summary_path.parent / "burgers_run_metadata.json"
    if fallback.exists():
        return fallback
    raise ValueError(f"cannot locate source run metadata for {summary_path}")


def load_analysis(path):
    summary = load_json(path)
    arrays_path = arrays_path_for_summary(path)
    required = (
        "x",
        "mean_profile",
        "mean_profile_standard_error",
        "temporal_variance_profile",
        "temporal_variance_profile_standard_error",
        "wavenumber",
        "mean_energy_spectrum",
        "mean_energy_spectrum_standard_error",
        "block_start_time",
        "block_end_time",
        "mean_profile_block_means",
        "temporal_variance_profile_block_means",
        "mean_energy_spectrum_block_means",
    )
    with np.load(arrays_path) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(
                f"{arrays_path} is missing Phase 6 block array(s): "
                f"{', '.join(missing)}; rerun analyze_run.py"
            )
        arrays = {name: np.asarray(archive[name]).copy() for name in required}
    for name, values in arrays.items():
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{arrays_path}:{name} contains non-finite values")

    metadata_path = source_metadata_path(path, summary)
    metadata = load_json(metadata_path)
    if metadata.get("result", {}).get("status") != "completed":
        raise ValueError(f"source run is not completed: {metadata_path}")
    analysis_time = min(path.stat().st_mtime, arrays_path.stat().st_mtime)
    if metadata_path.stat().st_mtime > analysis_time:
        raise ValueError(
            f"analysis predates its final run metadata: {path}; rerun analyze_run.py"
        )
    output_config = metadata["configuration"].get("output", {})
    profile_filename = output_config.get("profile_history_filename")
    if profile_filename:
        profile_path = path.parent / profile_filename
        if profile_path.exists() and profile_path.stat().st_mtime > analysis_time:
            raise ValueError(
                f"analysis predates its profile history: {path}; the run may be "
                "active or the analysis may be stale"
            )
    configured_cells = int(metadata["configuration"]["grid"]["cell_count"])
    if arrays["x"].ndim != 1 or arrays["x"].size != configured_cells:
        raise ValueError(
            f"analysis cell count does not match run metadata: {path}"
        )
    block_count = arrays["block_start_time"].size
    if arrays["block_end_time"].shape != (block_count,):
        raise ValueError(f"invalid block-time arrays in {arrays_path}")
    for name in (
        "mean_profile_block_means",
        "temporal_variance_profile_block_means",
    ):
        if arrays[name].shape != (block_count, configured_cells):
            raise ValueError(f"invalid {name} shape in {arrays_path}")
    if arrays["mean_energy_spectrum_block_means"].shape != (
            block_count, arrays["mean_energy_spectrum"].size):
        raise ValueError(
            f"invalid mean_energy_spectrum_block_means shape in {arrays_path}"
        )
    if float(metadata["result"]["final_time"]) + 1e-12 < float(
            summary["window"]["end"]):
        raise ValueError(f"analysis extends past the completed run: {path}")
    return {
        "summary_path": path,
        "arrays_path": arrays_path,
        "metadata_path": metadata_path,
        "summary": summary,
        "arrays": arrays,
        "metadata": metadata,
    }


def compatible_core(configuration, include_seed):
    time = configuration["time_integration"]
    document = {
        "domain": {
            "x_begin": configuration["grid"]["x_begin"],
            "x_end": configuration["grid"]["x_end"],
        },
        "initial_condition": configuration["initial_condition"],
        "viscosity": configuration["viscosity"],
        "numerical_method": configuration["numerical_method"],
        "forcing": configuration["forcing"],
        "closure": configuration["closure"],
        "time_integration": {
            "integrator": time["integrator"],
            "initial_time": time["initial_time"],
            "advective_cfl": time["advective_cfl"],
            "diffusive_cfl": time["diffusive_cfl"],
        },
    }
    if include_seed:
        document["random"] = configuration["random"]
    return document


def require_compatible(reference, candidate, context, include_seed):
    left = compatible_core(reference["metadata"]["configuration"], include_seed)
    right = compatible_core(candidate["metadata"]["configuration"], include_seed)
    if left != right:
        raise ValueError(f"scientific configurations differ for {context}")


def window_tuple(summary):
    window = summary["window"]
    return (float(window["start"]), float(window["end"]))


def require_window(analysis, expected, context):
    actual = window_tuple(analysis["summary"])
    if not (math.isclose(actual[0], expected[0], rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(actual[1], expected[1], rel_tol=0.0,
                             abs_tol=1e-12)):
        raise ValueError(f"{context} analysis window is {actual}, expected {expected}")


def restrict_cell_averages(values, target_size):
    source_size = values.shape[-1]
    if source_size == target_size:
        return values
    if source_size % target_size != 0:
        raise ValueError(
            f"cannot conservatively restrict {source_size} cells to {target_size}"
        )
    ratio = source_size // target_size
    return np.mean(values.reshape(*values.shape[:-1], target_size, ratio), axis=-1)


def relative_l2(left, right):
    denominator = float(np.linalg.norm(right))
    difference = float(np.linalg.norm(left - right))
    return difference / denominator if denominator > 0.0 else difference


def relative_scalar_difference(left, right):
    scale = abs(float(right))
    difference = abs(float(left) - float(right))
    return difference / scale if scale > 0.0 else difference


def check_entry(value, threshold, comparison="maximum"):
    value = float(value)
    threshold = float(threshold)
    if comparison == "maximum":
        passed = value <= threshold
    elif comparison == "minimum":
        passed = value >= threshold
    else:
        raise ValueError(f"unknown comparison {comparison}")
    return {
        "value": value,
        "threshold": threshold,
        "comparison": comparison,
        "passed": bool(passed),
    }


def grid_metrics(candidate, reference, minimum_mode, maximum_mode):
    candidate_arrays = candidate["arrays"]
    reference_arrays = reference["arrays"]
    candidate_cells = candidate_arrays["x"].size
    reference_cells = reference_arrays["x"].size
    if reference_cells <= candidate_cells:
        raise ValueError("grid reference must be finer than the DNS candidate")
    reference_mean = restrict_cell_averages(
        reference_arrays["mean_profile"], candidate_cells
    )
    reference_variance = restrict_cell_averages(
        reference_arrays["temporal_variance_profile"], candidate_cells
    )
    available_mode = min(
        candidate_arrays["mean_energy_spectrum"].size,
        reference_arrays["mean_energy_spectrum"].size,
    ) - 1
    if (minimum_mode < 1 or maximum_mode < minimum_mode
            or maximum_mode > available_mode):
        raise ValueError(
            "spectrum comparison modes must satisfy "
            f"1 <= minimum <= maximum <= {available_mode}"
        )
    selected = slice(minimum_mode, maximum_mode + 1)
    candidate_spectrum = candidate_arrays["mean_energy_spectrum"]
    reference_spectrum = reference_arrays["mean_energy_spectrum"]
    spectrum_denominator = float(np.sum(np.abs(reference_spectrum[selected])))
    spectrum_difference = float(np.sum(np.abs(
        candidate_spectrum[selected] - reference_spectrum[selected]
    )))
    nonzero_energy = float(np.sum(candidate_spectrum[1:]))
    energy_above_band = float(np.sum(candidate_spectrum[maximum_mode + 1:]))

    candidate_summary = candidate["summary"]
    reference_summary = reference["summary"]
    candidate_scalars = candidate_summary["scalars"]
    reference_scalars = reference_summary["scalars"]
    metrics = {
        "mean_profile_relative_l2": relative_l2(
            candidate_arrays["mean_profile"], reference_mean
        ),
        "temporal_variance_profile_relative_l2": relative_l2(
            candidate_arrays["temporal_variance_profile"], reference_variance
        ),
        "kinetic_energy_relative_difference": relative_scalar_difference(
            candidate_scalars["kinetic_energy"]["mean"],
            reference_scalars["kinetic_energy"]["mean"],
        ),
        "molecular_dissipation_relative_difference": relative_scalar_difference(
            candidate_scalars["molecular_dissipation"]["mean"],
            reference_scalars["molecular_dissipation"]["mean"],
        ),
        "mean_gradient_square_relative_difference": relative_scalar_difference(
            candidate_summary["profiles"]["mean_gradient_square"],
            reference_summary["profiles"]["mean_gradient_square"],
        ),
        "spectrum_relative_l1": (
            spectrum_difference / spectrum_denominator
            if spectrum_denominator > 0.0 else spectrum_difference
        ),
        "numerical_to_molecular_dissipation": candidate_summary[
            "energy_budget"
        ]["numerical_to_molecular_dissipation_ratio"],
        "energy_fraction_above_spectrum_band": (
            energy_above_band / nonzero_energy if nonzero_energy > 0.0 else 0.0
        ),
    }

    # The paired estimate uses matching physical-time blocks and therefore
    # removes most forcing-realization variance from the discretization check.
    candidate_blocks = candidate_arrays["mean_profile_block_means"]
    reference_blocks = reference_arrays["mean_profile_block_means"]
    if (candidate_blocks.shape[0] == reference_blocks.shape[0]
            and np.allclose(candidate_arrays["block_start_time"],
                            reference_arrays["block_start_time"],
                            rtol=0.0, atol=1e-12)
            and np.allclose(candidate_arrays["block_end_time"],
                            reference_arrays["block_end_time"],
                            rtol=0.0, atol=1e-12)):
        restricted_reference_blocks = restrict_cell_averages(
            reference_blocks, candidate_cells
        )
        paired_differences = candidate_blocks - restricted_reference_blocks
        if paired_differences.shape[0] >= 2:
            paired_se = np.std(paired_differences, axis=0, ddof=1) / math.sqrt(
                paired_differences.shape[0]
            )
            denominator = float(np.linalg.norm(reference_mean))
            metrics["paired_profile_difference_standard_error_relative_l2"] = (
                float(np.linalg.norm(paired_se)) / denominator
                if denominator > 0.0 else float(np.linalg.norm(paired_se))
            )
            metrics["paired_block_count"] = int(paired_differences.shape[0])
    return metrics


def block_mask(analysis, window):
    starts = analysis["arrays"]["block_start_time"]
    ends = analysis["arrays"]["block_end_time"]
    tolerance = 1e-12
    return ((starts >= window[0] - tolerance)
            & (ends <= window[1] + tolerance))


def sampling_statistics(targets, early_window, late_window, confidence,
                        bootstrap_samples, bootstrap_seed):
    reference_x = targets[0]["arrays"]["x"]
    early_blocks = []
    late_blocks = []
    all_blocks = []
    variance_profiles = []
    variance_standard_errors = []
    early_counts = []
    late_counts = []
    for target in targets:
        arrays = target["arrays"]
        if (arrays["x"].shape != reference_x.shape
                or not np.allclose(arrays["x"], reference_x,
                                   rtol=1e-12, atol=1e-14)):
            raise ValueError("target analyses use different spatial grids")
        early_mask = block_mask(target, early_window)
        late_mask = block_mask(target, late_window)
        blocks = arrays["mean_profile_block_means"]
        if not np.any(early_mask) or not np.any(late_mask):
            raise ValueError("sampling windows contain no complete profile blocks")
        early_blocks.append(blocks[early_mask])
        late_blocks.append(blocks[late_mask])
        all_blocks.append(blocks[early_mask | late_mask])
        early_counts.append(int(np.count_nonzero(early_mask)))
        late_counts.append(int(np.count_nonzero(late_mask)))
        variance_profiles.append(arrays["temporal_variance_profile"])
        variance_standard_errors.append(
            arrays["temporal_variance_profile_standard_error"]
        )

    seed_count = len(targets)
    # Preserve the exact snapshot-weighted means produced by analyze_run.py for
    # the exported target. Complete-block means are used for uncertainty and
    # early/late comparisons, where equal block weighting is intentional.
    seed_profiles = np.stack([
        target["arrays"]["mean_profile"] for target in targets
    ])
    seed_block_profiles = np.stack([
        np.mean(blocks, axis=0) for blocks in all_blocks
    ])
    target_profile = np.mean(seed_profiles, axis=0)
    target_block_profile = np.mean(seed_block_profiles, axis=0)
    early_profile = np.mean(
        np.stack([np.mean(blocks, axis=0) for blocks in early_blocks]), axis=0
    )
    late_profile = np.mean(
        np.stack([np.mean(blocks, axis=0) for blocks in late_blocks]), axis=0
    )
    observed_difference = relative_l2(late_profile, early_profile)

    temporal_variance_of_mean = np.zeros_like(target_profile)
    for blocks in all_blocks:
        if blocks.shape[0] >= 2:
            temporal_variance_of_mean += (
                np.var(blocks, axis=0, ddof=1) / blocks.shape[0]
            ) / (seed_count ** 2)
    within_temporal_se = np.sqrt(temporal_variance_of_mean)
    if seed_count >= 2:
        between_seed_se = np.std(seed_profiles, axis=0, ddof=1) / math.sqrt(
            seed_count
        )
    else:
        between_seed_se = np.zeros_like(target_profile)
    variance_profiles = np.stack(variance_profiles)
    target_variance = np.mean(variance_profiles, axis=0)
    within_variance_se = np.sqrt(np.sum(
        np.stack(variance_standard_errors) ** 2, axis=0
    )) / seed_count
    if seed_count >= 2:
        between_variance_se = np.std(
            variance_profiles, axis=0, ddof=1
        ) / math.sqrt(seed_count)
    else:
        between_variance_se = np.zeros_like(target_variance)
    combined_variance_se = np.sqrt(
        within_variance_se ** 2 + between_variance_se ** 2
    )

    generator = np.random.default_rng(bootstrap_seed)
    null_differences = np.empty(bootstrap_samples)
    sampled_differences = np.empty(bootstrap_samples)
    full_uncertainties = np.empty(bootstrap_samples)
    bootstrap_full_sum = np.zeros_like(target_profile)
    bootstrap_full_square_sum = np.zeros_like(target_profile)
    profile_norm = float(np.linalg.norm(target_profile))
    scale = profile_norm if profile_norm > 0.0 else 1.0
    early_means = [np.mean(blocks, axis=0) for blocks in early_blocks]
    late_means = [np.mean(blocks, axis=0) for blocks in late_blocks]
    for sample in range(bootstrap_samples):
        selected_seeds = generator.integers(0, seed_count, size=seed_count)
        sampled_early = []
        sampled_late = []
        sampled_full = []
        null_early = []
        null_late = []
        for seed_index in selected_seeds:
            early = early_blocks[seed_index]
            late = late_blocks[seed_index]
            full = all_blocks[seed_index]
            early_draw = early[generator.integers(0, early.shape[0],
                                                  size=early.shape[0])]
            late_draw = late[generator.integers(0, late.shape[0],
                                                size=late.shape[0])]
            full_draw = full[generator.integers(0, full.shape[0],
                                                size=full.shape[0])]
            sampled_early.append(np.mean(early_draw, axis=0))
            sampled_late.append(np.mean(late_draw, axis=0))
            sampled_full.append(np.mean(full_draw, axis=0))
            null_early.append(np.mean(early_draw - early_means[seed_index], axis=0))
            null_late.append(np.mean(late_draw - late_means[seed_index], axis=0))
        bootstrap_early = np.mean(np.stack(sampled_early), axis=0)
        bootstrap_late = np.mean(np.stack(sampled_late), axis=0)
        bootstrap_full = np.mean(np.stack(sampled_full), axis=0)
        bootstrap_full_sum += bootstrap_full
        bootstrap_full_square_sum += bootstrap_full ** 2
        sampled_differences[sample] = (
            np.linalg.norm(bootstrap_late - bootstrap_early) / scale
        )
        null_differences[sample] = np.linalg.norm(
            np.mean(np.stack(null_late), axis=0)
            - np.mean(np.stack(null_early), axis=0)
        ) / scale
        full_uncertainties[sample] = (
            np.linalg.norm(bootstrap_full - target_block_profile) / scale
        )
    bootstrap_variance = (
        bootstrap_full_square_sum
        - bootstrap_full_sum ** 2 / bootstrap_samples
    ) / (bootstrap_samples - 1)
    combined_se = np.sqrt(np.maximum(bootstrap_variance, 0.0))
    quantile = confidence
    return {
        "x": reference_x,
        "target_profile": target_profile,
        "within_temporal_standard_error": within_temporal_se,
        "between_seed_standard_error": between_seed_se,
        "combined_standard_error": combined_se,
        "temporal_variance_profile": target_variance,
        "temporal_variance_standard_error": combined_variance_se,
        "early_block_counts": early_counts,
        "late_block_counts": late_counts,
        "observed_early_late_relative_l2": observed_difference,
        "null_difference_uncertainty": float(np.quantile(null_differences,
                                                          quantile)),
        "early_late_relative_l2_upper_bound": float(np.quantile(
            sampled_differences, quantile
        )),
        "target_profile_relative_l2_uncertainty": float(np.quantile(
            full_uncertainties, quantile
        )),
    }


def provenance_entry(analysis):
    return {
        "analysis_summary_sha256": file_sha256(analysis["summary_path"]),
        "analysis_arrays_sha256": file_sha256(analysis["arrays_path"]),
        "run_metadata_sha256": file_sha256(analysis["metadata_path"]),
        "run_metadata": analysis["metadata"],
        "analysis": analysis["summary"],
    }


def source_identity(analysis):
    configuration = analysis["metadata"]["configuration"]
    return {
        "analysis_summary": str(analysis["summary_path"]),
        "analysis_arrays": str(analysis["arrays_path"]),
        "run_metadata": str(analysis["metadata_path"]),
        "analysis_summary_sha256": file_sha256(analysis["summary_path"]),
        "analysis_arrays_sha256": file_sha256(analysis["arrays_path"]),
        "run_metadata_sha256": file_sha256(analysis["metadata_path"]),
        "cell_count": int(configuration["grid"]["cell_count"]),
        "seed": int(configuration["random"]["seed"]),
        "source_revision": analysis["metadata"].get("build", {}).get(
            "source_revision", "unknown"
        ),
    }


def write_target_csv(path, sampling):
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow([
            "x",
            "mean_velocity",
            "within_temporal_standard_error",
            "between_seed_standard_error",
            "combined_standard_error",
            "temporal_variance",
            "temporal_variance_standard_error",
        ])
        columns = (
            sampling["x"],
            sampling["target_profile"],
            sampling["within_temporal_standard_error"],
            sampling["between_seed_standard_error"],
            sampling["combined_standard_error"],
            sampling["temporal_variance_profile"],
            sampling["temporal_variance_standard_error"],
        )
        for row in zip(*columns):
            writer.writerow([format(float(value), ".17g") for value in row])
    os.replace(temporary, path)


def finalize(study_path, output_override=None):
    study = load_json(study_path)
    if study.get("schema_version") != 1:
        raise ValueError("Phase 6 study schema_version must be 1")
    base = study_path.parent
    inputs = study["inputs"]
    candidate = load_analysis(resolve_path(base, inputs["grid_candidate_analysis"]))
    reference = load_analysis(resolve_path(base, inputs["grid_reference_analysis"]))
    target_paths = inputs["target_analyses"]
    if not target_paths:
        raise ValueError("at least one target analysis is required")
    targets = [load_analysis(resolve_path(base, path)) for path in target_paths]

    design = study["design"]
    grid_window = tuple(map(float, design["grid_window"]))
    early_window = tuple(map(float, design["sampling_windows"]["early"]))
    late_window = tuple(map(float, design["sampling_windows"]["late"]))
    target_window = tuple(map(float, design["sampling_windows"]["target"]))
    if not (early_window[0] == target_window[0]
            and early_window[1] == late_window[0]
            and late_window[1] == target_window[1]):
        raise ValueError("early and late windows must partition the target window")
    spinup_end = float(design["spinup_end"])
    if not (math.isclose(spinup_end, grid_window[0], rel_tol=0.0,
                         abs_tol=1e-12)
            and math.isclose(spinup_end, target_window[0], rel_tol=0.0,
                             abs_tol=1e-12)):
        raise ValueError("spinup_end must equal the start of analysis windows")
    require_window(candidate, grid_window, "grid candidate")
    require_window(reference, grid_window, "grid reference")
    require_compatible(candidate, reference, "grid refinement", include_seed=True)
    expected_candidate = int(design["candidate_resolution"])
    expected_reference = int(design["reference_resolution"])
    if int(candidate["metadata"]["configuration"]["grid"]["cell_count"]) != expected_candidate:
        raise ValueError("grid candidate metadata has the wrong cell count")
    if int(reference["metadata"]["configuration"]["grid"]["cell_count"]) != expected_reference:
        raise ValueError("grid reference metadata has the wrong cell count")
    if int(candidate["metadata"]["configuration"]["random"]["seed"]) != int(
            design["paired_grid_seed"]):
        raise ValueError("grid candidate does not use paired_grid_seed")
    if candidate["metadata"]["configuration"]["closure"]["type"] != "no_closure":
        raise ValueError("Phase 6 DNS runs must use no_closure")
    for label, analysis in (("grid candidate", candidate),
                            ("grid reference", reference)):
        if not math.isclose(float(analysis["summary"]["block_duration"]),
                            float(design["block_duration"]),
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{label} has the wrong block duration")

    for index, target in enumerate(targets):
        require_window(target, target_window, f"target {index}")
        require_compatible(candidate, target, f"target {index}", include_seed=False)
        if int(target["metadata"]["configuration"]["grid"]["cell_count"]) != expected_candidate:
            raise ValueError(f"target {index} has the wrong cell count")
        if not np.allclose(target["arrays"]["x"], candidate["arrays"]["x"],
                           rtol=1e-12, atol=1e-14):
            raise ValueError(f"target {index} does not use the candidate grid")
        if not math.isclose(float(target["summary"]["block_duration"]),
                            float(design["block_duration"]),
                            rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"target {index} has the wrong block duration")
    seeds = [int(target["metadata"]["configuration"]["random"]["seed"])
             for target in targets]
    if len(set(seeds)) != len(seeds):
        raise ValueError("target analyses must use distinct seeds")
    expected_seeds = design.get("target_seeds")
    if expected_seeds is not None and seeds != [int(seed) for seed in expected_seeds]:
        raise ValueError("target analyses do not match target_seeds in order")

    minimum_mode = int(design["spectrum_comparison"]["minimum_mode"])
    maximum_mode = int(design["spectrum_comparison"]["maximum_mode"])
    grid = grid_metrics(candidate, reference, minimum_mode, maximum_mode)
    confidence = finite_float(design["confidence_level"], "confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie between zero and one")
    bootstrap_samples = int(design["bootstrap_samples"])
    if bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be at least two")
    sampling = sampling_statistics(
        targets,
        early_window,
        late_window,
        confidence,
        bootstrap_samples,
        int(design["bootstrap_seed"]),
    )

    acceptance = study["acceptance"]
    checks = {
        "grid.mean_profile_relative_l2": check_entry(
            grid["mean_profile_relative_l2"],
            acceptance["mean_profile_relative_l2_grid"],
        ),
        "grid.temporal_variance_profile_relative_l2": check_entry(
            grid["temporal_variance_profile_relative_l2"],
            acceptance["variance_profile_relative_l2_grid"],
        ),
        "grid.kinetic_energy_relative_difference": check_entry(
            grid["kinetic_energy_relative_difference"],
            acceptance["kinetic_energy_relative_grid"],
        ),
        "grid.molecular_dissipation_relative_difference": check_entry(
            grid["molecular_dissipation_relative_difference"],
            acceptance["molecular_dissipation_relative_grid"],
        ),
        "grid.mean_gradient_square_relative_difference": check_entry(
            grid["mean_gradient_square_relative_difference"],
            acceptance["mean_gradient_square_relative_grid"],
        ),
        "grid.spectrum_relative_l1": check_entry(
            grid["spectrum_relative_l1"],
            acceptance["spectrum_relative_l1_grid"],
        ),
        "grid.numerical_to_molecular_dissipation": check_entry(
            grid["numerical_to_molecular_dissipation"],
            acceptance["numerical_to_molecular_dissipation"],
        ),
        "grid.energy_fraction_above_spectrum_band": check_entry(
            grid["energy_fraction_above_spectrum_band"],
            acceptance["energy_fraction_above_spectrum_band"],
        ),
        "grid.minimum_paired_blocks": check_entry(
            grid.get("paired_block_count", 0),
            acceptance["minimum_complete_grid_blocks"],
            comparison="minimum",
        ),
        "sampling.minimum_early_blocks": check_entry(
            min(sampling["early_block_counts"]),
            acceptance["minimum_complete_blocks_per_window"],
            comparison="minimum",
        ),
        "sampling.minimum_late_blocks": check_entry(
            min(sampling["late_block_counts"]),
            acceptance["minimum_complete_blocks_per_window"],
            comparison="minimum",
        ),
        "sampling.early_late_consistent_with_uncertainty": {
            "value": sampling["observed_early_late_relative_l2"],
            "threshold": sampling["null_difference_uncertainty"],
            "comparison": "maximum",
            "passed": bool(
                sampling["observed_early_late_relative_l2"]
                <= sampling["null_difference_uncertainty"]
            ),
        },
        "sampling.early_late_relative_l2_upper_bound": check_entry(
            sampling["early_late_relative_l2_upper_bound"],
            acceptance["mean_profile_relative_l2_sampling"],
        ),
        "sampling.minimum_target_blocks": check_entry(
            min(np.asarray(sampling["early_block_counts"])
                + np.asarray(sampling["late_block_counts"])),
            2 * int(acceptance["minimum_complete_blocks_per_window"]),
            comparison="minimum",
        ),
        "sampling.minimum_target_seed_count": check_entry(
            len(seeds),
            acceptance["minimum_target_seed_count"],
            comparison="minimum",
        ),
    }
    overall_passed = all(entry["passed"] for entry in checks.values())

    output_config = study["output"]
    output_directory = (
        Path(output_override).resolve() if output_override is not None
        else resolve_path(base, output_config["directory"])
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / output_config["report_filename"]
    target_csv_path = output_directory / output_config["target_csv_filename"]
    target_metadata_path = output_directory / output_config[
        "target_metadata_filename"
    ]
    report = {
        "schema_version": 1,
        "phase": 6,
        "overall_status": "passed" if overall_passed else "failed",
        "study_definition": str(study_path.resolve()),
        "design": design,
        "acceptance": acceptance,
        "seeds": seeds,
        "sources": {
            "grid_candidate": source_identity(candidate),
            "grid_reference": source_identity(reference),
            "target_runs": [source_identity(target) for target in targets],
        },
        "checks": checks,
        "grid_metrics": grid,
        "sampling_metrics": {
            "early_block_counts": sampling["early_block_counts"],
            "late_block_counts": sampling["late_block_counts"],
            "observed_early_late_relative_l2":
                sampling["observed_early_late_relative_l2"],
            "null_difference_uncertainty":
                sampling["null_difference_uncertainty"],
            "early_late_relative_l2_upper_bound":
                sampling["early_late_relative_l2_upper_bound"],
            "target_profile_relative_l2_uncertainty":
                sampling["target_profile_relative_l2_uncertainty"],
        },
        "outputs": {
            "target_files_written": overall_passed,
            "target_csv": str(target_csv_path),
            "target_metadata": str(target_metadata_path),
        },
    }
    atomic_json(report_path, report)

    if overall_passed:
        write_target_csv(target_csv_path, sampling)
        target_metadata = {
            "schema_version": 1,
            "phase": 6,
            "status": "accepted_dns_target",
            "target_csv_filename": target_csv_path.name,
            "target_csv_sha256": file_sha256(target_csv_path),
            "cell_count": int(sampling["x"].size),
            "domain": candidate["metadata"]["configuration"]["grid"],
            "seed_count": len(seeds),
            "seeds": seeds,
            "sampling_window": {
                "start": target_window[0],
                "end": target_window[1],
                "duration": target_window[1] - target_window[0],
            },
            "block_duration": float(design["block_duration"]),
            "complete_block_count_per_seed": [
                early + late for early, late in zip(
                    sampling["early_block_counts"], sampling["late_block_counts"]
                )
            ],
            "profile_sample_count_per_seed": [
                int(target["summary"]["profiles"]["profile_sample_count"])
                for target in targets
            ],
            "confidence_level": confidence,
            "uncertainty_method": (
                "temporal complete-block estimates with hierarchical seed/block "
                "bootstrap; CSV reports temporal, between-seed, and hierarchical-"
                "bootstrap combined pointwise standard errors"
            ),
            "acceptance_report": report,
            "provenance": {
                "grid_candidate": provenance_entry(candidate),
                "grid_reference": provenance_entry(reference),
                "target_runs": [provenance_entry(target) for target in targets],
            },
        }
        atomic_json(target_metadata_path, target_metadata)
    return overall_passed, report_path


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path,
                        help="versioned Phase 6 study-definition JSON")
    parser.add_argument(
        "--output-directory", type=Path,
        help="override the output directory declared by the study",
    )
    return parser, parser.parse_args()


def main():
    parser, arguments = parse_arguments()
    try:
        passed, report_path = finalize(
            arguments.study.resolve(), arguments.output_directory
        )
    except (KeyError, OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {report_path}")
    if passed:
        print("Phase 6 acceptance checks passed; DNS target exported.")
        return 0
    print("Phase 6 acceptance checks failed; DNS target was not exported.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
