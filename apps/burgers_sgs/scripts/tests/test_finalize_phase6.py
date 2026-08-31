#!/usr/bin/env python3
"""Integration tests for the Phase 6 DNS-reference finalizer."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import finalize_phase6  # noqa: E402


def configuration(cell_count, seed):
    return {
        "grid": {"x_begin": 0.0, "x_end": 1.0, "cell_count": cell_count},
        "initial_condition": {"type": "sinusoidal"},
        "viscosity": {"molecular": 0.01},
        "numerical_method": {"reconstruction": "muscl"},
        "forcing": {"type": "composite", "clock_interval": 0.1},
        "closure": {"type": "no_closure"},
        "time_integration": {
            "integrator": "ssp_rk3",
            "initial_time": 0.0,
            "final_time": 4.0,
            "advective_cfl": 0.4,
            "diffusive_cfl": 0.2,
            "maximum_steps": 1000,
        },
        "output": {"directory": "."},
        "random": {"seed": seed},
    }


def write_analysis(root, name, cell_count, seed, profile, perturbation=0.0):
    directory = root / name
    directory.mkdir()
    metadata_path = directory / "burgers_run_metadata.json"
    metadata = {
        "schema_version": 6,
        "build": {"source_revision": "test"},
        "configuration": configuration(cell_count, seed),
        "result": {"status": "completed", "final_time": 4.0},
    }
    metadata_path.write_text(json.dumps(metadata))

    profile = np.asarray(profile, dtype=float) + perturbation
    block_offsets = np.array([-0.1, 0.1, -0.1, 0.1])[:, None]
    profile_blocks = profile[None, :] + block_offsets
    variance_blocks = np.full((4, cell_count), 0.2)
    spectrum_size = cell_count // 2 + 1
    spectrum = np.zeros(spectrum_size)
    spectrum[1] = 0.5 + perturbation
    spectrum_blocks = np.repeat(spectrum[None, :], 4, axis=0)
    summary_path = directory / "analysis_summary.json"
    summary = {
        "schema_version": 1,
        "source_metadata": str(metadata_path),
        "seed": seed,
        "window": {"start": 0.0, "end": 4.0, "duration": 4.0},
        "block_duration": 1.0,
        "scalars": {
            name: {
                "mean": value + perturbation,
                "block_means": [value - 0.01, value + 0.01,
                                value - 0.01, value + 0.01],
            }
            for name, value in {
                "kinetic_energy": 1.0,
                "spatial_variance": 0.5,
                "molecular_dissipation": 1.0,
                "interval_numerical_dissipation_rate": 0.001,
                "total_power": 1.001,
            }.items()
        },
        "energy_budget": {
            "numerical_to_molecular_dissipation_ratio": 0.001,
        },
        "profiles": {
            "profile_sample_count": 40,
            "profile_block_count": 4,
            "mean_gradient_square": 1.0 + perturbation,
        },
    }
    summary_path.write_text(json.dumps(summary))
    np.savez_compressed(
        directory / "analysis_arrays.npz",
        x=(np.arange(cell_count) + 0.5) / cell_count,
        mean_profile=np.mean(profile_blocks, axis=0),
        mean_profile_standard_error=np.full(cell_count, 0.05),
        temporal_variance_profile=np.full(cell_count, 0.2),
        temporal_variance_profile_standard_error=np.full(cell_count, 0.01),
        wavenumber=np.arange(spectrum_size),
        mean_energy_spectrum=spectrum,
        mean_energy_spectrum_standard_error=np.full(spectrum_size, 0.01),
        block_start_time=np.arange(4, dtype=float),
        block_end_time=np.arange(1, 5, dtype=float),
        mean_profile_block_means=profile_blocks,
        temporal_variance_profile_block_means=variance_blocks,
        mean_energy_spectrum_block_means=spectrum_blocks,
    )
    return summary_path


class FinalizationTests(unittest.TestCase):
    def test_passing_study_exports_report_and_portable_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            coarse_profile = np.array([1.0, 2.0, 1.0, 0.5])
            fine_profile = np.repeat(coarse_profile, 2)
            candidate = write_analysis(
                root, "candidate", 4, 5489, coarse_profile
            )
            reference = write_analysis(
                root, "reference", 8, 5489, fine_profile, perturbation=1e-5
            )
            target_one = write_analysis(
                root, "target_one", 4, 101, coarse_profile
            )
            target_two = write_analysis(
                root, "target_two", 4, 202, coarse_profile, perturbation=1e-4
            )
            study_path = root / "study.json"
            output = root / "finalized"
            study = {
                "schema_version": 1,
                "inputs": {
                    "grid_candidate_analysis": str(candidate),
                    "grid_reference_analysis": str(reference),
                    "target_analyses": [str(target_one), str(target_two)],
                },
                "design": {
                    "candidate_resolution": 4,
                    "reference_resolution": 8,
                    "paired_grid_seed": 5489,
                    "target_seeds": [101, 202],
                    "spinup_end": 0.0,
                    "grid_window": [0.0, 4.0],
                    "sampling_windows": {
                        "early": [0.0, 2.0],
                        "late": [2.0, 4.0],
                        "target": [0.0, 4.0],
                    },
                    "block_duration": 1.0,
                    "confidence_level": 0.95,
                    "bootstrap_samples": 200,
                    "bootstrap_seed": 77,
                    "spectrum_comparison": {
                        "minimum_mode": 1,
                        "maximum_mode": 1,
                    },
                },
                "acceptance": {
                    "mean_profile_relative_l2_grid": 0.01,
                    "variance_profile_relative_l2_grid": 0.01,
                    "kinetic_energy_relative_grid": 0.01,
                    "molecular_dissipation_relative_grid": 0.01,
                    "mean_gradient_square_relative_grid": 0.01,
                    "spectrum_relative_l1_grid": 0.01,
                    "numerical_to_molecular_dissipation": 0.01,
                    "energy_fraction_above_spectrum_band": 0.01,
                    "mean_profile_relative_l2_sampling": 0.5,
                    "minimum_complete_grid_blocks": 2,
                    "minimum_complete_blocks_per_window": 2,
                    "minimum_target_seed_count": 2,
                },
                "output": {
                    "directory": str(output),
                    "report_filename": "phase6_report.json",
                    "target_csv_filename": "dns_target.csv",
                    "target_spectrum_filename": "dns_spectrum.csv",
                    "target_metadata_filename": "dns_target_metadata.json",
                },
            }
            study_path.write_text(json.dumps(study))

            passed, report_path = finalize_phase6.finalize(study_path)

            self.assertTrue(passed)
            self.assertEqual(report_path, output / "phase6_report.json")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["overall_status"], "passed")
            self.assertTrue(all(
                check["passed"] for check in report["checks"].values()
            ))
            target_csv = output / "dns_target.csv"
            target_metadata = output / "dns_target_metadata.json"
            target_spectrum = output / "dns_spectrum.csv"
            self.assertTrue(target_csv.exists())
            self.assertTrue(target_metadata.exists())
            self.assertTrue(target_spectrum.exists())
            with target_csv.open(newline="") as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(len(rows), 4)
            self.assertIn("combined_standard_error", rows[0])
            metadata = json.loads(target_metadata.read_text())
            self.assertEqual(metadata["status"], "accepted_dns_target")
            self.assertEqual(metadata["seed_count"], 2)
            self.assertEqual(len(metadata["target_csv_sha256"]), 64)
            self.assertEqual(len(metadata["target_spectrum_sha256"]), 64)
            self.assertIn("kinetic_energy", metadata["scalar_statistics"])
            with target_spectrum.open(newline="") as input_file:
                spectrum_rows = list(csv.DictReader(input_file))
            self.assertEqual(len(spectrum_rows), 3)
            self.assertIn("combined_standard_error", spectrum_rows[0])

            failed_output = root / "failed"
            study["acceptance"]["mean_profile_relative_l2_grid"] = 0.0
            study["output"]["directory"] = str(failed_output)
            study_path.write_text(json.dumps(study))

            passed, failed_report_path = finalize_phase6.finalize(study_path)

            self.assertFalse(passed)
            failed_report = json.loads(failed_report_path.read_text())
            self.assertEqual(failed_report["overall_status"], "failed")
            self.assertFalse((failed_output / "dns_target.csv").exists())
            self.assertFalse(
                (failed_output / "dns_target_metadata.json").exists()
            )
            self.assertFalse((failed_output / "dns_spectrum.csv").exists())


if __name__ == "__main__":
    unittest.main()
