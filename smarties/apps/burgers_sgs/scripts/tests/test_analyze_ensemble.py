#!/usr/bin/env python3
"""Unit tests for analyze_ensemble.py profile aggregation."""

import contextlib
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import analyze_ensemble  # noqa: E402


class ProfileStatisticsTests(unittest.TestCase):
    def test_computes_pointwise_mean_and_between_seed_standard_error(self):
        profiles = np.array([
            [1.0, 3.0],
            [3.0, 7.0],
            [5.0, 11.0],
        ])

        actual = analyze_ensemble.pointwise_profile_statistics(profiles)

        np.testing.assert_allclose(actual["mean_profile"], [3.0, 7.0])
        np.testing.assert_allclose(
            actual["mean_profile_seed_sample_standard_deviation"], [2.0, 4.0]
        )
        np.testing.assert_allclose(
            actual["mean_profile_standard_error"],
            [2.0 / math.sqrt(3.0), 4.0 / math.sqrt(3.0)],
        )

    def test_derives_arrays_filename_from_summary_output(self):
        self.assertEqual(
            analyze_ensemble.arrays_output_path(
                Path("runs/case/ensemble_summary.json")
            ),
            Path("runs/case/ensemble_arrays.npz"),
        )


class EnsembleOutputTests(unittest.TestCase):
    def test_writes_compatible_profile_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_paths = []
            coordinates = np.array([0.25, 0.75])
            profiles = ([1.0, 3.0], [5.0, 7.0])
            for seed, profile in zip((101, 202), profiles):
                run_path = root / f"seed_{seed}"
                run_path.mkdir()
                summary = {
                    "seed": seed,
                    "window": {"start": 1.0, "end": 2.0, "duration": 1.0},
                    "block_duration": 0.5,
                    "scalars": {
                        "kinetic_energy": {
                            "mean": float(seed),
                            "block_means": [float(seed), float(seed + 1)],
                        }
                    },
                }
                with (run_path / "analysis_summary.json").open("w") as output:
                    json.dump(summary, output)
                np.savez(
                    run_path / "analysis_arrays.npz",
                    x=coordinates,
                    mean_profile=np.asarray(profile),
                )
                run_paths.append(run_path)

            summary_output = root / "ensemble_summary.json"
            argv = [
                "analyze_ensemble.py",
                *(str(path) for path in run_paths),
                "--bootstrap-samples", "10",
                "-o", str(summary_output),
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                    io.StringIO()):
                analyze_ensemble.main()

            arrays_output = root / "ensemble_arrays.npz"
            self.assertTrue(summary_output.exists())
            self.assertTrue(arrays_output.exists())
            with np.load(arrays_output) as arrays:
                np.testing.assert_allclose(arrays["x"], coordinates)
                np.testing.assert_allclose(arrays["mean_profile"], [3.0, 5.0])
                np.testing.assert_allclose(
                    arrays["mean_profile_standard_error"], [2.0, 2.0]
                )
                self.assertEqual(
                    arrays["mean_profile_standard_error_kind"].item(),
                    "between_seed",
                )
                self.assertEqual(arrays["seed_count"].item(), 2)


if __name__ == "__main__":
    unittest.main()
