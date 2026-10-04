#!/usr/bin/env python3
"""Tests for portable DNS-target comparison."""

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import compare_dns_target  # noqa: E402


class DnsTargetComparisonTests(unittest.TestCase):
    def test_compares_analyzed_les_with_verified_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            les = root / "les"
            target = root / "target"
            les.mkdir()
            target.mkdir()

            summary = {
                "scalars": {
                    "kinetic_energy": {"mean": 1.1, "standard_error": 0.1},
                    "sgs_dissipation": {"mean": 0.2,
                                        "standard_error": 0.01},
                }
            }
            (les / "analysis_summary.json").write_text(json.dumps(summary))
            np.savez_compressed(
                les / "analysis_arrays.npz",
                mean_profile=np.array([1.5, 3.5]),
                temporal_variance_profile=np.array([0.2, 0.4]),
                mean_energy_spectrum=np.array([0.0, 0.5]),
            )

            profile_path = target / "dns_target.csv"
            with profile_path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow([
                    "mean_velocity", "combined_standard_error",
                    "temporal_variance", "temporal_variance_standard_error",
                ])
                for row in (
                    (1.0, 0.1, 0.1, 0.01),
                    (2.0, 0.1, 0.3, 0.01),
                    (3.0, 0.1, 0.3, 0.01),
                    (4.0, 0.1, 0.5, 0.01),
                ):
                    writer.writerow(row)
            spectrum_path = target / "dns_spectrum.csv"
            with spectrum_path.open("w", newline="") as output:
                writer = csv.writer(output)
                writer.writerow([
                    "wavenumber", "mean_energy", "combined_standard_error",
                ])
                writer.writerow((0, 0.0, 0.0))
                writer.writerow((1, 0.5, 0.01))
            metadata_path = target / "dns_target_metadata.json"
            metadata = {
                "status": "accepted_dns_target",
                "target_csv_filename": profile_path.name,
                "target_csv_sha256": compare_dns_target.file_sha256(
                    profile_path
                ),
                "target_spectrum_filename": spectrum_path.name,
                "target_spectrum_sha256": compare_dns_target.file_sha256(
                    spectrum_path
                ),
                "cell_count": 4,
                "scalar_statistics": {
                    "kinetic_energy": {
                        "mean": 1.0,
                        "combined_standard_error": 0.05,
                    },
                    "sgs_dissipation": {
                        "mean": 0.0,
                        "combined_standard_error": 0.0,
                    },
                },
            }
            metadata_path.write_text(json.dumps(metadata))

            result = compare_dns_target.compare(les, metadata_path, 1)

            self.assertEqual(result["les_cell_count"], 2)
            self.assertEqual(result["dns_cell_count"], 4)
            self.assertAlmostEqual(
                result["mean_profile"]["relative_l2_error"], 0.0
            )
            self.assertAlmostEqual(
                result["temporal_variance_profile"]["relative_l2_error"],
                0.0,
            )
            self.assertAlmostEqual(result["spectrum_relative_l1_error"], 0.0)
            self.assertAlmostEqual(
                result["scalars"]["kinetic_energy"]["difference"], 0.1
            )
            self.assertIsNone(
                result["scalars"]["sgs_dissipation"]["relative_difference"]
            )

    def test_rejects_nondivisible_restriction(self):
        with self.assertRaises(ValueError):
            compare_dns_target.restrict_periodic_cell_averages(
                np.arange(5, dtype=float), 2
            )


if __name__ == "__main__":
    unittest.main()
