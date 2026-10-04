#!/usr/bin/env python3
"""Unit tests for dependency-light numerical parts of analyze_run.py."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import analyze_run  # noqa: E402


class SpectrumTests(unittest.TestCase):
    def test_parseval_for_even_and_odd_cell_counts(self):
        generator = np.random.default_rng(771)
        for cell_count in (63, 64, 255, 256):
            values = generator.normal(size=(3, cell_count))
            domain_length = 2.0 * math.pi
            spectra = analyze_run.one_sided_energy_spectra(
                values, domain_length
            )
            expected = 0.5 * domain_length * np.mean(values ** 2, axis=1)
            np.testing.assert_allclose(
                np.sum(spectra, axis=1), expected,
                rtol=5e-14, atol=5e-14
            )

    def test_single_sine_mode_normalization(self):
        cell_count = 128
        amplitude = 0.3
        mode = 7
        locations = 2.0 * math.pi * np.arange(cell_count) / cell_count
        values = amplitude * np.sin(mode * locations)
        spectrum = analyze_run.one_sided_energy_spectra(
            values[None, :], 2.0 * math.pi
        )[0]
        expected = 2.0 * math.pi * amplitude ** 2 / 4.0
        self.assertAlmostEqual(spectrum[mode], expected, places=14)
        self.assertAlmostEqual(np.sum(spectrum), expected, places=14)


class SamplingTests(unittest.TestCase):
    def test_old_history_is_read_as_zero_sgs(self):
        header = (
            "time,mean,kinetic_energy,spatial_variance,"
            "molecular_dissipation,deterministic_power,stochastic_power,"
            "manufactured_power,interval_numerical_dissipation_rate\n"
        )
        rows = (
            "0,0,1,1,0.2,0.1,0.2,0,0.05\n"
            "1,0,1.05,1.05,0.2,0.1,0.2,0,0.05\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            path.write_text(header + rows)
            history = analyze_run.read_history(path)
        np.testing.assert_array_equal(history["sgs_dissipation"], [0.0, 0.0])
        np.testing.assert_array_equal(history["prescribed_power"], [0.0, 0.0])
        np.testing.assert_array_equal(history["mean_coefficient"], [0.0, 0.0])
        np.testing.assert_allclose(
            history["interval_budget_residual_rate"],
            history["total_power"] - history["molecular_dissipation"]
            - history["interval_numerical_dissipation_rate"]
            - history["interval_energy_change_rate"],
        )

    def test_constant_series_has_zero_autocorrelation_time(self):
        times = np.arange(20, dtype=float) * 0.1
        values = np.ones(20)
        self.assertEqual(
            analyze_run.integrated_autocorrelation_time(times, values), 0.0
        )

    def test_block_means_use_complete_blocks(self):
        times = np.arange(10, dtype=float)
        values = np.arange(10, dtype=float)
        blocks = analyze_run.snapshot_block_means(
            times, values, 0.0, 10.0, 5.0
        )
        np.testing.assert_allclose(blocks, [2.0, 7.0])

    def test_profile_statistics_retain_complete_block_arrays(self):
        times = np.array([0.5, 1.5, 2.5, 3.5])
        coordinates = np.array([0.25, 0.75])
        profiles = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])

        summary, arrays = analyze_run.profile_statistics(
            times, coordinates, profiles, 0.0, 4.0, 2.0, 1.0
        )

        self.assertEqual(summary["profile_block_count"], 2)
        np.testing.assert_allclose(arrays["block_start_time"], [0.0, 2.0])
        np.testing.assert_allclose(arrays["block_end_time"], [2.0, 4.0])
        np.testing.assert_allclose(
            arrays["mean_profile_block_means"], [[2.0, 3.0], [6.0, 7.0]]
        )
        self.assertEqual(
            arrays["temporal_variance_profile_block_means"].shape, (2, 2)
        )
        self.assertEqual(
            arrays["mean_energy_spectrum_block_means"].shape, (2, 2)
        )


if __name__ == "__main__":
    unittest.main()
