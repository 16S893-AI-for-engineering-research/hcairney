#!/usr/bin/env python3
"""Unit tests for dependency-light numerical parts of analyze_run.py."""

import math
import sys
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


if __name__ == "__main__":
    unittest.main()
