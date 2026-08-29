#!/usr/bin/env python3
"""Unit tests for plot_mean_spectrum.py input handling."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import plot_mean_spectrum  # noqa: E402


class NpzSpectrumTests(unittest.TestCase):
    def test_reads_analysis_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis_arrays.npz"
            np.savez(
                path,
                wavenumber=np.array([0, 1, 2]),
                mean_energy_spectrum=np.array([0.0, 0.5, 0.25]),
                compensated_energy_spectrum=np.array([0.0, 0.5, 0.75]),
            )

            actual = plot_mean_spectrum.read_spectrum(path)

        np.testing.assert_allclose(actual[0], [0.0, 1.0, 2.0])
        np.testing.assert_allclose(actual[1], [0.0, 0.5, 0.25])
        np.testing.assert_allclose(actual[2], [0.0, 0.5, 0.75])

    def test_rejects_missing_analysis_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis_arrays.npz"
            np.savez(
                path,
                wavenumber=np.array([0, 1]),
                mean_energy_spectrum=np.array([0.0, 0.5]),
            )

            with self.assertRaisesRegex(
                ValueError, "compensated_energy_spectrum"
            ):
                plot_mean_spectrum.read_spectrum(path)


if __name__ == "__main__":
    unittest.main()
