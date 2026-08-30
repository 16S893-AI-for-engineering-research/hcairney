#!/usr/bin/env python3
"""Unit tests for plot_mean_spectrum.py input handling."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import plot_mean_spectrum  # noqa: E402


class OutputPathTests(unittest.TestCase):
    def test_default_output_filename_is_spectrum_png(self):
        with mock.patch.object(
            sys, "argv", ["plot_mean_spectrum.py", "runs/case/data.npz"]
        ):
            _, arguments = plot_mean_spectrum.parse_arguments()

        self.assertEqual(arguments.output, Path("spectrum.png"))

    def test_places_output_beside_csv_input(self):
        actual = plot_mean_spectrum.output_path_for_spectrum(
            Path("runs/case/burgers_mean_spectrum.csv"),
            Path("mean_spectrum.png"),
        )

        self.assertEqual(actual, Path("runs/case/mean_spectrum.png"))

    def test_uses_only_output_filename_for_npz_input(self):
        actual = plot_mean_spectrum.output_path_for_spectrum(
            Path("runs/case/analysis_arrays.npz"),
            Path("figures/mean_spectrum.png"),
        )

        self.assertEqual(actual, Path("runs/case/mean_spectrum.png"))


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
