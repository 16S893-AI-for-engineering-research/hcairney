#!/usr/bin/env python3
"""Unit tests for plot_input_distribution.py."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import plot_input_distribution  # noqa: E402


class InputFeatureTests(unittest.TestCase):
    def test_matches_periodic_centered_observation_feature(self):
        profiles = np.array([[0.0, 1.0, 0.0, -1.0]])

        actual = plot_input_distribution.input_features(
            profiles, cell_width=1.0, viscosity=2.0
        )

        np.testing.assert_allclose(
            actual, [[0.5, 0.0, -0.5, 0.0]]
        )

    def test_second_derivative_feature_uses_periodic_centered_stencil(self):
        profiles = np.array([[0.0, 1.0, 0.0, -1.0]])

        actual = plot_input_distribution.second_derivative_features(
            profiles, cell_width=2.0, viscosity=4.0
        )

        np.testing.assert_allclose(actual, [[0.0, -1.0, 0.0, 1.0]])

    def test_default_bin_width_depends_on_data(self):
        narrow = plot_input_distribution.default_bin_width(
            np.linspace(0.0, 1.0, 100)
        )
        wide = plot_input_distribution.default_bin_width(
            np.linspace(0.0, 10.0, 100)
        )

        self.assertAlmostEqual(wide, 10.0 * narrow)

    def test_custom_edges_have_requested_width_and_cover_data(self):
        edges = plot_input_distribution.bin_edges(
            np.array([0.12, 0.81]), 0.2
        )

        np.testing.assert_allclose(np.diff(edges), 0.2)
        self.assertLessEqual(edges[0], 0.12)
        self.assertGreaterEqual(edges[-1], 0.81)

    def test_rejects_impractically_small_bin_width(self):
        with self.assertRaisesRegex(ValueError, "more than 1,000,000 bins"):
            plot_input_distribution.bin_edges(np.array([0.0, 1.0]), 1e-7)


class RunInputTests(unittest.TestCase):
    def test_loads_a_complete_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            run_directory = Path(directory)
            metadata = {
                "configuration": {
                    "grid": {
                        "x_begin": 0.0,
                        "x_end": 4.0,
                        "cell_count": 4,
                    },
                    "viscosity": {"molecular": 2.0},
                }
            }
            (run_directory / "burgers_run_metadata.json").write_text(
                json.dumps(metadata)
            )
            (run_directory / "burgers_profiles.csv").write_text(
                "step,time,x,cell_average\n"
                "0,0,0.5,0\n0,0,1.5,1\n0,0,2.5,0\n0,0,3.5,-1\n"
                "1,1,0.5,0\n1,1,1.5,1\n1,1,2.5,0\n1,1,3.5,-1\n"
            )

            times, features, second_derivatives, cell_width, viscosity = (
                plot_input_distribution.load_run(run_directory)
            )

        np.testing.assert_allclose(times, [0.0, 1.0])
        np.testing.assert_allclose(
            features,
            np.tile([0.5, 0.0, -0.5, 0.0], (2, 1)),
        )
        np.testing.assert_allclose(
            second_derivatives,
            np.tile([0.0, -1.0, 0.0, 1.0], (2, 1)),
        )
        self.assertEqual(cell_width, 1.0)
        self.assertEqual(viscosity, 2.0)

    def test_default_output_is_inside_run_directory(self):
        with mock.patch.object(
            sys, "argv", ["plot_input_distribution.py", "runs/case"]
        ):
            _, arguments = plot_input_distribution.parse_arguments()

        self.assertIsNone(arguments.output)
        self.assertIsNone(arguments.second_derivative_bin_width)
        self.assertIsNone(arguments.second_derivative_output)


if __name__ == "__main__":
    unittest.main()
