#!/usr/bin/env python3
"""Unit tests for plot_mean_velocity_profile.py input handling."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import plot_mean_velocity_profile  # noqa: E402


class OutputPathTests(unittest.TestCase):
    def test_default_output_filename(self):
        with mock.patch.object(
            sys,
            "argv",
            ["plot_mean_velocity_profile.py", "runs/case/analysis_arrays.npz"],
        ):
            _, arguments = plot_mean_velocity_profile.parse_arguments()

        self.assertEqual(arguments.output, Path("mean_velocity_profile.png"))

    def test_parses_optional_y_limits(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "plot_mean_velocity_profile.py",
                "runs/case/analysis_arrays.npz",
                "--ylim",
                "-0.5",
                "0.5",
            ],
        ):
            _, arguments = plot_mean_velocity_profile.parse_arguments()

        self.assertEqual(arguments.ylim, [-0.5, 0.5])

    def test_parses_optional_x_limits(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "plot_mean_velocity_profile.py",
                "runs/case/analysis_arrays.npz",
                "--xlim",
                "0.25",
                "0.75",
            ],
        ):
            _, arguments = plot_mean_velocity_profile.parse_arguments()

        self.assertEqual(arguments.xlim, [0.25, 0.75])

    def test_places_output_beside_npz_input(self):
        actual = plot_mean_velocity_profile.output_path_for_profile(
            Path("runs/case/analysis_arrays.npz"),
            Path("figures/profile.png"),
        )

        self.assertEqual(actual, Path("runs/case/profile.png"))


class ProfileInputTests(unittest.TestCase):
    def test_reference_cli_and_plot(self):
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "run"
            run.mkdir()
            path = run / "analysis_arrays.npz"
            np.savez(
                path, x=[0.25, 0.75], mean_profile=[1.0, 2.0],
                mean_profile_standard_error=[0.1, 0.2],
            )
            reference = root / "dns_target.csv"
            reference.write_text("x,mean_velocity\n0.1,3.0\n0.5,4.0\n0.9,5.0\n")
            explicit = root / "custom.csv"
            explicit.write_text("x,mean_velocity\n0.2,6.0\n0.8,7.0\n")
            for flags, expected_x, expected_mean in (
                ([], None, None),
                (["--reference"], [0.1, 0.5, 0.9], [3.0, 4.0, 5.0]),
                (["--reference", str(explicit)], [0.2, 0.8], [6.0, 7.0]),
            ):
                with self.subTest(flags=flags), mock.patch.object(
                    sys, "argv", ["plot_mean_velocity_profile.py", str(path), *flags]
                ):
                    _, args = plot_mean_velocity_profile.parse_arguments()
                    figure = plot_mean_velocity_profile.make_figure(
                        args.analysis_file, reference=args.reference
                    )
                    axis = figure.axes[0]
                    self.assertEqual(len(axis.collections), 1)
                    self.assertEqual(len(axis.lines), 2 if flags else 1)
                    if flags:
                        np.testing.assert_allclose(axis.lines[1].get_xdata(), expected_x)
                        np.testing.assert_allclose(axis.lines[1].get_ydata(), expected_mean)
                    plt.close(figure)

            reference.unlink()
            for requested in (True, explicit.with_name("missing.csv")):
                with self.subTest(reference=requested), self.assertRaisesRegex(
                    FileNotFoundError, "DNS reference file not found:.*\\.csv"
                ):
                    plot_mean_velocity_profile.make_figure(path, reference=requested)

    def test_rejects_invalid_reference_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dns_target.csv"
            for contents, message in (
                ("x,other\n0,1\n", "must contain x and mean_velocity"),
                ("x,mean_velocity\n", "contains no data"),
                ("x,mean_velocity\n0,nan\n", "non-finite"),
                ("x,mean_velocity\n0,bad\n", "invalid data"),
            ):
                with self.subTest(contents=contents):
                    path.write_text(contents)
                    with self.assertRaisesRegex(ValueError, message):
                        plot_mean_velocity_profile.read_reference_profile(path)

    def test_applies_requested_x_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis_arrays.npz"
            np.savez(
                path,
                x=np.array([0.25, 0.75]),
                mean_profile=np.array([-0.25, 0.25]),
                mean_profile_standard_error=np.array([0.1, 0.1]),
            )

            figure = plot_mean_velocity_profile.make_figure(
                path, xlim=(0.2, 0.8)
            )

        self.assertEqual(figure.axes[0].get_xlim(), (0.2, 0.8))

    def test_applies_requested_y_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis_arrays.npz"
            np.savez(
                path,
                x=np.array([0.25, 0.75]),
                mean_profile=np.array([-0.25, 0.25]),
                mean_profile_standard_error=np.array([0.1, 0.1]),
            )

            figure = plot_mean_velocity_profile.make_figure(
                path, ylim=(-0.5, 0.5)
            )

        self.assertEqual(figure.axes[0].get_ylim(), (-0.5, 0.5))

    def test_reads_run_analysis_as_temporal_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis_arrays.npz"
            np.savez(
                path,
                x=np.array([0.25, 0.75]),
                mean_profile=np.array([1.0, 2.0]),
                mean_profile_standard_error=np.array([0.1, 0.2]),
            )

            actual = plot_mean_velocity_profile.read_mean_velocity_profile(path)

        np.testing.assert_allclose(actual[0], [0.25, 0.75])
        np.testing.assert_allclose(actual[1], [1.0, 2.0])
        np.testing.assert_allclose(actual[2], [0.1, 0.2])
        self.assertEqual(actual[3], "temporal")
        self.assertIsNone(actual[4])

    def test_reads_ensemble_analysis_as_between_seed_uncertainty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ensemble_arrays.npz"
            np.savez(
                path,
                x=np.array([0.25, 0.75]),
                mean_profile=np.array([1.0, 2.0]),
                mean_profile_standard_error=np.array([0.1, 0.2]),
                mean_profile_standard_error_kind=np.asarray("between_seed"),
                seed_count=np.asarray(4),
            )

            actual = plot_mean_velocity_profile.read_mean_velocity_profile(path)

        self.assertEqual(actual[3], "between_seed")
        self.assertEqual(actual[4], 4)
        title, error_label = plot_mean_velocity_profile.profile_labels(
            actual[3], actual[4]
        )
        self.assertIn("4 seeds", title)
        self.assertIn("Between-seed", error_label)

    def test_rejects_missing_standard_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "analysis_arrays.npz"
            np.savez(
                path,
                x=np.array([0.25, 0.75]),
                mean_profile=np.array([1.0, 2.0]),
            )

            with self.assertRaisesRegex(
                ValueError, "mean_profile_standard_error"
            ):
                plot_mean_velocity_profile.read_mean_velocity_profile(path)


if __name__ == "__main__":
    unittest.main()
