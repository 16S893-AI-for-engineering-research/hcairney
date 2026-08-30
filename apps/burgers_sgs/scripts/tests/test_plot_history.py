#!/usr/bin/env python3
"""Unit tests for plot_history.py input handling."""

import tempfile
import unittest
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import plot_history  # noqa: E402


HEADER = (
    "time,kinetic_energy,molecular_dissipation,deterministic_power,"
    "stochastic_power,manufactured_power,"
    "interval_numerical_dissipation_rate\n"
)


class HistoryTests(unittest.TestCase):
    def test_derives_energy_budget_columns_for_legacy_history(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "burgers_history.csv"
            path.write_text(
                HEADER
                + "0,2,0.2,0.4,0.1,0.0,0.05\n"
                + "1,2.25,0.2,0.4,0.1,0.0,0.05\n"
            )

            history = plot_history.read_history(path)

        self.assertEqual(history["total_power"], [0.5, 0.5])
        self.assertEqual(history["interval_energy_change_rate"], [0.0, 0.25])
        self.assertAlmostEqual(history["interval_budget_residual_rate"][1], 0.0)

    def test_rejects_nonincreasing_times(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "burgers_history.csv"
            path.write_text(
                HEADER
                + "1,2,0.2,0.4,0.1,0.0,0.05\n"
                + "1,2.25,0.2,0.4,0.1,0.0,0.05\n"
            )

            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                plot_history.read_history(path)


if __name__ == "__main__":
    unittest.main()
