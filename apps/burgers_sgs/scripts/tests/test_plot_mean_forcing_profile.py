"""Check block statistics, warmup, and configuration precedence."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import plot_mean_forcing_profile as plot


class ForcingProfileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "burgers_rl_fields.csv"
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["decision_index", "start_time", "end_time", "x", "applied_action"])
            for step in range(26):
                # Warmup includes t=2. Partial tail starts at t=22.
                value = 1000 if step <= 2 or step >= 22 else (1 if step < 12 else 3)
                for x in (0.25, 0.75):
                    writer.writerow([step, step, step + 1, x, value * (1 if x < 0.5 else -1)])
        self.write_json("burgers_run_metadata.json", {
            "configuration": {"output": {"statistics_start_time": 2}}})

    def write_json(self, name, value):
        (self.path.parent / name).write_text(json.dumps(value))

    def test_fallback_warmup_partial_tail_and_sample_se(self):
        x, mean, se, count = plot.read_mean_forcing_profile(self.path)
        np.testing.assert_allclose(x, [0.25, 0.75])
        np.testing.assert_allclose(mean, [2, -2])
        np.testing.assert_allclose(se, [1, 1])
        self.assertEqual(count, 2)

    def test_summary_and_cli_precedence(self):
        self.write_json("analysis_summary.json", {
            "block_duration": 5, "window": {"start": 2, "end": 22}})
        self.assertEqual(plot.read_mean_forcing_profile(self.path)[3], 4)
        self.assertEqual(plot.read_mean_forcing_profile(self.path, 10)[3], 2)
        self.assertEqual(plot.read_mean_forcing_profile(self.path, 5, 7)[3], 3)

    def test_invalid_duration_and_insufficient_blocks(self):
        for duration in (0, -1, float("nan"), float("inf"), 100):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                plot.read_mean_forcing_profile(self.path, duration)

    def test_missing_warmup_requires_explicit_cutoff(self):
        self.write_json("burgers_run_metadata.json", {})
        with self.assertRaisesRegex(ValueError, "warmup cutoff unavailable"):
            plot.read_mean_forcing_profile(self.path)
        self.assertEqual(plot.read_mean_forcing_profile(self.path, statistics_start_time=2)[3], 2)


if __name__ == "__main__":
    unittest.main()
