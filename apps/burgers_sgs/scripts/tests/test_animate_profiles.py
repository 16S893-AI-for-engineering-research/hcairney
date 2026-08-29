#!/usr/bin/env python3
"""Unit tests for animate_profiles.py title formatting."""

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

import animate_profiles  # noqa: E402


class AnimationTitleTests(unittest.TestCase):
    def test_numeric_fields_have_fixed_width(self):
        frames = [
            (0, 0.0, [], []),
            (31, 0.3, [], []),
            (5177, 50.0, [], []),
        ]
        format_title = animate_profiles.make_title_formatter(frames)

        titles = [format_title(step, time) for step, time, _, _ in frames]

        self.assertEqual(titles[0], "Step 0000, t = 00.000")
        self.assertEqual(titles[1], "Step 0031, t = 00.300")
        self.assertEqual(titles[2], "Step 5177, t = 50.000")
        self.assertEqual(len({len(title) for title in titles}), 1)

    def test_time_width_accounts_for_a_negative_value(self):
        frames = [
            (0, -10.0, [], []),
            (1, 2.0, [], []),
        ]
        format_title = animate_profiles.make_title_formatter(frames)

        self.assertEqual(format_title(0, -10.0), "Step 0, t = -10.000")
        self.assertEqual(format_title(1, 2.0), "Step 1, t = 002.000")


if __name__ == "__main__":
    unittest.main()
