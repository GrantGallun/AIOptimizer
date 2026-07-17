"""Tests for treatment integrity.

The motivating failure: for five days the product treated 4 of 39 eligible turns while
`episodes inspect` reported `context_route: rate 1.0`. Coverage said healthy because the
field was present. These tests pin the distinction — a health check that cannot report a
dead optimizer is the thing we already had.
"""

import time
import unittest

from aioptimizer.health import assess


def turn(chars, route, *, injected=False, ts=None, error_type=None, route_reason=None):
    row = {"history_chars": chars, "route": route, "injected": injected}
    if ts is not None:
        row["ts"] = ts
    if error_type is not None:
        row["error_type"] = error_type
    if route_reason is not None:
        row["route_reason"] = route_reason
    return row


class TreatmentRateTests(unittest.TestCase):
    def test_healthy_when_eligible_turns_are_treated(self):
        now = time.time()
        rows = [turn(20_000, "attention", injected=True, ts=now - 60) for _ in range(10)]
        report = assess(rows, now=now)
        self.assertEqual(report["treatment_applied_rate"], 1.0)
        self.assertTrue(report["healthy"], report["findings"])

    def test_the_real_2026_07_17_outage_is_reported(self):
        now = time.time()
        rows = [turn(20_000, "attention", injected=True, ts=now - 5 * 24 * 3600) for _ in range(4)]
        rows += [turn(20_000, "raw") for _ in range(27)]
        rows += [turn(20_000, "error", error_type="URLError") for _ in range(8)]
        report = assess(rows, now=now)
        self.assertAlmostEqual(report["treatment_applied_rate"], 4 / 39, places=3)
        self.assertFalse(report["healthy"])
        codes = " ".join(report["findings"])
        self.assertIn("treatment_not_applied", codes)
        self.assertIn("sidecar_unreachable", codes)
        self.assertIn("stale_injection", codes)
        self.assertIn("unexplained_bypass", codes)

    def test_below_threshold_turns_cannot_inflate_the_rate(self):
        """The bug this guards: counting turns we could never help as successes would let
        a quiet week of short prompts hide a dead sidecar behind a healthy-looking rate."""
        now = time.time()
        rows = [turn(100, "below_threshold") for _ in range(500)]
        rows += [turn(20_000, "raw") for _ in range(10)]
        report = assess(rows, now=now)
        self.assertEqual(report["eligible_turns"], 10)
        self.assertEqual(report["treatment_applied_rate"], 0.0)
        self.assertFalse(report["healthy"])

    def test_no_eligible_turns_is_reported_as_uninformative_not_healthy(self):
        rows = [turn(100, "below_threshold") for _ in range(20)]
        report = assess(rows, now=time.time())
        self.assertIsNone(report["treatment_applied_rate"])
        self.assertFalse(report["healthy"])
        self.assertIn("no_eligible_turns", " ".join(report["findings"]))

    def test_a_recent_injection_is_not_stale(self):
        now = time.time()
        rows = [turn(20_000, "attention", injected=True, ts=now - 3600) for _ in range(5)]
        report = assess(rows, now=now)
        self.assertNotIn("stale_injection", " ".join(report["findings"]))

    def test_route_attention_without_injection_does_not_count_as_treated(self):
        now = time.time()
        rows = [turn(20_000, "attention", injected=False, ts=now) for _ in range(5)]
        report = assess(rows, now=now)
        self.assertEqual(report["treated_turns"], 0)
        self.assertFalse(report["healthy"])

    def test_explained_bypass_is_not_flagged_as_unexplained(self):
        now = time.time()
        rows = [turn(20_000, "attention", injected=True, ts=now) for _ in range(9)]
        rows.append(turn(20_000, "raw", route_reason="covered_by_recent_tail"))
        report = assess(rows, now=now)
        self.assertNotIn("unexplained_bypass", " ".join(report["findings"]))


if __name__ == "__main__":
    unittest.main()
