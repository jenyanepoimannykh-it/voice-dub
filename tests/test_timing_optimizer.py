import unittest

from voice_dub.timing_optimizer import best_start_index, next_variant_index, timing_violation


class TimingOptimizerTests(unittest.TestCase):
    def test_violation_accepts_duration_within_gap_limit(self):
        self.assertEqual(timing_violation(4.85, 5.0), 0.0)

    def test_next_variant_moves_longer_only_after_measured_gap(self):
        self.assertEqual(next_variant_index([3.0, 4.7, 5.4], [0], 3.0, 5.0), 1)
        self.assertIsNone(next_variant_index([3.0, 4.7], [1], 4.85, 5.0))

    def test_next_variant_moves_shorter_after_measured_overlap(self):
        self.assertEqual(next_variant_index([4.5, 5.5], [1], 5.5, 5.0), 0)
    def test_pause_before_the_next_cue_is_not_something_to_fill(self):
        # Source speaks 5s then pauses 1s. A 4.9s take is on target, not short,
        # and a 5.8s take runs into the pause without overlapping the next line.
        self.assertEqual(timing_violation(4.9, 5.0, 0.2, 6.0), 0.0)
        self.assertEqual(timing_violation(5.8, 5.0, 0.2, 6.0), 0.0)
        self.assertAlmostEqual(timing_violation(6.4, 5.0, 0.2, 6.0), 0.4)
        self.assertAlmostEqual(timing_violation(4.0, 5.0, 0.2, 6.0), 0.8)

    def test_start_index_picks_the_first_wording_predicted_to_fit(self):
        self.assertEqual(best_start_index([2.0, 4.9, 5.1, 7.0], 5.0), 1)

    def test_start_index_falls_back_to_the_closest_when_none_fit(self):
        self.assertEqual(best_start_index([1.0, 2.0, 9.0], 5.0), 1)

    def test_start_index_prefers_earlier_wording_on_a_tie(self):
        self.assertEqual(best_start_index([4.9, 4.9, 5.0], 5.0), 0)


if __name__ == "__main__":
    unittest.main()
