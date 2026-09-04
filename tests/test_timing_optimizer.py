import unittest

from voice_dub.timing_optimizer import choose_variant_indices, next_variant_index, timing_violation


class TimingOptimizerTests(unittest.TestCase):
    def test_violation_accepts_duration_within_gap_limit(self):
        self.assertEqual(timing_violation(4.85, 5.0), 0.0)

    def test_next_variant_moves_longer_only_after_measured_gap(self):
        self.assertEqual(next_variant_index([3.0, 4.7, 5.4], [0], 3.0, 5.0), 1)
        self.assertIsNone(next_variant_index([3.0, 4.7], [1], 4.85, 5.0))

    def test_next_variant_moves_shorter_after_measured_overlap(self):
        self.assertEqual(next_variant_index([4.5, 5.5], [1], 5.5, 5.0), 0)
    def test_keeps_first_variants_when_constraints_are_satisfied(self):
        self.assertEqual(
            choose_variant_indices([0, 5], [5, 10], [[4.5, 3.0], [4.0, 3.0]]),
            [0, 0],
        )

    def test_uses_shorter_previous_variant_to_remove_overlap(self):
        self.assertEqual(
            choose_variant_indices([0, 5], [5, 10], [[6.0, 4.0], [4.0]]),
            [1, 0],
        )

    def test_uses_longer_previous_variant_to_remove_artificial_gap(self):
        self.assertEqual(
            choose_variant_indices([0, 8], [8, 12], [[2.0, 7.0], [3.0]]),
            [1, 0],
        )

    def test_accepts_overlap_only_when_no_assignment_can_avoid_it(self):
        self.assertEqual(
            choose_variant_indices([0, 5], [5, 8], [[6.0], [4.0]]),
            [0, 0],
        )


if __name__ == "__main__":
    unittest.main()
