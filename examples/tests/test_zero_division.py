"""
Unit tests for calculate_average.
"""

import unittest
from examples.zero_division import calculate_average


class TestCalculateAverage(unittest.TestCase):

    def test_non_empty_list(self):
        self.assertEqual(calculate_average([10.0, 20.0, 30.0]), 20.0)

    def test_empty_list_returns_zero(self):
        # Expected behavior: returns 0.0 or handles empty gracefully
        self.assertEqual(calculate_average([]), 0.0)


if __name__ == "__main__":
    unittest.main()
