"""
Unit tests for find_median.
"""

import unittest
from examples.off_by_one import find_median


class TestFindMedian(unittest.TestCase):

    def test_odd_length_list(self):
        # 3 elements: median is middle element 20
        self.assertEqual(find_median([10, 20, 30]), 20.0)

    def test_single_element(self):
        # 1 element: median is that element
        self.assertEqual(find_median([42]), 42.0)

    def test_five_elements(self):
        self.assertEqual(find_median([1, 3, 5, 7, 9]), 5.0)

    def test_even_length_list(self):
        # 4 elements: median is average of 2 and 3 -> 2.5
        self.assertEqual(find_median([1, 2, 3, 4]), 2.5)

    def test_empty_list_raises_value_error(self):
        with self.assertRaises(ValueError):
            find_median([])


if __name__ == "__main__":
    unittest.main()
