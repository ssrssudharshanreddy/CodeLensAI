"""
Sample program with an off-by-one error for CodeLens AI demonstration.
"""

def find_median(sorted_numbers: list[int | float]) -> float:
    """
    Return the median value of a sorted list of numbers.

    For odd-length lists, returns the exact middle element.
    For even-length lists, returns the average of the two middle elements.

    Bug: The odd-length case indexes 'mid + 1' instead of 'mid',
    causing an IndexError or returning the wrong value.
    """
    n = len(sorted_numbers)
    if n == 0:
        raise ValueError("Cannot calculate median of an empty list.")

    mid = n // 2

    # Bug is here: mid + 1 is off-by-one (0-indexed median is sorted_numbers[mid])
    if n % 2 == 1:
        return float(sorted_numbers[mid + 1])
    else:
        return float(sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2.0


if __name__ == "__main__":
    nums = [10, 20, 30]
    # Expected: 20
    # Actual: returns 30 or fails
    print(f"Median of {nums} is {find_median(nums)}")
