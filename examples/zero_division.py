"""
Sample program with unhandled ZeroDivisionError.
"""

def calculate_average(scores: list[float]) -> float:
    """
    Calculate the average of scores.
    Bug: Does not handle empty scores list, causing ZeroDivisionError.
    """
    total = sum(scores)
    # Bug: crashes with ZeroDivisionError when scores is empty
    return total / len(scores)


if __name__ == "__main__":
    print("Average:", calculate_average([]))
