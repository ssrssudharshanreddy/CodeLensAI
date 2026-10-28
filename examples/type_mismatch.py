"""
Sample program with a TypeError.
"""

def generate_report(user_id: int, username: str, active: bool) -> str:
    """
    Format user details into a status string.
    Bug: Directly concatenates int user_id to string without str() conversion.
    """
    status = "Active" if active else "Inactive"
    # Bug: TypeError: can only concatenate str (not "int") to str
    return "User #" + user_id + ": " + username + " (" + status + ")"


if __name__ == "__main__":
    print(generate_report(42, "Alice", True))
