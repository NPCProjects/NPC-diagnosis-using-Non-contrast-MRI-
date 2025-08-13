def str2bool(v):
    """
    Convert a string value to a boolean.

    Args:
        v (str): The string to be converted. Expected values are
                 'yes', 'true', 't', 'y', '1' for True and
                 'no', 'false', 'f', 'n', '0' for False.

    Returns:
        bool: The corresponding boolean value.

    Raises:
        ValueError: If the input is not a recognized string.
    """
    truthy_values = ('yes', 'true', 't', 'y', '1')
    falsy_values = ('no', 'false', 'f', 'n', '0')

    normalized_input = v.strip().lower()

    if normalized_input in truthy_values:
        return True
    elif normalized_input in falsy_values:
        return False
    else:
        raise ValueError(f"Unsupported value encountered: '{v}'")
