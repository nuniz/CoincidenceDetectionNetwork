def validate_inputs(inputs, expected_dims, var_name="input"):
    """
    Validates the dimensions of the input array against a set of expected dimensions.

    Parameters:
        inputs (np.ndarray): The array to validate.
        expected_dims (list or tuple): A list or tuple of integers specifying the acceptable dimensions.
        var_name (str, optional): The name of the variable to include in the error message. Defaults to "input".

    Returns:
        np.ndarray: The validated input array if it meets the criteria.

    Raises:
        ValueError: If the input array's dimensionality is not in the expected dimensions.
    """
    if inputs.ndim not in expected_dims:
        raise ValueError(
            f"{var_name} must be {' or '.join(map(str, expected_dims))}D array."
        )
    return inputs


def hashable_input(inputs):
    """
    Converts a 2D array into a hashable tuple format to allow it to be used as a dictionary key or stored in sets.

    Parameters:
        inputs (np.ndarray): The 2D input array to be converted.

    Returns:
        tuple: A tuple of tuples, where each inner tuple represents a row from the input array.
    """
    return tuple(map(tuple, inputs))
