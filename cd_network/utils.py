def validate_inputs(inputs, expected_dims, var_name="input"):
    if inputs.ndim not in expected_dims:
        raise ValueError(f"{var_name} must be {' or '.join(map(str, expected_dims))}D array.")
    return inputs


def hashable_input(inputs):
    return tuple(map(tuple, inputs))
