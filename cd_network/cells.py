import numpy as np
from coincidence_integral import coincidence_integral


def ei(excitatory_input: np.ndarray, inhibitory_inputs: np.ndarray, delta_s: float, fs: float) -> np.ndarray:
    """
    Calculates the excitatory-inhibitory interaction.

    Parameters:
        excitatory_input (np.ndarray): 1D array of excitatory inputs.
        inhibitory_inputs (np.ndarray): 1D or 2D array of inhibitory inputs.
        fs (float): sampling frequency.
        delta_s: coincidence integration duration in seconds.

    Returns:
        np.ndarray: Output after applying the excitatory-inhibitory interaction.
    """
    assert excitatory_input.ndim == 1, "Excitatory input must be a 1D array."
    assert inhibitory_inputs.ndim in [1, 2], "Inhibitory inputs must be either 1D or 2D array."

    if inhibitory_inputs.ndim == 1:
        inhibitory_inputs = inhibitory_inputs[np.newaxis, ...]

    assert len(excitatory_input) == inhibitory_inputs.shape[-1], \
        "Excitatory input length must match the size of inhibitory inputs along the last axis."

    output = excitatory_input * np.prod(1 - coincidence_integral(inhibitory_inputs, delta_s, fs), axis=0)
    return output


def ee(inputs: np.ndarray, delta_s: float, fs: float) -> np.ndarray:
    """
    Calculates the simple excitatory-excitatory interaction.

    Parameters:
        inputs (np.ndarray): 2D array of excitatory inputs.
        fs (float): sampling frequency.
        delta_s: coincidence integration duration in seconds.

    Returns:
        np.ndarray: Output after applying the excitatory-excitatory interaction.
    """
    assert inputs.ndim == 2, "Excitatory inputs must be a 2D array."
    num_inputs, samples = inputs.shape

    coincidence_integral_outputs = coincidence_integral(inputs, delta_s, fs)
    coincidence_prod = np.prod(coincidence_integral_outputs, axis=0)

    output = np.zeros(samples)
    for i in range(num_inputs):
        output += inputs[i] * coincidence_prod / coincidence_integral_outputs[i]
    return output
