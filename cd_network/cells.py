import numpy as np

from coincidence_integral import coincidence_integral


def ei(excitatory_input: np.ndarray, inhibitory_inputs: np.ndarray, delta_s: float, fs: float) -> np.ndarray:
    """
    The general EI cell spikes whenever the excitatory input spikes and in the preceding ∆ seconds none of the
    inhibitory inputs spike.

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


def _ee(inputs: np.ndarray, delta_s: float, fs: float) -> np.ndarray:
    """
    Calculates the excitatory-excitatory interaction.

    Parameters:
        inputs (np.ndarray): 2D array of excitatory inputs.
        fs (float): Sampling frequency.
        delta_s: Coincidence integration duration in seconds.

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


def ee(inputs, min_inputs: int, delta_s: float, fs: float) -> np.ndarray:
    """
    A general EE cell (depicted in Figure 2-6) has N excitatory inputs.
    It generates a spike whenever at least min_input of its inputs spikes during an interval ∆.

    Parameters
        inputs (np.ndarray): 2D array of excitatory inputs.
        min_inputs (int): Minimum number of inputs that must spike.
        delta_s (float): Coincidence integration duration in seconds.
        fs (float): Sampling frequency.

    Returns:
        np.ndarray: Output after applying the excitatory-excitatory interaction with the specified criteria.

    Raises:
        ValueError: If inputs are not 2D or if min_inputs exceeds the number of inputs.
    """
    if inputs.ndim != 2:
        raise ValueError("Excitatory inputs must be a 2D array.")

    num_inputs, samples = inputs.shape
    if min_inputs > num_inputs:
        raise ValueError("min_inputs should be less than or equal to the number of inputs.")

    output = np.zeros(samples)
    for i in range(min_inputs, num_inputs + 1):
        output += _ee(inputs, delta_s, fs)

    return output


def simple_ee(inputs, delta_s: float, fs: float) -> np.ndarray:
    """
    The simple EE cell generates an output spike whenever both inputs spike within a time interval of less than
    ∆ seconds.

    Parameters
        inputs (np.ndarray): 2D array of excitatory inputs.
        delta_s (float): Coincidence integration duration in seconds.
        fs (float): Sampling frequency.

    Returns:
        np.ndarray: Output after applying the excitatory-excitatory interaction with the specified criteria.

    Raises:
        ValueError: If inputs are not 2D or if min_inputs exceeds the number of inputs.
    """
    if inputs.ndim != 2:
        raise ValueError("Excitatory inputs must be a 2D array.")

    return _ee(inputs, delta_s, fs)
