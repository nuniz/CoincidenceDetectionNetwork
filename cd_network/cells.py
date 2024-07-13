from itertools import combinations

import numpy as np

from .coincidence_integral import cached_coincidence_integral, coincidence_integral

EPS = 1e-15


def ei(
        excitatory_input: np.ndarray,
        inhibitory_inputs: np.ndarray,
        delta_s: float,
        fs: float,
) -> np.ndarray:
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
    assert inhibitory_inputs.ndim in [
        1,
        2,
    ], "Inhibitory inputs must be either 1D or 2D array."

    if inhibitory_inputs.ndim == 1:
        inhibitory_inputs = inhibitory_inputs[np.newaxis, ...]

    assert (
            len(excitatory_input) == inhibitory_inputs.shape[-1]
    ), "Length of excitatory input must match the size of inhibitory inputs along the last axis."

    output = excitatory_input * np.prod(
        1 - coincidence_integral(inhibitory_inputs, delta_s, fs), axis=0
    )
    return output


def _all_spikes_ee(inputs: np.ndarray, delta_s: float, fs: float) -> np.ndarray:
    """
    An all-spike EE cell generates a spike whenever all of its inputs spikes during an interval ∆.

    Parameters:
        inputs (np.ndarray): 2D array of excitatory inputs.
        fs (float): Sampling frequency.
        delta_s: Coincidence integration duration in seconds.

    Returns:
        np.ndarray: Output after applying the excitatory-excitatory interaction.
    """
    assert inputs.ndim == 2, "Excitatory inputs must be a 2D array."

    coincidence_integral_outputs = cached_coincidence_integral(inputs, delta_s, fs)
    coincidence_prod = np.prod(coincidence_integral_outputs, axis=0)

    n_inputs, samples = inputs.shape
    output = np.zeros(samples)
    for i in range(n_inputs):
        output += inputs[i] * coincidence_prod / (coincidence_integral_outputs[i] + EPS)
    return output


def _exactly_n_spikes_ee(
        inputs: np.ndarray, n_spikes: int, delta_s: float, fs: float
) -> np.ndarray:
    """
    An all-spikes EE cell generates a spike whenever exactly n_spikes of its inputs spikes during an interval ∆.

    Parameters:
        inputs (np.ndarray): 2D array of excitatory inputs.
        n_spikes (int): Exact number of inputs that must spike.
        fs (float): Sampling frequency.
        delta_s: Coincidence integration duration in seconds.

    Returns:
        np.ndarray: Output after applying the excitatory-excitatory interaction.
    """
    assert inputs.ndim == 2, "Excitatory inputs must be a 2D array."

    n_inputs, samples = inputs.shape
    assert (
            n_inputs <= n_inputs
    ), "n_spikes should be less than or equal to the number of inputs."

    output = np.zeros(samples)
    binomial_combinations = list(combinations(range(n_inputs), n_spikes))

    for comb in binomial_combinations:
        indices_spike = np.array(comb)
        indices_not_spike = np.array(list(set(range(n_inputs)) - set(indices_spike)))
        if len(indices_not_spike) > 0 & len(indices_spike) > 0:
            ei_output = ei(
                excitatory_input=_all_spikes_ee(
                    inputs=inputs[indices_spike], delta_s=delta_s, fs=fs
                ),
                inhibitory_inputs=inputs[indices_not_spike],
                delta_s=delta_s,
                fs=fs,
            )
            output += ei_output

    return output


def simple_ee(inputs, delta_s: float, fs: float) -> np.ndarray:
    """
    The simple EE cell generates an output spike whenever both inputs spike within a time interval of less than
    ∆ seconds.

    Parameters:
        inputs (np.ndarray): 2D array of excitatory inputs.
        delta_s (float): Coincidence integration duration in seconds.
        fs (float): Sampling frequency.

    Returns:
        np.ndarray: Output after applying the excitatory-excitatory interaction with the specified criteria.

    Raises:
        ValueError: If inputs are not 2D.
    """
    if inputs.ndim != 2:
        raise ValueError("Excitatory inputs must be a 2D array.")

    return _all_spikes_ee(inputs, delta_s, fs)


def ee(inputs, n_spikes: int, delta_s: float, fs: float) -> np.ndarray:
    """
    A general EE cell generates a spike whenever at least min_input of its inputs spikes during an interval ∆.

    Parameters:
        inputs (np.ndarray): 2D array of excitatory inputs.
        n_spikes (int): Minimum number of inputs that must spike.
        delta_s (float): Coincidence integration duration in seconds.
        fs (float): Sampling frequency.

    Returns:
        np.ndarray: Output after applying the excitatory-excitatory interaction with the specified criteria.

    Raises:
        ValueError: If inputs are not 2D or if min_spike_inputs exceeds the number of inputs.
    """
    if inputs.ndim != 2:
        raise ValueError("Excitatory inputs must be a 2D array.")

    n_inputs, samples = inputs.shape
    assert (
            n_inputs <= n_inputs
    ), "n_spikes should be less than or equal to the number of inputs."

    output = np.zeros(samples)
    for i in range(n_spikes, n_inputs + 1):
        output += _exactly_n_spikes_ee(inputs, i, delta_s, fs)

    return output


def cd(
        excitatory_inputs: np.ndarray,
        inhibitory_inputs: np.ndarray,
        n_spikes: int,
        delta_s: float,
        fs: float,
) -> np.ndarray:
    """
    A general CD cell is defined as one with n_excitatory_inputs excitatory inputs and n_inhibitory_inputs inhibitory
    inputs. This type of cell generates a spike if, during an interval of length ∆, there are at least n_spikes more
    excitatory spikes than inhibitory spikes

    Parameters
        excitatory_inputs (np.ndarray): Excitatory input spikes, shape (n_excitatory_inputs, excitatory_samples).
        inhibitory_inputs (np.ndarray): Inhibitory input spikes, shape (n_inhibitory_inputs, inhibitory_samples).
        n_spikes (int): Minimum excess of excitatory spikes over inhibitory spikes to generate an output spike.
        delta_s (float): Interval length ∆ in seconds.
        fs (float): Sampling frequency in Hz.

    Returns:
        np.ndarray: Output after applying the cd interaction with the specified criteria.

    Raises:
        ValueError: If inputs are not 2D or if min_inputs exceeds the number of inputs.
    """
    assert excitatory_inputs.ndim in [
        1,
        2,
    ], "Excitatory inputs must be either 1D or 2D array."
    assert inhibitory_inputs.ndim in [
        1,
        2,
    ], "Inhibitory inputs must be either 1D or 2D array."

    if excitatory_inputs.ndim == 1:
        excitatory_inputs = excitatory_inputs[np.newaxis, ...]
    if inhibitory_inputs.ndim == 1:
        inhibitory_inputs = inhibitory_inputs[np.newaxis, ...]

    n_excitatory_inputs, excitatory_samples = excitatory_inputs.shape
    n_inhibitory_inputs, inhibitory_samples = inhibitory_inputs.shape

    assert (
            inhibitory_samples == excitatory_samples
    ), "Number of samples in inhibitory and excitatory inputs must match."

    output = np.zeros(excitatory_samples)
    for i in range(min(n_inhibitory_inputs, n_excitatory_inputs - n_spikes) + 1):
        if i == 0:
            output += ei(
                excitatory_input=ee(excitatory_inputs, n_spikes, delta_s, fs),
                inhibitory_inputs=inhibitory_inputs,
                delta_s=delta_s,
                fs=fs,
            )
        elif 1 <= i <= min(inhibitory_samples - 1, excitatory_samples - n_spikes):
            output += ei(
                excitatory_input=ee(excitatory_inputs, n_spikes + i, delta_s, fs),
                inhibitory_inputs=_all_spikes_ee(inhibitory_inputs, delta_s, fs),
                delta_s=delta_s,
                fs=fs,
            )
        elif i == inhibitory_inputs:
            output += ee(excitatory_inputs, n_spikes + i, delta_s, fs)
        else:
            raise Exception("Unexpected case in loop.")

    return output
