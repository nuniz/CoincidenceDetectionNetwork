from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import integrate, signal

from .utils import hashable_input


def create_trapezoid_kernel(samples_integral: int) -> np.ndarray:
    """
    Create a trapezoidal kernel for signal integration.

    Parameters:
        samples_integral (int): Number of samples over which to integrate.

    Returns:
        np.ndarray: The trapezoidal kernel.
    """
    return np.concatenate(([0], np.ones(samples_integral - 1))) + np.concatenate(
        (np.ones(samples_integral - 1), [0])
    )


def apply_filter(
    x: np.ndarray, kernel: np.ndarray, dt: float, filter_func: Callable
) -> np.ndarray:
    """
    Apply a filtering function to an input signal using a specified kernel.

    Parameters:
        x (np.ndarray): The input signal.
        kernel (np.ndarray): The filter kernel.
        dt (float): The time step.
        filter_func (Callable): The filtering function from scipy.signal.

    Returns:
        np.ndarray: The filtered signal.
    """
    return filter_func(kernel, 1, x) * dt / 2


def integrate_signal(
    x: np.ndarray, dt: float, delta_samples: int, method: str
) -> np.ndarray:
    """
    Compute the integral of the input signal using a specified method.

    Parameters:
        x (np.ndarray): The input signal.
        dt (float): The time step.
        delta_samples (int): Number of samples to shift for delta calculations.
        method (str): Integration method ('trapz', 'simps', 'romb').

    Returns:
        np.ndarray: The integrated signal.
    """
    if method not in ["trapz", "simps", "romb"]:
        raise ValueError("Unknown integration method.")

    num_samples = x.size
    integrated_values = np.zeros(num_samples)
    for j in range(num_samples):
        end_sample = j + 1
        start_sample = max(0, end_sample - delta_samples)
        samples_slice = x[start_sample:end_sample]
        integrate_func = getattr(integrate, method)
        integrated_values[j] = integrate_func(y=samples_slice, dx=dt)
    return integrated_values


def coincidence_integral(
    x: np.ndarray, integration_duration: float, fs: float, method: str = "filtfilt"
) -> np.ndarray:
    """
    Computes the coincidence integral of the input signal.

    Parameters:
        x (np.ndarray): The input signal.
        integration_duration (float): The duration over which to integrate.
        fs (float): The sampling frequency.
        method (str): The method for integration ('filtfilt', 'lfilter', 'cumtrapz', 'trapz', 'simps', or 'romb').

    Returns:
        np.ndarray: The coincidence integral of the signal.
    """
    dt = 1 / fs
    samples_integral = int(np.floor(integration_duration * fs))
    kernel = create_trapezoid_kernel(samples_integral)

    filter_methods = {
        "filtfilt": lambda x: apply_filter(x, kernel, dt, signal.filtfilt),
        "lfilter": lambda x: apply_filter(x, kernel, dt, signal.lfilter),
    }
    if method in filter_methods:
        return filter_methods[method](x)

    num_inputs, num_samples = x.shape
    output = np.zeros((num_inputs, num_samples))

    for i in range(num_inputs):
        if method == "cumtrapz":
            output[i, :] = integrate.cumtrapz(y=x[i, :], dx=dt, initial=0)
            output[i, samples_integral:] -= output[i, :-samples_integral]
        else:
            output[i, :] = integrate_signal(x[i, :], dt, samples_integral, method)

    return output


@lru_cache(maxsize=None)
def cached_coincidence_integral_computation(inputs_tuple, delta_s, fs):
    """Cached version of the coincidence_integral computation, uses hashable input."""
    inputs = np.array(inputs_tuple)  # Convert tuple back to numpy array
    return coincidence_integral(inputs, delta_s, fs)


def cached_coincidence_integral(inputs, delta_s, fs):
    """Prepare the input for caching and call the cached computation."""
    inputs_tuple = hashable_input(inputs)
    return cached_coincidence_integral_computation(inputs_tuple, delta_s, fs)
