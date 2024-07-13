from functools import lru_cache
from typing import Callable

import numpy as np
from scipy import signal

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


def coincidence_integral(
        x: np.ndarray, integration_duration: float, fs: float, method: str = "filtfilt"
) -> np.ndarray:
    """
    Computes the coincidence integral of the input signal.

    Parameters:
        x (np.ndarray): The input signal.
        integration_duration (float): The duration over which to integrate.
        fs (float): The sampling frequency.
        method (str): The method for integration ('filtfilt', 'lfilter', 'cumtrapz').

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

    raise ValueError(f'method {method} is not supported.')


@lru_cache(maxsize=None)
def cached_coincidence_integral_computation(inputs_tuple, delta_s, fs):
    """Cached version of the coincidence_integral computation, uses hashable input."""
    inputs = np.array(inputs_tuple)  # Convert tuple back to numpy array
    return coincidence_integral(inputs, delta_s, fs)


def cached_coincidence_integral(inputs, delta_s, fs):
    """Prepare the input for caching and call the cached computation."""
    inputs_tuple = hashable_input(inputs)
    return cached_coincidence_integral_computation(inputs_tuple, delta_s, fs)
