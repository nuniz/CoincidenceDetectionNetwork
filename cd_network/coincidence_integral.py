import numpy as np
from scipy import signal, integrate
from typing import Union


def create_trapezoid_kernel(samples_integral: int) -> np.ndarray:
    """
    Creates a trapezoidal kernel for signal processing.

    Parameters:
        samples_integral (int): The number of samples for integration.

    Returns:
        np.ndarray: The trapezoidal kernel.
    """
    return np.concatenate(([0], np.ones(samples_integral - 1))) + np.concatenate((np.ones(samples_integral - 1), [0]))


def apply_filter_method(x: np.ndarray, trapezoid_kernel: np.ndarray, dt: float, method: str) -> np.ndarray:
    """
    Applies a filtering method to the input signal.

    Parameters:
        x (np.ndarray): The input signal.
        trapezoid_kernel (np.ndarray): The trapezoidal kernel.
        dt (float): The time step.
        method (str): The filtering method ('filtfilt' or 'lfilter').

    Returns:
        np.ndarray: The filtered signal.
    """
    if method == 'filtfilt':
        return signal.filtfilt(trapezoid_kernel, 1, x) * dt / 2
    elif method == 'lfilter':
        return signal.lfilter(trapezoid_kernel, 1, x) * dt / 2
    else:
        raise ValueError('Unknown filter method.')


def integrate_cumtrapz(x: np.ndarray, dt: float, delta_samples: int) -> np.ndarray:
    """
    Computes the cumulative integral of the input signal using the trapezoidal rule.

    Parameters:
        x (np.ndarray): The input signal.
        dt (float): The time step.
        delta_samples (int): The number of samples for the delta calculation.

    Returns:
        np.ndarray: The cumulative integral of the signal.
    """
    cumulative_integral = integrate.cumtrapz(y=x, dx=dt, initial=0)
    cumulative_integral[delta_samples:] -= cumulative_integral[:-delta_samples]
    return cumulative_integral


def integrate_samples(x: np.ndarray, dt: float, delta_samples: int, method: str) -> np.ndarray:
    """
    Integrates the input signal over a sliding window.

    Parameters:
        x (np.ndarray): The input signal.
        dt (float): The time step.
        delta_samples (int): The number of samples for the sliding window.
        method (str): The integration method ('trapz', 'simps', or 'romb').

    Returns:
        np.ndarray: The integrated signal.
    """
    num_samples = x.size
    integrated_values = np.zeros(num_samples)
    for j in range(num_samples):
        end_sample = j + 1
        start_sample = max(0, end_sample - delta_samples)
        samples_slice = x[start_sample:end_sample]

        if method == "trapz":
            integrated_values[j] = integrate.trapz(y=samples_slice, dx=dt)
        elif method == "simps":
            integrated_values[j] = integrate.simps(y=samples_slice, dx=dt)
        elif method == "romb":
            integrated_values[j] = integrate.romb(y=samples_slice, dx=dt)
        else:
            raise ValueError('Unknown integration method.')
    return integrated_values


def coincidence_integral(x: np.ndarray, integration_duration: float, fs: float, method: str = "filtfilt") -> np.ndarray:
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
    trapezoid_kernel = create_trapezoid_kernel(samples_integral)

    if method in ['filtfilt', 'lfilter']:
        return apply_filter_method(x, trapezoid_kernel, dt, method)

    delta_samples = samples_integral
    num_inputs, num_samples = x.shape
    output = np.zeros((num_inputs, num_samples))

    for i in range(num_inputs):
        if method == "cumtrapz":
            output[i, :] = integrate_cumtrapz(x[i, :], dt, delta_samples)
        else:
            output[i, :] = integrate_samples(x[i, :], dt, delta_samples, method)

    return output
