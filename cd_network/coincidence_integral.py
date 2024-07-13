import numpy as np
from scipy import signal, integrate


def create_trapezoid_kernel(samples_integral):
    return np.concatenate(([0], np.ones(samples_integral - 1))) + np.concatenate((np.ones(samples_integral - 1), [0]))


def apply_filter_method(x, trapezoid_kernel, dt, method):
    if method == 'filtfilt':
        return signal.filtfilt(trapezoid_kernel, 1, x) * dt / 2
    elif method == 'lfilter':
        return signal.lfilter(trapezoid_kernel, 1, x) * dt / 2
    else:
        raise ValueError('Unknown filter method.')


def integrate_cumtrapz(x, dt, delta_samples):
    cumulative_integral = integrate.cumtrapz(y=x, dx=dt, initial=0)
    cumulative_integral[delta_samples:] -= cumulative_integral[:-delta_samples]
    return cumulative_integral


def integrate_samples(x, dt, delta_samples, method):
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


def coincidence_integral(x, integration_duration, fs, method="filtfilt"):
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
