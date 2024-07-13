import unittest
import numpy as np
from scipy import signal
from cd_network.coincidence_integral import create_trapezoid_kernel, apply_filter, integrate_signal, \
    coincidence_integral


class TestCD(unittest.TestCase):
    def test_create_trapezoid_kernel(self):
        """Test the creation of the trapezoidal kernel."""
        samples_integral = 5
        expected_kernel = np.array([1., 2., 2., 2., 1.])
        result = create_trapezoid_kernel(samples_integral)
        np.testing.assert_array_equal(result, expected_kernel)

    def test_apply_filter(self):
        """Test applying filters using filtfilt and lfilter."""
        x = np.random.randn(100)
        kernel = np.array([1, 2, 1], dtype=float)
        dt = 0.1
        result_f = apply_filter(x, kernel, dt, signal.filtfilt)
        result_l = apply_filter(x, kernel, dt, signal.lfilter)
        # Check results for general shape and type
        self.assertEqual(result_f.shape, x.shape)
        self.assertEqual(result_l.shape, x.shape)

    def test_integrate_signal_trapz(self):
        """Test the integrate_signal function using the trapezoidal method."""
        x = np.linspace(0, 1, 10)
        dt = 0.1
        delta_samples = 5
        expected_length = len(x)
        result = integrate_signal(x, dt, delta_samples, 'trapz')
        self.assertEqual(len(result), expected_length)

    def test_integrate_signal_simps(self):
        """Test the integrate_signal function using Simpson's rule."""
        x = np.linspace(0, 1, 10)
        dt = 0.1
        delta_samples = 5
        expected_length = len(x)
        result = integrate_signal(x, dt, delta_samples, 'simps')
        self.assertEqual(len(result), expected_length)

    def test_integrate_signal_unknown_method(self):
        """Test the integrate_signal function for error handling."""
        x = np.linspace(0, 1, 10)
        dt = 0.1
        delta_samples = 5
        with self.assertRaises(ValueError):
            integrate_signal(x, dt, delta_samples, 'unknown_method')

    def test_coincidence_integral(self):
        """Test the coincidence_integral function for various integration methods."""
        x = np.random.randn(3, 100)
        integration_duration = 1
        fs = 5  # sample frequency
        methods = ['filtfilt', 'lfilter', 'cumtrapz', 'trapz', 'simps', 'romb']
        for method in methods:
            if method in ['romb'] and x.shape[1] % 2 == 0:  # Romb requires 2^n + 1 samples
                continue
            result = coincidence_integral(x, integration_duration, fs, method)
            self.assertEqual(result.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
