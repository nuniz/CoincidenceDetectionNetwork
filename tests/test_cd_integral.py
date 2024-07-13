import unittest

import numpy as np
from scipy import signal

from cd_network.coincidence_integral import (
    apply_filter,
    coincidence_integral,
    create_trapezoid_kernel,
)


class TestCoincidenceIntegration(unittest.TestCase):
    def test_create_trapezoid_kernel(self):
        """Test the creation of the trapezoidal kernel."""
        samples_integral = 5
        expected_kernel = np.array([1.0, 2.0, 2.0, 2.0, 1.0])
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

    def test_coincidence_integral(self):
        """Test the coincidence_integral function for various integration methods."""
        x = np.random.randn(3, 100)
        integration_duration = 1
        fs = 5  # sample frequency
        methods = ["filtfilt", "lfilter"]
        for method in methods:
            result = coincidence_integral(x, integration_duration, fs, method)
            self.assertEqual(result.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
