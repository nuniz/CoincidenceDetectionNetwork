import unittest
import numpy as np
from cd_network.network import NeuralNetwork


class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        """Set up the Neural Network with a configuration path."""
        self.config_path = r'../external_data/config.json'
        self.network = NeuralNetwork(self.config_path)
        self.external_inputs = {
            'external1': np.random.randn(1000),  # Example external excitatory inputs
            'external2': np.random.randn(1000),  # Example external inhibitory inputs
            'external3': np.random.randn(1000)  # Example external inhibitory inputs
        }

    def test_run_network(self):
        """Test that the network runs and outputs in the correct format, with keys for cells."""
        outputs = self.network.run_network(self.external_inputs)

        # Check that outputs is a dictionary
        self.assertIsInstance(outputs, dict, "Output should be a dictionary.")

        # Check that each key in the dictionary corresponds to the expected cells
        expected_keys = {'cell1', 'cell2', 'cell3'}
        self.assertEqual(set(outputs.keys()), expected_keys, "Output keys should match expected cell identifiers.")

        # Check that each value is a numpy array and has the expected properties
        for key, output in outputs.items():
            self.assertIsInstance(output, np.ndarray, f"Output for cell {key} should be a numpy array.")
            self.assertEqual(output.shape[0], 1000, f"Output length for cell {key} should be 1000.")
            self.assertEqual(output.dtype, np.float64, f"Output elements for cell {key} should be of type np.float64.")


if __name__ == '__main__':
    unittest.main()
