import unittest

import numpy as np

from cd_network.cells import _all_spikes_ee, _exactly_n_spikes_ee, cd, ee, ei, simple_ee


class TestNeuralFunctions(unittest.TestCase):
    INPUT_LENGTH = 32000
    NUM_INPUT_CHANNELS = 4
    OUTPUT_LENGTH = 32000

    @classmethod
    def setUpClass(cls):
        np.random.seed(42)  # for reproducibility

    def setUp(self):
        self.excitatory_input = np.random.choice([0, 1], size=self.INPUT_LENGTH)
        self.inhibitory_input = np.random.choice([0, 1], size=self.INPUT_LENGTH)
        self.inputs = np.random.choice(
            [0, 1], size=(self.NUM_INPUT_CHANNELS, self.INPUT_LENGTH)
        )
        self.delta_s = 0.1  # 100 ms
        self.fs = 16000  # 16000 Hz

    def test_ei_output_shape(self):
        result = ei(self.excitatory_input, self.inhibitory_input, self.delta_s, self.fs)
        self.assertEqual(
            result.shape,
            (self.OUTPUT_LENGTH,),
            "Output shape mismatch for ei function.",
        )

    def test_all_spikes_ee_output_shape(self):
        result = _all_spikes_ee(self.inputs, self.delta_s, self.fs)
        self.assertEqual(
            result.shape,
            (self.OUTPUT_LENGTH,),
            "Output shape mismatch for _all_spikes_ee function.",
        )

    def test_exactly_n_spikes_ee_output_shape(self):
        result = _exactly_n_spikes_ee(self.inputs, 2, self.delta_s, self.fs)
        self.assertEqual(
            result.shape,
            (self.OUTPUT_LENGTH,),
            "Output shape mismatch for _exactly_n_spikes_ee function.",
        )

    def test_simple_ee_output_shape(self):
        result = simple_ee(self.inputs, self.delta_s, self.fs)
        self.assertEqual(
            result.shape,
            (self.OUTPUT_LENGTH,),
            "Output shape mismatch for simple_ee function.",
        )

    def test_ee_output_shape(self):
        result = ee(self.inputs, 1, self.delta_s, self.fs)
        self.assertEqual(
            result.shape,
            (self.OUTPUT_LENGTH,),
            "Output shape mismatch for ee function.",
        )

    def test_cd_output_shape(self):
        inhibitory_inputs = np.random.choice(
            [0, 1], size=(self.NUM_INPUT_CHANNELS, self.INPUT_LENGTH)
        )
        result = cd(self.inputs, inhibitory_inputs, 1, self.delta_s, self.fs)
        self.assertEqual(
            result.shape,
            (self.OUTPUT_LENGTH,),
            "Output shape mismatch for cd function.",
        )


if __name__ == "__main__":
    unittest.main()
