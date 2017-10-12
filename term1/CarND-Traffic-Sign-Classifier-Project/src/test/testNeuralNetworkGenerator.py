import unittest

from NeuralNetworkOperations import *
from NeuralNetworkGenerator import *


class TestNeuralNetworkGenerator(unittest.TestCase):
    def test_number_of_permutations(self):
        expected_n_conv2d_permutations = 4
        expected_n_dense_permutations = 100
        tuning_options_list = [
            Conv2dOperation.get_training_options(max_n_permutations=expected_n_conv2d_permutations,
                                                 layer_size=0.5),
            DenseOperation.get_training_options(max_n_permutations=expected_n_dense_permutations,
                                                layer_size=0.1)
        ]
        generator = NeuralNetworkGenerator(n_classes=10, tuning_options_list=tuning_options_list,
                                           image_shape=[32, 32, 3])
        actual_n_permutations = 0
        for nn, x, y in generator.generate():
            actual_n_permutations += 1

        self.assertEqual(actual_n_permutations,
                         expected_n_conv2d_permutations * expected_n_dense_permutations)
