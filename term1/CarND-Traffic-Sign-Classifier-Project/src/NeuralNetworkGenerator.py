import numpy as np

from NeuralNetwork import NeuralNetwork
from NeuralNetworkOperations import *


class NeuralNetworkGenerator(object):
    def __init__(self, n_classes, n_conv_layers=0, n_dense_layers=0, use_max_pool=False, use_dropout=False):
        self.n_classes = n_classes
        self.n_conv_layers = n_conv_layers
        self.n_dense_layers = n_dense_layers
        self.use_max_pool = use_max_pool
        self.use_dropout = use_dropout

    def generate(self, x):
        for generator_filter_size, generator_channel_factor in zip(range(2, 8), range(1, 6)):
            nn = NeuralNetwork(x)
            for layer_factor in np.arange(self.n_conv_layers) + 1.0:
                n_output_channels = np.round(6 * layer_factor * generator_channel_factor).astype(np.int)
                nn.add(Conv2dOperation, n_output_channels=n_output_channels,
                       filter_shape=[generator_filter_size, generator_filter_size])
                if self.use_max_pool:
                    nn.add(MaxPoolOperation)

            for layer_factor in np.arange(self.n_dense_layers) + 1.0:
                n_output_channels = np.round(120 * generator_channel_factor / layer_factor).astype(np.int)
                nn.add(DenseOperation, n_output_channels=120 * generator_channel_factor)

            if self.use_dropout:
                nn.add(DropoutOperation, keep_prob=0.5)

            # Non optional output layer
            nn.add(DenseOperation, n_output_channels=self.n_classes, activation=None)

            yield nn
