"""
Wrapper for neural network operations in Tensorflow.

This module simplifies configuration of neural network layers in tensorflow, by
hiding some aspects such as weights/biases size calculation and initialization.

"""

import tensorflow as tf
import math
import functools


class NeuralNetworkOperation(object):
    """
    Super class inherited by all operations.
    """

    def __init__(self, name, weights=None, biases=None):
        self.name = name
        self.weights = weights
        self.biases = biases

    def get_weights(self):
        """
        Get weights for this operation if available, otherwise returns None.
        """
        return self.weights

    def get_biases(self):
        """
        Get biases for this operation if available, otherwise returns None.
        """
        return self.biases

    def get_operation(self, x):
        """
        Call neural-network operation(s) with parameter x and return result.
        This is an "abstract" method and should be implemented by all sub-classes.
        """
        assert(False)
        return x

    def get_feed_dict(self, training):
        """
        Override this method in subclasses having placeholder variables that must
        be configured differently for training and validation. The default is to
        not provide anything for the feed_dict.
        """
        return dict()


class DenseOperation(NeuralNetworkOperation):
    """
    Sub-class for a full neural network layer operation.

    Arguments:
        name: The name of this operation.
        input_shape: Shape of input array. The input will be flatten if multidimensional.
        n_output_channels: The output from this layer is a 1D array of this length.
        mu: Mean value for the initial, generated weights.
        sigma: Standard deviation for the initial, generated weights.
        activation: Activation function. If None, than no activation is applied.
    """

    def __init__(self, name, input_shape, n_output_channels, mu=0.0, sigma=0.1, activation=tf.nn.relu):
        if len(input_shape) > 1:
            n_input_channels = functools.reduce(lambda x, y: x * y, input_shape)
            self.flatten = True
        else:
            n_input_channels = input_shape[0]
            self.flatten = False

        weights = tf.Variable(tf.truncated_normal([n_input_channels, n_output_channels],
                                                  mean=mu, stddev=sigma, name=f"{name}_weights"))
        biases = tf.Variable(tf.zeros(n_output_channels), name=f"{name}_biases")
        NeuralNetworkOperation.__init__(self, name=name, weights=weights, biases=biases)
        self.activation = activation

    def get_operation(self, x):
        if self.flatten:
            x = tf.contrib.layers.flatten(x)
        x = tf.matmul(x, self.weights, name=f"{self.name}_matmul")
        x = tf.add(x, self.biases, name=f"{self.name}_add")
        if self.activation is not None:
            x = self.activation(x, name=f"{self.name}_activation")
        return x


class Conv2dOperation(NeuralNetworkOperation):
    """Sub-class for a convolutional neural network layer operation.

    Arguments:
        name: The name of this operation.
        input_shape: Shape of input array: [rows, cols, channels].
        n_output_channels: Number of output channels.
        filter_shape: Shape of the filter kernel.
        strides: Filter stride: (rows, cols)
        mu: Mean value for the initial, generated weights.
        sigma: Standard deviation for the initial, generated weights.
        activation: Activation function. If None, than no activation is applied.
    """

    def __init__(self, name, input_shape, n_output_channels, filter_shape, strides=(1, 1), mu=0.0,
                 sigma=0.1, activation=tf.nn.relu):

        assert(len(input_shape) == 3)
        output_shape = [math.ceil(float(input_shape[0] - filter_shape[0] + 1) / float(strides[0])),
                        math.ceil(float(input_shape[1] - filter_shape[1] + 1) / float(strides[1])),
                        n_output_channels]

        assert(output_shape[0] >= 1)
        assert(output_shape[1] >= 1)

        weights = tf.Variable(tf.truncated_normal([filter_shape[0],
                                                   filter_shape[1],
                                                   input_shape[2],
                                                   n_output_channels],
                                                  mean=mu, stddev=sigma), name=f"{name}_weights")
        biases = tf.Variable(tf.zeros(n_output_channels), name=f"{name}_biases")
        NeuralNetworkOperation.__init__(self, name=name, weights=weights, biases=biases)
        self.strides = [1, strides[0], strides[1], 1]  # Batch and channel stride hard-coded to 1
        self.activation = activation

    def get_operation(self, x):
        x = tf.nn.conv2d(x, self.weights, self.strides, padding='VALID', name=f"{self.name}_conv2d")
        x = tf.nn.bias_add(x, self.biases, name=f"{self.name}_bias_add")
        if self.activation is not None:
            x = self.activation(x, name=f"{self.name}_activation")
        return x


class MaxPoolOperation(NeuralNetworkOperation):
    """Sub-class for a max-pool neural network operation.

    Arguments:
        name: The name of this operation.
        input_shape: Shape of input array: [rows, cols, channels].
        n_output_channels: Number of output channels.
        ksize: Shape of the pool [batch, rows, cols, channels].
        strides: Pool stride: [batch, rows, cols, channels].
    """

    def __init__(self, name, input_shape, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
        assert(len(input_shape) == 3)
        assert(len(ksize) == 4)
        assert(len(strides) == 4)
        NeuralNetworkOperation.__init__(self, name=name)
        self.ksize = ksize
        self.strides = strides

    def get_operation(self, x):
        x = tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding='VALID')
        return x


class DropoutOperation(NeuralNetworkOperation):
    """Sub-class for a dropout neural network operation.

    This operation have tensorflow placeholder variables that must have different
    value when training and validating. This can easily be achieved by appending
    the output from the get_feed_dict method to the feed_dict used for the
    tensorflow session run.

    Arguments:
        name: The name of this operation.
        input_shape: Not used by this NeuralNetworkOperation.
        keep_prob: The probability that each element is kept.
    """

    def __init__(self, name, input_shape, keep_prob):
        NeuralNetworkOperation.__init__(self, name=name)
        self.keep_prob = keep_prob
        self.tf_placeholder = tf.placeholder(tf.float32, (None), name=f"{name}_keep_prob")

    def get_operation(self, x):
        x = tf.nn.dropout(x, self.tf_placeholder, name=f"{self.name}_dropout")
        return x

    def get_feed_dict(self, training):
        """
        Use different keep_prob for training and validation.
        """
        if training:
            return {self.tf_placeholder: self.keep_prob}
        else:
            return {self.tf_placeholder: 1.0}
