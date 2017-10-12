"""
Wrapper for neural network operations in Tensorflow.

This module simplifies configuration of neural network layers in tensorflow, by
hiding some aspects such as weights/biases size calculation and initialization.

"""

import tensorflow as tf
import math
import functools
import numpy as np
import warnings


class NeuralNetworkOperation(object):
    """Super class inherited by all operations."""

    def __init__(self, name, weights=None, biases=None):
        self.name = name
        self.weights = weights
        self.biases = biases

    def get_weights(self):
        """Get weights for this operation if available, otherwise returns None."""
        return self.weights

    def get_biases(self):
        """Get biases for this operation if available, otherwise returns None."""
        return self.biases

    def get_operation(self, x):
        """Call neural-network operation(s) with input tensor x and return result.
        This is an "abstract" method and must be implemented by all sub-classes.
        """
        assert(False)
        return x

    def get_feed_dict(self, training):
        """Override this method in subclasses having placeholder variables that must
        be configured differently for training and validation. The default is to
        not provide anything for the feed_dict.
        """
        return dict()

    def __str__(self):
        """Return string representation of the nn-operation instance.

        Present each operation as a one line string, with all relevant arguments.
        Sub-classes must either override this method or set self.str_repr.
        """
        return self.str_repr

    @classmethod
    def get_training_options(cls, max_n_permutations=1, layer_size=0.0, **kwargs):
        """Provides suitable argument ranges to try for this operation. This method
        should be overridden by all subclasses with arguments.

        Arguments:
            max_n_permutations: Max number of ways to alternate the arguments.
            layer_size: Defines the expected size of this operation with an arbitrary
                        number in the range 0 to 1, where 0 is "small" and 1 is "huge".
            kwargs: user specified argument values to use.

        Returns:
            Tuple with object class and a dictionary with lists of values to try for each argument.
        """
        return (cls, dict())


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
        self.str_repr = (f"{type(self).__name__}(name=\"{name}\", input_shape={input_shape}, "
                         f"n_output_channels={n_output_channels}, mu={mu}, sigma={sigma}, "
                         f"activation={'None' if activation is None else activation.__name__})")

    def get_operation(self, x):
        if self.flatten:
            x = tf.contrib.layers.flatten(x)
        x = tf.matmul(x, self.weights, name=f"{self.name}_matmul")
        x = tf.add(x, self.biases, name=f"{self.name}_add")
        if self.activation is not None:
            x = self.activation(x, name=f"{self.name}_activation")
        return x

    @classmethod
    def get_training_options(cls, max_n_permutations=1, layer_size=0.0, **kwargs):
        """Get training options for this operation

        Only tune number of output channels for now, and use default values for
        mu, sigma and the activation function.

        Arguments:
            max_n_permutations: Max number of ways to alternate the arguments.
            layer_size: Defines the expected size of this operation with an arbitrary
                        number in the range 0 to 1, where 0 is "small" and 1 is "huge".
            kwargs: user specified argument values to use.

        Returns:
            Tuple with object class and a dictionary with lists of values to try for each argument.
        """
        n_remaining_permutations = training_options_arg_helper(cls.__name__, max_n_permutations, layer_size,
                                                               **kwargs)
        training_options = {'n_output_channels': None,  # Will be replaced below
                            'mu': [0.0],
                            'sigma': [0.1],
                            'activation': [tf.nn.relu]}

        if 'n_output_channels' not in kwargs:

            target_n_output_channels = layer_size_to_absolute_value(layer_size=layer_size,
                                                                    small_value=50,
                                                                    huge_value=1000000)

            # Distribute n_remaining_permutations linearly in region [0.9, 1.1] * target_n_output_channels
            n_output_channels_list = target_n_output_channels * get_linear_distribution_list(n_remaining_permutations)
            n_output_channels_list = np.round(n_output_channels_list).astype(np.int)
            n_output_channels_list = np.unique(n_output_channels_list)  # Remove any duplicates

            training_options['n_output_channels'] = n_output_channels_list

        training_options.update(kwargs)  # Replace with user defined.
        return (cls, training_options)


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

    def __init__(self, name, input_shape, n_output_channels, filter_shape=[5, 5], strides=(1, 1), mu=0.0,
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
        self.str_repr = (f"{type(self).__name__}(name=\"{name}\", input_shape={input_shape}, "
                         f"n_output_channels={n_output_channels}, filter_shape={filter_shape}, strides={strides}, "
                         f"mu={mu}, sigma={sigma}, activation={'None' if activation is None else activation.__name__})")

    def get_operation(self, x):
        x = tf.nn.conv2d(x, self.weights, self.strides, padding='VALID', name=f"{self.name}_conv2d")
        x = tf.nn.bias_add(x, self.biases, name=f"{self.name}_bias_add")
        if self.activation is not None:
            x = self.activation(x, name=f"{self.name}_activation")
        return x

    @classmethod
    def get_training_options(cls, max_n_permutations=1, layer_size=0.0, **kwargs):
        """Get training options for this operation

        Only tune number of output channels for now, and use default values for
        filter_shape, strides, mu, sigma and the activation function.

        Arguments:
            max_n_permutations: Max number of ways to alternate the arguments.
            layer_size: Defines the expected size of this operation with an arbitrary
                        number in the range 0 to 1, where 0 is "small" and 1 is "huge".
            kwargs: user specified argument values to use.

        Returns:
            Tuple with object class and a dictionary with lists of values to try for each argument.
        """
        n_remaining_permutations = training_options_arg_helper(cls.__name__, max_n_permutations, layer_size,
                                                               **kwargs)

        # Default training options
        training_options = {'n_output_channels': None,  # Will be replaced below
                            'filter_shape': [[5, 5]],
                            'strides': [(1, 1)],
                            'mu': [0.0],
                            'sigma': [0.1],
                            'activation': [tf.nn.relu]}

        if 'n_output_channels' not in kwargs:

            target_n_output_channels = layer_size_to_absolute_value(layer_size=layer_size,
                                                                    small_value=6,
                                                                    huge_value=1000)

            # Distribute n_remaining_permutations linearly in region [0.9, 1.1] * target_n_output_channels
            n_output_channels_list = target_n_output_channels * get_linear_distribution_list(n_remaining_permutations)
            n_output_channels_list = np.round(n_output_channels_list).astype(np.int)
            n_output_channels_list = np.unique(n_output_channels_list)  # Remove any duplicates

            training_options['n_output_channels'] = n_output_channels_list

        training_options.update(kwargs)  # Replace with user defined.
        return (cls, training_options)


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
        self.str_repr = (f"{type(self).__name__}(name=\"{name}\", input_shape={input_shape}, ksize={ksize}, "
                         f"strides={strides}")

    def get_operation(self, x):
        x = tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding='VALID')
        return x

    @classmethod
    def get_training_options(cls, max_n_permutations=1, layer_size=0.0, **kwargs):
        """Get training options for this operation

        Arguments:
            max_n_permutations: Max number of ways to alternate the arguments.
            layer_size: Defines the expected size of this operation with an arbitrary
                        number in the range 0 to 1, where 0 is "small" and 1 is "huge".
            kwargs: user specified argument values to use.

        Returns:
            Tuple with object class and a dictionary with lists of values to try for each argument.
        """
        n_remaining_permutations = training_options_arg_helper(cls.__name__, max_n_permutations, layer_size,
                                                               **kwargs)
        training_options = {'ksize': None,
                            'strides': None}

        # Distribute the remaining permutations
        n_remaining_args = len(training_options) - len(kwargs)
        n_arg_variants = calc_n_argument_values(n_remaining_args, n_remaining_permutations)

        if 'ksize' not in kwargs:
            ksize_variants = [[1, 2, 2, 1], [1, 3, 3, 1], [1, 2, 3, 1], [1, 3, 2, 1], [1, 3, 3, 1]]
            training_options['ksize'] = ksize_variants[:n_arg_variants[0]]

        if 'strides' not in kwargs:
            strides_variants = [[1, 2, 2, 1], [1, 3, 3, 1], [1, 2, 3, 1], [1, 3, 2, 1], [1, 3, 3, 1]]
            training_options['strides'] = strides_variants[:n_arg_variants[1]]

        training_options.update(kwargs)  # Replace with user defined.
        return (cls, training_options)


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
        self.str_repr = f"{type(self).__name__}(name=\"{name}\", input_shape={input_shape}, keep_prob={keep_prob})"

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

    @classmethod
    def get_training_options(cls, max_n_permutations=1, layer_size=0.0, **kwargs):
        """Get training options for this operation

        Arguments:
            max_n_permutations: Max number of ways to alternate the arguments.
            layer_size: Not of interest for this operation.
            kwargs: user specified argument values to use.

        Returns:
            Tuple with object class and a dictionary with lists of values to try for each argument.
        """
        n_remaining_permutations = training_options_arg_helper(cls.__name__, max_n_permutations, layer_size,
                                                               **kwargs)
        training_options = {'keep_prob': None}

        # Distribute the remaining permutations
        n_remaining_args = len(training_options) - len(kwargs)
        n_arg_variants = calc_n_argument_values(n_remaining_args, n_remaining_permutations)
        if 'keep_prob' not in kwargs:
            keep_prob_variants = [0.5, 0.55, 0.45, 0.60, 0.40, 0.65, 0.35, 0.70, 0.75, 0.80, 0.85, 0.90]
            training_options['keep_prob'] = keep_prob_variants[:n_arg_variants[0]]

        training_options.update(kwargs)  # Replace with user defined.
        return (cls, training_options)


def layer_size_to_absolute_value(layer_size, small_value, huge_value, k=4.):
    """Calculates an absolute value for the arbitrary layer size.

    Based on definitions of what is a small and huge value, this method calculates
    an absolute value for the layer_size. An exponential function is applied for
    this since it seems reasonable to scale up more as the layer becomes larger.

    Arguments:
        layer_size {[type]} -- [description]
        small_value: expected output for layer_size = 0
        huge_value: expected output for layer_size = 1
    Keyword Arguments:
        k: exponential function curvature (default: {4.})

    Returns:
        value of the layer size in absolute numbers
    """
    a, b, k = get_exponential_function(y0=small_value, y1=huge_value, k=k)
    return a * math.exp(k * layer_size) + b


def get_exponential_function(y0, y1, k=4.):
    """Get an exponential function for layer size definition.

    Calculate coefficients for an exponential function on the form
    y = a * e^(k * x) + b, with y(0)=y0 and y(1)=y1.
    The value k controls the curvature, larger k gives more curve.

    Arguments:
        y0: expected output for x = 0
        y1: expected output for x = 1

    Keyword Arguments:
        k: curvature value (default: {4.})

    Returns:
        a, b, k in the equation y = a * e^(k * x) + b
    """
    a = (y0 - y1) / (1. - math.exp(k))
    b = y0 - a
    return a, b, k


def get_linear_distribution_list(n_values, target=1.0, max_deviation=0.1):
    """Calculates a list of values around the target value

    The values are evenly distributed in the range target ± max_deviation.
    """
    if n_values > 1:
        distribution_list = np.arange(target - max_deviation,
                                      target + max_deviation,
                                      (max_deviation * 2.0) / (n_values - 1))
    else:
        distribution_list = np.array([target])
    return distribution_list


def training_options_arg_helper(name, max_n_permutations, layer_size, **kwargs):
    """Assert and common calculations on input arguments.

    Returns:
        Number of remaining permutations to be distributed on other arguments.
    """
    assert(max_n_permutations >= 1)
    assert(layer_size >= 0.0)
    assert(layer_size <= 1.0)
    assert(all(isinstance(values, list) for _, values in kwargs.items()))

    n_input_permutations = calc_permutations_in_arg_dict(kwargs)
    n_remaining_permutations = math.floor(max_n_permutations / n_input_permutations)

    if n_remaining_permutations < 1.0:
        warn_msg = (f"{name} user defined number of training options have more permutations "
                    f"({n_input_permutations:.0f}) than max_n_permutations ({max_n_permutations}).")
        warnings.warn(warn_msg)
        n_remaining_permutations = 1.0  # Must be at least 1
    return n_remaining_permutations


def calc_permutations_in_arg_dict(arg_dict):
    n_permutations = 1.
    for _, values in arg_dict.items():
        n_permutations *= len(values)
    return n_permutations


def calc_n_argument_values(n_args, n_permutations):
    """Distribute a number of permutations among a number of arguments"""
    n_arg_values = np.zeros(n_args)
    for arg_idx in range(n_args, 0, -1):
        n_arg_values[arg_idx - 1] = math.floor(n_permutations**(1. / arg_idx))
        n_permutations /= n_arg_values[arg_idx - 1]

    return n_arg_values.astype(np.int)
