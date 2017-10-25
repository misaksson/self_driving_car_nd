import numpy as np
import tensorflow as tf
import unittest
import itertools
import warnings

from NeuralNetworkOperations import *


class TestNeuralNetworkOperation(unittest.TestCase):
    def test_get_weights(self):
        obj = NeuralNetworkOperation(name='', weights=42, biases=1)
        self.assertEqual(obj.get_weights(), 42)


class TestDenseOperation(unittest.TestCase):
    def test_forward_operation(self):
        """Dense operation should apply linear function

        Use weight and biases from initialized object to calculate reference.
        No activation operation is used.
        """
        in_data = np.random.randn(16)
        in_data = np.reshape(in_data, (1, 16))
        x = tf.placeholder(tf.float32, in_data.shape)
        obj = DenseOperation(name='test_instance', input_shape=[i for i in in_data.shape[1:]],
                             n_output_channels=16, activation=None)

        logits = obj.get_operation(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Calculate reference result
            biases, weights = sess.run([obj.get_biases(), obj.get_weights()])
            expected_result = np.matmul(in_data[0], weights) + biases
            actual_result = sess.run(logits, feed_dict={x: in_data})

            self.assertTrue(np.allclose(actual_result, expected_result))

    def test_forward_operation_with_activation(self):
        """Dense operation should apply linear operation and then relu activation.

        Use weight and biases from initialized object and calculate reference.
        """
        in_data = np.random.randn(16)
        in_data = np.reshape(in_data, (1, 16))
        x = tf.placeholder(tf.float32, in_data.shape)
        obj = DenseOperation(name='test_instance', input_shape=[i for i in in_data.shape[1:]],
                             n_output_channels=16)

        logits = obj.get_operation(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Calculate reference result
            biases, weights = sess.run([obj.get_biases(), obj.get_weights()])
            expected_result = (np.matmul(in_data[0], weights) + biases)  # Linear function
            expected_result[expected_result < 0.0] = 0.0  # Relu activation

            actual_result = sess.run(logits, feed_dict={x: in_data})
            self.assertTrue(np.allclose(actual_result, expected_result))

    def test_should_flatten_input(self):
        """Dense operation should flatten the input image.

        This operation should handle both 1 and 2 dimensional inputs. Use same
        setup as above but reshape input to 4x4 array.
        """
        in_data_flat = np.random.randn(16)
        in_data = np.reshape(in_data_flat, (1, 4, 4, 1))
        x = tf.placeholder(tf.float32, in_data.shape)
        obj = DenseOperation(name='test_instance', input_shape=[i for i in in_data.shape[1:]],
                             n_output_channels=16, activation=None)

        logits = obj.get_operation(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # Calculate reference result
            biases, weights = sess.run([obj.get_biases(), obj.get_weights()])
            expected_result = np.matmul(in_data_flat, weights) + biases

            actual_result = sess.run(logits, feed_dict={x: in_data})
            self.assertTrue(np.allclose(actual_result, expected_result))

    def test_should_provide_valid_training_options(self):
        common_verification_of_training_options(self, DenseOperation)

    def test_should_override_training_options(self):
        _, default = DenseOperation.get_training_options(max_n_permutations=15, layer_size=0.5)
        _, overrided = DenseOperation.get_training_options(max_n_permutations=15, layer_size=0.5,
                                                           n_output_channels=[42, 32, 22, 11])

        self.assertEqual(len(default['n_output_channels']), 15)
        self.assertEqual(len(overrided['n_output_channels']), 4)

    def test_should_not_add_more_training_options(self):
        _, training_options = DenseOperation.get_training_options(max_n_permutations=5, layer_size=0.5,
                                                                  mu=[-0.2, -0.1, 0.0, 0.1, 0.2])
        for key, values in training_options.items():
            if key is 'mu':
                self.assertEqual(len(values), 5)
            else:
                self.assertEqual(len(values), 1)

    def test_should_warn_if_user_breaks_max_n_permutations(self):
        with warnings.catch_warnings(record=True) as w:
            _, training_options = DenseOperation.get_training_options(max_n_permutations=1, layer_size=0.5,
                                                                      mu=[-0.2, -0.1, 0.0, 0.1, 0.2])
            self.assertEqual(len(w), 1)
            self.assertTrue("user defined number of training options have more" in str(w[-1].message))

            # But it should still work as user has defined
            for key, values in training_options.items():
                if key is 'mu':
                    self.assertEqual(len(values), 5)
                else:
                    self.assertEqual(len(values), 1)


class TestConv2dOperation(unittest.TestCase):
    def test_forward_operation(self):
        """Conv2d operation should apply weight filter.

        The weights are initialized to mu=42 and 0 stddev, making it possible to
        know the expected result. No activation operation is used.
        """
        in_data = np.array([[1, 0, 2, 3],
                            [4, 6, 6, 8],
                            [3, 1, 1, 0],
                            [1, 2, 2, 4]])
        in_data = np.reshape(in_data, (1, 4, 4, 1))
        weights_value = 42.0  # Initialize all weights to this value
        expected_result = np.array([[[11], [19]],
                                    [[7], [7]]]) * weights_value

        x = tf.placeholder(tf.float32, in_data.shape)
        obj = Conv2dOperation(name='test_instance', input_shape=[i for i in in_data.shape[1:]],
                              n_output_channels=1, filter_shape=(2, 2), strides=(2, 2), mu=weights_value,
                              sigma=0.0, activation=None)
        logits = obj.get_operation(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual_result = sess.run(logits, feed_dict={x: in_data})
            self.assertTrue(np.allclose(actual_result, expected_result))

    def test_forward_operation_with_activation(self):
        """Conv2d operation should apply weight filter and then relu activation.

        Top-left and bottom-right pixel should get at negative value after filter,
        making the relu activation function output 0.
        """
        in_data = np.array([[1, 0, 2, 3],
                            [4, -6, 6, 8],
                            [3, 1, 1, 0],
                            [1, 2, 2, -4]])
        in_data = np.reshape(in_data, (1, 4, 4, 1))
        weights_value = 42.0  # Initialize all weights to this value
        expected_result = np.array([[[0], [19]],
                                    [[7], [0]]]) * weights_value

        x = tf.placeholder(tf.float32, in_data.shape)
        obj = Conv2dOperation(name='test_instance', input_shape=[i for i in in_data.shape[1:]],
                              n_output_channels=1, filter_shape=(2, 2), strides=(2, 2), mu=weights_value,
                              sigma=0.0)
        logits = obj.get_operation(x)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            actual_result = sess.run(logits, feed_dict={x: in_data})
            self.assertTrue(np.allclose(actual_result, expected_result))

    def test_should_provide_valid_training_options(self):
        common_verification_of_training_options(self, Conv2dOperation)


class TestMaxPoolOperation(unittest.TestCase):
    def test_should_downsample_example(self):
        """Max pool should down sample the example image.

        Use default ksize and strides to downsample the image by factor of 2.
        Verify that the max pixel values are extracted by comparing with
        reference.
        """
        in_data = np.array([[1, 0, 2, 3],
                            [4, 6, 6, 8],
                            [3, 1, 1, 0],
                            [1, 2, 2, 4]])
        in_data = np.reshape(in_data, (1, 4, 4, 1))
        expected_result = np.array([[[6], [8]],
                                    [[3], [4]]])
        x = tf.placeholder(tf.float32, in_data.shape)

        obj = MaxPoolOperation(name='test_instance', input_shape=[i for i in in_data.shape[1:]])
        logits = obj.get_operation(x)

        with tf.Session() as sess:
            actual_result = sess.run(logits, feed_dict={x: in_data})
            self.assertTrue(np.allclose(actual_result, expected_result))

    def test_should_provide_valid_training_options(self):
        common_verification_of_training_options(self, MaxPoolOperation)


class TestDropoutOperation(unittest.TestCase):
    def test_should_dropout(self):
        """Dropout should remove elements.

        Use different keep_prob values and verify that the result is reasonable.
        """
        in_data = np.array(np.random.randn(1000))
        x = tf.placeholder(tf.float32, len(in_data))

        for keep_prob in np.arange(0.1, 1.1, 0.1):
            obj = DropoutOperation(name='test_instance', input_shape=None, keep_prob=keep_prob)
            logits = obj.get_operation(x)
            feed_dict = obj.get_feed_dict(training=True)
            feed_dict.update({x: in_data})

            with tf.Session() as sess:
                result = sess.run(logits, feed_dict=feed_dict)

                # The actual keep rate not deviate too much from keep probability
                keep_rate = (np.count_nonzero(result) / len(result))
                self.assertLess(np.abs(keep_prob - keep_rate), 0.05)

                # The accumulated sums should be roughly the same.
                abs_sum_input = np.abs(in_data).sum()
                abs_sum_result = np.abs(result).sum()
                self.assertLess(np.abs(abs_sum_input - abs_sum_result),
                                (abs_sum_input * 0.4))

    def test_should_not_dropout(self):
        """Dropout should only be active when training.

        Set training argument to False in call of get_feed_dict(), then verifies
        that the result is the same as input.
        """
        in_data = np.array(np.random.randn(1000))
        x = tf.placeholder(tf.float32, len(in_data))

        obj = DropoutOperation(name='test_instance', input_shape=None, keep_prob=0.5)
        logits = obj.get_operation(x)
        feed_dict = obj.get_feed_dict(training=False)
        feed_dict.update({x: in_data})

        with tf.Session() as sess:
            result = sess.run(logits, feed_dict=feed_dict)

            # result is close, but no cigar
            self.assertTrue(np.allclose(result, in_data))

    def test_should_provide_valid_training_options(self):
        common_verification_of_training_options(self, DropoutOperation)


class TestExponentialFunction(unittest.TestCase):
    def test_should_intercept_points(self):
        """The exponential function should intercept points y(0) and y(1)"""
        test_vector = [
            (50., 10000000),
            (50., -10000000),
            (100., 40.),
            (100000., 10000000.),
            (50., 50.)
        ]
        for y0, y1 in test_vector:
            a, b, k = get_exponential_function(y0, y1)
            actual_y0 = a * math.exp(k * 0) + b
            actual_y1 = a * math.exp(k * 1) + b
            self.assertAlmostEqual(actual_y0, y0)
            self.assertAlmostEqual(actual_y1, y1)

    def test_should_intercept_points_for_all_k(self):
        """The exponential function should intercept points regardless of value of k"""
        y0 = 50.
        y1 = 10000000.
        for k in np.arange(1., 10., 1.):
            a, b, _ = get_exponential_function(y0, y1, k=k)
            actual_y0 = a * math.exp(k * 0) + b
            actual_y1 = a * math.exp(k * 1) + b
            self.assertAlmostEqual(actual_y0, y0)
            self.assertAlmostEqual(actual_y1, y1)


class TestLayerSizeMapping(unittest.TestCase):
    def test_medium_value_should_decrease_with_larger_k(self):
        """The exponential function should have more curvature for larger k"""
        small_value = 50.
        huge_value = 10000000.
        medium_layer_size = 0.5
        prev_value = math.inf
        for k in np.arange(1., 10., 1.):
            medium_value = layer_size_to_absolute_value(medium_layer_size, small_value, huge_value, k=k)
            self.assertLess(medium_value, prev_value)


class TestLinearDistribution(unittest.TestCase):
    def test_distributions(self):
        test_vector = [
            (1, [1.0]),
            (2, [0.9, 1.1]),
            (3, [0.9, 1.0, 1.1]),
            (4, [0.9, 0.967, 1.033, 1.1]),
            (5, [0.9, 0.95, 1.0, 1.05, 1.1])
        ]
        for n_values, expected in test_vector:
            actual = get_linear_distribution_list(n_values=n_values, target=1.0, max_deviation=0.1)
            self.assertTrue(np.allclose(actual, expected, rtol=1e-2))


class TestPermutationDistribution(unittest.TestCase):
    def test_calc_n_argument_values(self):

        test_vector = [(1, [(1, [1]), (2, [1, 1]), (3, [1, 1, 1]), (4, [1, 1, 1, 1])]),
                       (2, [(1, [2]), (2, [2, 1]), (3, [2, 1, 1]), (4, [2, 1, 1, 1])]),
                       (25, [(1, [25]), (2, [5, 5]), (3, [4, 3, 2]), (4, [3, 2, 2, 2])]),
                       (27, [(1, [27]), (2, [5, 5]), (3, [3, 3, 3]), (4, [3, 2, 2, 2])])]
        for n_permutations, args_expected_list in test_vector:
            for n_args, expected in args_expected_list:
                actual = calc_n_argument_values(n_args, n_permutations)
                self.assertTrue(np.array_equal(actual, expected))


def common_verification_of_training_options(self, operation):
    for max_n_permutations in range(1, 10):
        (operation, tuning_options) = operation.get_training_options(max_n_permutations, layer_size=0.5)

        key_list = []
        values_list = []
        actual_n_permutations = 1
        for key, values in tuning_options.items():
            key_list.append(key)
            values_list.append(values)
            actual_n_permutations *= len(values)

        self.assertLessEqual(actual_n_permutations, max_n_permutations)

        # Verify that the operation can be initialized with each permutation.
        for arg_values in itertools.product(*values_list):
                args = dict(zip(key_list, arg_values))
                operation(name='dc', input_shape=[32, 32, 3], **args)
