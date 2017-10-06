import numpy as np
import tensorflow as tf
import unittest

from src.NeuralNetworkOperations import *


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
