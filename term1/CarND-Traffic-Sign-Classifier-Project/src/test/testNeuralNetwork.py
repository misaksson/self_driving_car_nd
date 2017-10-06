import numpy as np
import tensorflow as tf
import unittest
import math

from src.NeuralNetwork import *
from src.NeuralNetworkOperations import NeuralNetworkOperation, MaxPoolOperation


class MockOperation(NeuralNetworkOperation):
    instances = []
    n_get_operation_calls = 0

    @classmethod
    def clear_mock(cls):
        cls.instances = []
        cls.n_get_operation_calls = 0

    def __init__(self, **kwargs):
        self.idx = len(MockOperation.instances)
        self.kwargs = kwargs
        self.n_prior_get_operation_calls = None

        weights = None if 'weights' not in kwargs else kwargs['weights']
        biases = None if 'biases' not in kwargs else kwargs['biases']
        NeuralNetworkOperation.__init__(self, name=kwargs['name'], weights=weights, biases=biases)
        MockOperation.instances.append(self)

    def get_operation(self, x):
        assert(self.n_prior_get_operation_calls is None)
        self.n_prior_get_operation_calls = MockOperation.n_get_operation_calls
        MockOperation.n_get_operation_calls += 1
        return x

    def get_feed_dict(self, training):
        if self.idx % 2 == 0:
            if training is True:
                return {self.kwargs['name']: "training mode"}
            elif training is False:
                return {self.kwargs['name']: "validation mode"}
        else:
            return dict()


class TestNearalNetwork(unittest.TestCase):
    def setUp(self):
        MockOperation.clear_mock()

    def test_should_generate_names(self):
        """NeuralNetwork should generate names when not given.

        The second operation will provide it's own name, the other two should
        get generated names.
        """
        expected_names = ["op_1_MockOperation", "given_name", "op_3_MockOperation"]

        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        nn = NeuralNetwork(x)
        nn.add(MockOperation)
        nn.add(MockOperation, name="given_name")
        nn.add(MockOperation)

        self.assertEqual(len(MockOperation.instances), len(expected_names))

        for i, (obj, expected) in enumerate(zip(MockOperation.instances, expected_names)):
            self.assertEqual(obj.kwargs['name'], expected)

    def test_should_provide_input_shape(self):
        """NeuralNetwork should provide input shape to each operation.

        Use MockOperations to extract input_shape before and after a MaxPoolOperation
        down sampling the input by a factor 2 in all dimensions.
        """
        expected_shapes = [[32, 32, 2], [16, 16, 1]]
        x = tf.placeholder(tf.float32, (None, 32, 32, 2))
        nn = NeuralNetwork(x)
        nn.add(MockOperation)
        nn.add(MaxPoolOperation, ksize=[1, 2, 2, 2], strides=[1, 2, 2, 1])
        nn.add(MockOperation)
        self.assertEqual(len(MockOperation.instances), len(expected_shapes))

        for i, (obj, expected) in enumerate(zip(MockOperation.instances, expected_shapes)):
            self.assertEqual(obj.kwargs['input_shape'], expected)

    def test_should_get_all_operations(self):
        """NeuralNetwork should apply operations in the same order as added.

        The mocked get_operation method stores the total number of prior calls
        made to this method in other instances.
        """
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        nn = NeuralNetwork(x)
        nn.add(MockOperation)
        nn.add(MockOperation)
        nn.add(MockOperation)
        self.assertEqual(len(MockOperation.instances), 3)

        for i, obj in enumerate(MockOperation.instances):
            self.assertEqual(obj.n_prior_get_operation_calls, i)

    def test_should_calculate_loss_regularizer(self):
        """NeuralNetwork should calculate L2 loss regularizer.

        Add some MockOperations with weights, then calculate the L2 loss regularizer for the
        network and compare it to reference.
        """
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        nn = NeuralNetwork(x)
        nn.add(MockOperation, weights=tf.constant(np.arange(0, 5).astype(np.float32)))
        nn.add(MockOperation, weights=tf.constant(np.arange(5, 10).astype(np.float32)))
        nn.add(MockOperation, weights=tf.constant(np.arange(10, 15).astype(np.float32)))
        self.assertEqual(len(MockOperation.instances), 3)

        with tf.Session() as sess:
            actual = sess.run(nn.get_loss_regularizer())
            expected = sess.run(tf.nn.l2_loss(tf.constant(np.arange(0, 15).astype(np.float32))))

            self.assertEqual(actual, expected)

    def test_should_get_all_feed_dict(self):
        """NeuralNetwork should concatenate feed_dict from all operations.

        Every second instance of the mock returns a dictionary of length 1, where
        the value is a string replicating training or validation. The remaining
        mock instances return empty dictionary.
        """
        x = tf.placeholder(tf.float32, (None, 32, 32, 1))
        nn = NeuralNetwork(x)
        nn.add(MockOperation)
        nn.add(MockOperation)
        nn.add(MockOperation)
        self.assertEqual(len(MockOperation.instances), 3)

        test_vector = [
            (True, 'training mode'),
            (False, 'validation mode')
        ]
        expected_feed_dict_length = math.ceil(len(MockOperation.instances) / 2.)

        for training_arg, expected_mode in test_vector:
            feed_dict = nn.get_feed_dict(training=training_arg)
            self.assertEqual(len(feed_dict), expected_feed_dict_length)
            for _, actual_mode in feed_dict.items():
                self.assertEqual(actual_mode, expected_mode)


if __name__ == '__main__':
    unittest.main()
