import itertools
from NeuralNetwork import NeuralNetwork
from NeuralNetworkOperations import *


class NeuralNetworkGenerator(object):
    def __init__(self, n_classes, tuning_options_list, image_shape):
        self.n_classes = n_classes
        self.image_shape = image_shape
        self._calc_all_permutations(tuning_options_list)

    def _calc_all_permutations(self, tuning_options_list):
        op_list = []  # All operations
        op_arg_names_list = []  # List of argument names for each operation.
        op_arg_value_permutation_list = []  # List of all argument value permutations for each operation.
        for operation, argument_dict in tuning_options_list:
            op_list.append(operation)
            op_arg_names_list.append(list(argument_dict.keys()))
            arg_values = list(argument_dict.values())
            arg_permutations = itertools.product(*arg_values)
            op_arg_value_permutation_list.append(arg_permutations)

        self.op_list = op_list
        self.op_arg_names_list = op_arg_names_list
        self.op_arg_value_iterator = itertools.product(*op_arg_value_permutation_list)

    def generate(self):
        for op_arg_values in self.op_arg_value_iterator:

            # Initialize x and y every training because of tensor reset.
            x = tf.placeholder(tf.float32, (None,
                                            self.image_shape[0],
                                            self.image_shape[1],
                                            self.image_shape[2]))
            y = tf.placeholder(tf.int32, (None))

            nn = NeuralNetwork(x)
            for operation, arg_names, arg_values in zip(self.op_list, self.op_arg_names_list, op_arg_values):
                arg_dict = dict(zip(arg_names, arg_values))
                nn.add(operation, **arg_dict)

            # Non optional output layer
            nn.add(DenseOperation, n_output_channels=self.n_classes, activation=None)

            yield nn, x, y

            # Reset tensor before each training.
            tf.reset_default_graph()
