"""
Setup helper for a neural network

This module makes it easier to setup and use a list of NeuralNetworkOperations.

"""

import tensorflow as tf


class NeuralNetwork(object):
    """Implements a list of NeuralNetworkOperation's

    It also helps to extract and calculate common data such as loss regulizer
    and training resp. validation entries for the feed_dict.
    """

    def __init__(self, x):
        """
        Arguments:
            x: input tensor
        """
        self.x = x
        self.operations = []

    def _generate_name(self, nn_operation):
        return f"op_{len(self.operations) + 1}_{nn_operation.__name__}"

    def add(self, nn_operation, name=None, **kwargs):
        """Add NeuralNetworkOperation to list

        Initialize and append the NeuralNetworkOperation to the list. Propagate
        the tensor x trough the new tensorflow operation(s).

        Arguments:
            nn_operation: An operation derived from the NeuralNetworkOperation class
            **kwargs: any additional arguments for the nn_operation

        Keyword Arguments:
            name: Name used for related tensors. Will be generated if None. (default: {None})
        """
        # Generate name if not given
        if name is None:
            name = self._generate_name(nn_operation)

        # Get shape from tensor object
        input_shape = self.x.get_shape().as_list()[1:]

        # Initialize the operation
        operation = nn_operation(name=name, input_shape=input_shape, **kwargs)

        # Apply operation on tensor
        self.x = operation.get_operation(self.x)

        # Store operation instance
        self.operations.append(operation)

    def get_logits(self):
        """Get the logits of the neural network.

        Call this method to get the final logits when done setting up the network.

        Returns:
            tensor logits
        """
        return self.x

    def get_loss_regularizer(self):
        """Calculates the L2 loss regularizer for the network.

        Get weights from all operations in the list, and calculate the
        accumulated L2 loss regularizer.

        Returns:
            L2 loss regularizer
        """
        loss_regularizer = 0
        for operation in self.operations:
            weights = operation.get_weights()
            if weights is not None:
                loss_regularizer += tf.nn.l2_loss(weights)
        return loss_regularizer

    def get_feed_dict(self, training):
        """Get all feed_dict additions from the operations.

        Concatenates the feed_dict additions from all operations in the list.

        Arguments:
            training: Set to True when training, otherwise False (e.g. for validation).

        Returns:
            Additional entries for the feed_dict.
        """
        feed_dict = dict()
        for operation in self.operations:
            feed_dict.update(operation.get_feed_dict(training))
        return feed_dict

    def __str__(self):
        """Get string representation for all operations"""
        str_repr = ""
        for operation in self.operations:
            str_repr += operation.__str__() + "\n"
        return str_repr
